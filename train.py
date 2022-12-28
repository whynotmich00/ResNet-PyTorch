# Copyright 2022 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import os
import time
from glob import glob
import numpy as np
from datetime import datetime
import random

import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from warm import WarmupLR
import config
import model
from dataset import CUDAPrefetcher, ImageDataset
from utils import accuracy, load_state_dict, make_directory, save_checkpoint, Summary, AverageMeter, ProgressMeter

model_names = sorted(
    name for name in model.__dict__ if name.islower() and not name.startswith("__") and callable(model.__dict__[name]))

def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training network evaluation indicators
    best_acc1 = 0.0
    for ind in range(config.k):
        train_prefetcher, valid_prefetcher = load_dataset(ind)
        print(f"Load `{config.model_arch_name}` datasets successfully.")

        resnet_model = build_model() #, ema_resnet_model
        print(f"Build `{config.model_arch_name}` model successfully.")

        pixel_criterion = define_loss()
        print("Define all loss functions successfully.")

        optimizer = define_optimizer(resnet_model)
        print("Define all optimizer functions successfully.")

        scheduler = define_scheduler(optimizer,config.warmup)
        print("Define all optimizer scheduler functions successfully.")

        print("Check whether to load pretrained model weights...")
        if config.pretrained_model_weights_path:
            resnet_model,  start_epoch, best_acc1, optimizer, scheduler = load_state_dict(
                resnet_model,
                config.pretrained_model_weights_path,
                start_epoch,
                best_acc1,
                optimizer,
                scheduler)
            print(f"Loaded `{config.pretrained_model_weights_path}` pretrained model weights successfully.")
        else:
            print("Pretrained model weights not found.")

        print("Check whether the pretrained model is restored...")
        if config.resume:
            resnet_model, start_epoch, best_acc1, optimizer, scheduler = load_state_dict(
                resnet_model,
                config.pretrained_model_weights_path,
                start_epoch,
                best_acc1,
                optimizer,
                scheduler,
                "resume")
            print("Loaded pretrained generator model weights.")
        else:
            print("Resume training model not found. Start training from scratch.")

        # Create a experiment results
        samples_dir = os.path.join("samples", config.exp_name)
        results_dir = os.path.join("results", config.exp_name)
        make_directory(samples_dir)
        make_directory(results_dir)
        name = '_warmup' if config.warmup else ''
        reg = str(l2_lambda) if config.regularization else ''
        now = datetime.now().strftime('%y_%m_%d__%H_%M_%S')
        # Create training process log file
        writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name+reg+name,now))

        # Initialize the gradient scaler
        # scaler = amp.GradScaler()
        for epoch in range(start_epoch, config.epochs):
            scheduler.step()
            print('epoch: ',epoch,' ','lr: ',optimizer.param_groups[0]['lr'])
            
            train(resnet_model, train_prefetcher, pixel_criterion, optimizer, epoch, writer) 
            acc1 = validate(resnet_model, valid_prefetcher, epoch, writer, "Valid")
            print("\n")
            # write weights distrib
            if epoch % 10 == 0:
                with torch.no_grad():
                    for name, weights in resnet_model.named_parameters():
                        writer.add_histogram(f'{name} weights distr', weights.reshape(-1), epoch + 1)
                
            # Automatically save the model with the highest index
            is_best = acc1 > best_acc1
            is_last = (epoch + 1) == config.epochs
            best_acc1 = max(acc1, best_acc1)
            save_checkpoint({"epoch": epoch + 1,
                            "best_acc1": best_acc1,
                            "state_dict": resnet_model.state_dict(),
                            #"ema_state_dict": ema_resnet_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict()},
                            f"epoch_{epoch + 1}.pth.tar",
                            samples_dir,
                            results_dir,
                            is_best,
                            is_last)
        writer.close()
        
def load_dataset(ind) -> CUDAPrefetcher:
    # Load train, test and valid datasets
    paths = glob(f"{config.image_dir}/*/*")
    random.shuffle(paths)
    valid_file_paths = paths[10_000*ind:10_000+10_000*ind]
    train_file_paths = [i for i in paths if i not in valid_file_paths]
    train_dataset = ImageDataset(config.image_size,
                                 train_file_paths,
                                 config.model_mean_parameters,
                                 config.model_std_parameters,
                                 "Train")
    valid_dataset = ImageDataset(config.image_size,
                                 valid_file_paths,
                                 config.model_mean_parameters,
                                 config.model_std_parameters,
                                 "Valid")

    # Generator all dataloader
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=False,
                                  num_workers=config.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  persistent_workers=True)

    # Place all data on the preprocessing data loader
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)

    return train_prefetcher, valid_prefetcher


def build_model() -> nn.Module:
    resnet_model = model.__dict__[config.model_arch_name](num_classes=config.model_num_classes)
    resnet_model = resnet_model.to(device=config.device, memory_format=torch.channels_last)
    
    return resnet_model


def define_loss() -> nn.CrossEntropyLoss:
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device=config.device, memory_format=torch.channels_last)

    return criterion


def define_optimizer(model) -> optim.SGD:
    optimizer = optim.SGD(model.parameters(),
                          lr=config.model_lr,
                          momentum=config.model_momentum,
                          weight_decay=config.model_weight_decay)

    return optimizer



def define_scheduler(optimizer: optim.SGD,warmup: bool=False) -> lr_scheduler.MultiStepLR:
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                        milestones= [config.primo_drop, 
                                                     config.secondo_drop]
                                        )
    if warmup:
        scheduler = WarmupLR(scheduler, init_lr=0.01, num_warmup=config.warmup_drop, warmup_strategy='constant')

    return scheduler

def train(
        model: nn.Module,
        train_prefetcher: CUDAPrefetcher,
        criterion: nn.functional.cross_entropy,
        optimizer: optim.SGD,
        epoch: int,
        # scaler: amp.GradScaler,
        writer: SummaryWriter
) -> None:
    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    acc1 = AverageMeter("Acc@1", ":6.2f")
    # acc5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time, losses, acc1],
                             prefix=f"Epoch: [{epoch + 1}]")

    # Put the generative network model in training mode
    model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        # Transfer in-memory data to CUDA devices to speed up training
        images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        target = batch_data["target"].to(device=config.device, non_blocking=True)

        # Get batch size
        batch_size = images.size(0)

        # Initialize generator gradients
        optimizer.zero_grad()

        # Mixed precision training
        # with amp.autocast():
        output = model(images/255)
        if config.regularization:
            l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        else:
            l2_norm = 0
        loss = config.loss_weights * criterion(output, target) + config.l2_lambda * l2_norm

        # Backpropagation
        loss.backward()
        # scaler.scale(loss).backward()
        # update generator weights
        # scaler.step(optimizer)
        optimizer.step()
        
        # scaler.update()

        # measure accuracy and record loss
        top1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), batch_size)
        acc1.update(top1[0].item(), batch_size)
        # acc5.update(top5[0].item(), batch_size)
        
        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.train_print_frequency == 0:

            # Record loss during training and output to file
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            writer.add_scalar("Train/Accuracy", acc1.avg, batch_index + epoch * batches + 1)
            writer.add_scalar("Train/Errors", 1-acc1.avg/100, batch_index + epoch * batches + 1)
            
            progress.display(batch_index + 1)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # Add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1


def validate(
        model: nn.Module,
        data_prefetcher: CUDAPrefetcher,
        epoch: int,
        writer: SummaryWriter,
        mode: str
) -> float:
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f", Summary.NONE)
    acc1 = AverageMeter("Acc@1", ":6.2f", Summary.AVERAGE)
    # acc5 = AverageMeter("Acc@5", ":6.2f", Summary.AVERAGE)
    progress = ProgressMeter(batches, [batch_time, acc1], prefix=f"{mode}: ")

    # Put the exponential moving average model in the verification mode
    model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            # Transfer in-memory data to CUDA devices to speed up training
            images = batch_data["image"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            target = batch_data["target"].to(device=config.device, non_blocking=True)

            # Get batch size
            batch_size = images.size(0)

            # Inference
            output = model(images)
            

            # measure accuracy and record loss
            top1 = accuracy(output, target, topk=(1,))
            acc1.update(top1[0].item(), batch_size)
            # acc5.update(top5[0].item(), batch_size)

            # Calculate the time it takes to fully train a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Write the data during training to the training log file
            if batch_index % config.valid_print_frequency == 0:
                writer.add_scalar("Validation/Accuracy" ,acc1.avg , epoch + 1)
                writer.add_scalar("Validation/Errors" ,1-acc1.avg/100 ,epoch + 1)
                progress.display(batch_index + 1)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # Add 1 to the number of data batches to ensure that the terminal prints data normally
            batch_index += 1

    # print metrics
    progress.display_summary()

    return acc1.avg


if __name__ == "__main__":
    main()
