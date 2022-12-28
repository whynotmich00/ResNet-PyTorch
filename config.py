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
import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Model arch name
model_arch_name = "resnet20_bottleneck_v1"
warmup = True
regularization = False
k = 1
# Model normalization parameters
model_mean_parameters = [0.49139968, 0.48215845, 0.4465309]
model_std_parameters = [1 , 1, 1] #[0.12835708, 0.12578596, 0.15331733]
# Model number class
model_num_classes = 10
# Current configuration parameter method
mode = "train"
# Experiment name, easy to save weights and log files
n = "_regularized" if regularization else ""
exp_name = f"{model_arch_name}-CIFAR10" + n

if mode == "train":
    # Dataset address
    image_dir = "./data/CIFAR10/CIFAR10_img_train"

    image_size = 36
    batch_size = 128
    num_workers = 4

    # The address to load the pretrained model
    pretrained_model_weights_path = None #"./results/pretrained_models/ResNet18-CIFAR10-57bb63e.pth.tar"

    # Incremental training and migration training
    resume = ""

    # Total num epochs
        # 164 per 50000 imgs, 205 per 40000 (corrispondono piu o meno a 64000 steps)
    epochs = 207 if warmup else 205 # prime 2 epoche di warm-up

    # Loss parameters
    # loss_label_smoothing = 0 #.1
    loss_weights = 1.0
    l2_lambda = 2e-4 if regularization else 0 # 1e-4, 2e-4, 3e-4

    # Optimizer parameter
    model_lr = 0.1
    model_momentum = 0.9
    model_weight_decay = 1e-04
    
    # Scheduler parameters
    warmup_drop = 2 # about 800 steps
    primo_drop = 102 # 82 epoche: 32000 steps (batch_size = 128 e dataset 50000 elementi)
                    # 102 epoche: 32000 steps (batch_size = 128 e dataset 40000 elementi)
    secondo_drop = 138 # 123  epoche: 48000 steps (batch_size = 128 e dataset 50000 elementi) 
                    # 138 epoche: 48000 steps (batch_size = 128 e dataset 40000 elementi)

    # How many iterations to print the training/validate result
    train_print_frequency = 200
    valid_print_frequency = 20

if mode == "test":
    # Test data address
    test_image_dir = "./data/CIFAR10/CIFAR10_img_test"

    # Test dataloader parameters
    image_size = 32
    batch_size = 128
    num_workers = 4

    # How many iterations to print the testing result
    test_print_frequency = 20

    model_weights_path = "./results/pretrained_models/resnet56_last.pth.tar"
