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
from typing import Any, List, Type, Union, Optional

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "ResNet",
    "resnet20_basicblock_v1",
"resnet32_basicblock_v1",
"resnet44_basicblock_v1",
"resnet56_basicblock_v1",
"resnet110_basicblock_v1",
"resnet20_bottleneck_v1",
"resnet32_bottleneck_v1",
"resnet44_bottleneck_v1",
"resnet56_bottleneck_v1",
"resnet110_bottleneck_v1",
"resnet20_basicblock_v2",
"resnet32_basicblock_v2",
"resnet44_basicblock_v2",
"resnet56_basicblock_v2",
"resnet110_basicblock_v2",
"resnet20_bottleneck_v2",
"resnet32_bottleneck_v2",
"resnet44_bottleneck_v2",
"resnet56_bottleneck_v2",
"resnet110_bottleneck_v2" ]


class _BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_channels: int = 16,
            shortcut: str = "identity",
    ) -> None:
        super(_BasicBlock, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels
        self.shortcut = shortcut

        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (stride, stride), (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            if self.shortcut == "identity":
                shape = identity.shape[2]//4            
                identity = self.downsample(x)[:,:, shape:shape*3, shape:shape*3]
            elif self.shortcut == "projection":
                identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class _Bottleneck_v1(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_channels: int = 16,
            shortcut: str = "identity",

    ) -> None:
        super(_Bottleneck_v1, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels
        self.shortcut = shortcut

        channels = int(out_channels * (base_channels / 16.0)) * groups

        self.conv1 = nn.Conv2d(in_channels, channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (stride, stride), (1, 1), groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, int(out_channels * self.expansion), (1, 1), (1, 1), (0, 0), bias=False)
        self.bn3 = nn.BatchNorm2d(int(out_channels * self.expansion))
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            if self.shortcut == "identity":
                shape = identity.shape[2]//4            
                identity = self.downsample(x)[:,:, shape:shape*3, shape:shape*3]
            elif self.shortcut == "projection":
                identity = self.downsample(x)

        out = torch.add(out, identity)
        out = self.relu(out)

        return out

class _BasicBlock_v2(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_channels: int = 16,
            shortcut: str = "identity",
    ) -> None:
        super(_BasicBlock_v2, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels
        self.shortcut = shortcut
        
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, (3, 3), (stride, stride), (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, (3, 3), (1, 1), (1, 1), groups=groups, bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            if self.shortcut == "identity":
                shape = identity.shape[2]//4            
                identity = self.downsample(x)[:,:, shape:shape*3, shape:shape*3]
            elif self.shortcut == "projection":
                identity = self.downsample(x)


        out = torch.add(out, identity)

        return out


class _Bottleneck_v2(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_channels: int = 16,
            shortcut: str = "identity",
    ) -> None:
        super(_Bottleneck_v2, self).__init__()
        self.stride = stride
        self.downsample = downsample
        self.groups = groups
        self.base_channels = base_channels
        self.shortcut = shortcut

        channels = int(out_channels * (base_channels / 16.0)) * groups

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, channels, (1, 1), (1, 1), (0, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, (3, 3), (stride, stride), (1, 1), groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, int(out_channels * self.expansion), (1, 1), (1, 1), (0, 0), bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            if self.shortcut == "identity":
                shape = identity.shape[2]//4            
                identity = self.downsample(x)[:,:, shape:shape*3, shape:shape*3]
            elif self.shortcut == "projection":
                identity = self.downsample(x)


        out = torch.add(out, identity)
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            arch_cfg: List[int],
            block: Type[Union[_BasicBlock, _Bottleneck_v1, _BasicBlock_v2, _Bottleneck_v2]],
            groups: int = 1,
            channels_per_group: int = 16,
            num_classes: int = 10,
            shortcut: str = "identity",
    ) -> None:
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.dilation = 1
        self.groups = groups
        self.base_channels = channels_per_group

        self.conv1 = nn.Conv2d(3, self.in_channels, (7, 7), (2, 2), (3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(True)
        self.maxpool = nn.MaxPool2d((3, 3), (2, 2), (1, 1))

        self.layer1 = self._make_layer(arch_cfg[0], block, 16, 1, shortcut)
        self.layer2 = self._make_layer(arch_cfg[1], block, 32, 2, shortcut)
        self.layer3 = self._make_layer(arch_cfg[2], block, 64, 2, shortcut)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(64 * block.expansion, num_classes)

        # Initialize neural network weights
        self._initialize_weights()

    def _make_layer(
            self,
            repeat_times: int,
            block: Type[Union[_BasicBlock, _Bottleneck_v1, _BasicBlock_v2, _Bottleneck_v2]],
            channels: int,
            stride: int = 1,
            shortcut: str = "identity",
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.in_channels != channels * block.expansion:
            if shortcut == "projection":
                downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, channels * block.expansion, (1, 1), (stride, stride), (0, 0), bias=False),
                    nn.BatchNorm2d(channels * block.expansion),
                )
            elif shortcut == "identity":
                downsample = nn.ZeroPad2d([ 0, 0, 0, 0, channels//block.expansion, channels//block.expansion])
                

        layers = [
            block(
                self.in_channels,
                channels,
                stride,
                downsample,
                self.groups,
                self.base_channels
            )
        ]
        self.in_channels = channels * block.expansion
        for _ in range(1, repeat_times):
            layers.append(
                block(
                    self.in_channels,
                    channels,
                    1,
                    None,
                    self.groups,
                    self.base_channels,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

#Basicblock v_1

def resnet20_basicblock_v1(**kwargs: Any) -> ResNet:
    model = ResNet([3*2]*3, _BasicBlock, **kwargs) # n = 3 means 6*18 blocks

    return model


def resnet32_basicblock_v1(**kwargs: Any) -> ResNet:
    model = ResNet([5*2]*3, _BasicBlock, **kwargs) # n = 5 means 6*18 blocks

    return model


def resnet44_basicblock_v1(**kwargs: Any) -> ResNet:
    model = ResNet([7*2]*3, _BasicBlock, **kwargs) # n = 7 means 6*18 blocks

    return model


def resnet56_basicblock_v1(**kwargs: Any) -> ResNet:
    model = ResNet([9*2]*3, _BasicBlock, **kwargs) # n = 9 means 6*18 blocks

    return model


def resnet110_basicblock_v1(**kwargs: Any) -> ResNet:
    model = ResNet([18*2]*3, _BasicBlock, **kwargs) # n = 18 means 6*18 blocks

    return model

#Bottleneck block v_1


def resnet20_bottleneck_v1(**kwargs: Any) -> ResNet:
    model = ResNet([3*2]*3, _Bottleneck_v1, **kwargs) # n = 3 means 6*18 blocks

    return model


def resnet32_bottleneck_v1(**kwargs: Any) -> ResNet:
    model = ResNet([5*2]*3, _Bottleneck_v1, **kwargs) # n = 5 means 6*18 blocks

    return model


def resnet44_bottleneck_v1(**kwargs: Any) -> ResNet:
    model = ResNet([7*2]*3, _Bottleneck_v1, **kwargs) # n = 7 means 6*18 blocks

    return model


def resnet56_bottleneck_v1(**kwargs: Any) -> ResNet:
    model = ResNet([9*2]*3, _Bottleneck_v1, **kwargs) # n = 9 means 6*18 blocks

    return model


def resnet110_bottleneck_v1(**kwargs: Any) -> ResNet:
    model = ResNet([18*2]*3, _Bottleneck_v1, **kwargs) # n = 18 means 6*18 blocks

    return model


#Basicblock v_2

def resnet20_basicblock_v2(**kwargs: Any) -> ResNet:
    model = ResNet([3*2]*3, _BasicBlock_v2, **kwargs) # n = 3 means 6*18 blocks

    return model


def resnet32_basicblock_v2(**kwargs: Any) -> ResNet:
    model = ResNet([5*2]*3, _BasicBlock_v2, **kwargs) # n = 5 means 6*18 blocks

    return model


def resnet44_basicblock_v2(**kwargs: Any) -> ResNet:
    model = ResNet([7*2]*3, _BasicBlock_v2, **kwargs) # n = 7 means 6*18 blocks

    return model


def resnet56_basicblock_v2(**kwargs: Any) -> ResNet:
    model = ResNet([9*2]*3, _BasicBlock_v2, **kwargs) # n = 9 means 6*18 blocks

    return model


def resnet110_basicblock_v2(**kwargs: Any) -> ResNet:
    model = ResNet([18*2]*3, _BasicBlock_v2, **kwargs) # n = 18 means 6*18 blocks

    return model


#Bottleneck block v_2


def resnet20_bottleneck_v2(**kwargs: Any) -> ResNet:
    model = ResNet([3*2]*3, _Bottleneck_v2, **kwargs) # n = 3 means 6*18 blocks

    return model


def resnet32_bottleneck_v2(**kwargs: Any) -> ResNet:
    model = ResNet([5*2]*3, _Bottleneck_v2, **kwargs) # n = 5 means 6*18 blocks

    return model


def resnet44_bottleneck_v2(**kwargs: Any) -> ResNet:
    model = ResNet([7*2]*3, _Bottleneck_v2, **kwargs) # n = 7 means 6*18 blocks

    return model


def resnet56_bottleneck_v2(**kwargs: Any) -> ResNet:
    model = ResNet([9*2]*3, _Bottleneck_v2, **kwargs) # n = 9 means 6*18 blocks

    return model


def resnet110_bottleneck_v2(**kwargs: Any) -> ResNet:
    model = ResNet([18*2]*3, _Bottleneck_v2, **kwargs) # n = 18 means 6*18 blocks

    return model