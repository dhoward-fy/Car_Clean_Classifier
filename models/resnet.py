# vim: set ts=4 sw=4 :
# _*_ coding: utf-8 _*_
#
# Copyright (c) 2023 Fyusion Inc.

import torchvision
from torch import nn


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        # Two output classes: Hot dog, not hot dog
        self.resnet.fc = nn.Linear(512, 2)

    def forward(self, x):
        x = self.resnet(x)
        return x


class ResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        # Two output classes: Hot dog, not hot dog
        self.resnet.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.resnet(x)
        return x
