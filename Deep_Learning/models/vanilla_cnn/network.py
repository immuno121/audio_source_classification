from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import skimage
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
# import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
# from torchsummary import summary
S_PATH = '/home/shasvatmukes/project/audio_classification/All_Spectrograms/Mel_Spectrograms/Recording_'


class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 7), stride=1),  # Nxchxw
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 7)))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.fc1 = nn.Linear(256 * 132 * 41, 256)  # Mel
        #self.fc2 = nn.Linear(1024, 512)
        #self.fc2 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        # print(out.size())
        out = self.layer2(out)
        # print(out.size())
        out = self.layer3(out)
        # print(out.size())
        out = self.layer4(out)
        # print(out.size())
        out = self.layer5(out)
        # print(out.size())
        out = out.reshape(out.size(0), -1)
        # print(out.size())

        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        #out = self.relu(out)
        #out = self.fc3(out)
        #out = self.relu(out)
        #out = self.fc4(out) 
        return out


# model = ConvNet(num_classes).cuda()  # to(device)
# return model
