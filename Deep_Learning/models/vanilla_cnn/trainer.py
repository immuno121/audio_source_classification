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


def fit(train_loader, val_loader, model, loss_fn, optimizer, n_epochs, cuda, train_mode):

    if train_mode:
        train(train_loader, model, loss_fn, optimizer, n_epochs, cuda)
    else:
        test(val_loader, model, cuda)


def train(train_loader, model, loss_fn, optimizer, num_epochs, cuda):

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = np.swapaxes(images, 1, -1)
            images = images.float()
            labels = labels.long()
            #print(epoch,i)
            #print(labels)
            if cuda:
                images = images.cuda()  # to(device)
                labels = labels.cuda()  # to(device)
            #print(labels)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 1 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()))


def test(test_loader, model, cuda):
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = np.swapaxes(images, 1, -1)
            images = images.float()
            # labels = labels.float()
            # print(images.shape, 'IMAGES')
            # print(labels.shape,'LABELS')
            if cuda:
                images = images.cuda()  # to(device)
                labels = labels.cuda()  # to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model: {} %'.format(100 * correct / total))
