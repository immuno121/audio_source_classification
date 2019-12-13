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
#import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
#from torchsummary import summary
S_PATH = '/home/shasvatmukes/project/audio_classification/All_Spectrograms/Mel_Spectrograms/Recording_'


class TrainDataset(torch.utils.data.Dataset):
    '''Characterizes a dataset for PyTorch'''

    def __init__(self, train_list_IDs, y_train):
        'Initialization'
        self.list_IDs = train_list_IDs
        self.y = y_train
        # later on we can do some preprocessing on-the-fly
        # self.transform = transform

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        # Select sample
        ID = self.list_IDs[index]
        recording_id = ID.split('_')[-1].split('.')[0]
        '''
        Example ID is Electronic_2_7.png. So we first split with '_' [Electronic, 2, 7.png], and get the last element and then split again with '.' [7,png]and get the first element
        '''
        if "Natural" in ID:
            image_path = os.path.join(S_PATH + recording_id, 'Natural', ID + '.png')
        else:
            image_path = os.path.join(S_PATH + recording_id, 'Electronic', ID + '.png')

        image = skimage.io.imread(image_path)  # returns a RGB numpy array
        image = image  # .to(device)
        label = self.y[index]  # .to(device)

        return image, label


class TestDataset(torch.utils.data.Dataset):
    '''Characterizes a dataset for PyTorch'''

    def __init__(self, test_list_IDs, y_test):
        'Initialization'
        self.list_IDs = test_list_IDs
        self.y = y_test
        # later on we can do some preprocessing on-the-fly
        # self.transform = transform

    def __len__(self):
        '''Denotes the total number of samples'''
        return len(self.list_IDs)

    def __getitem__(self, index):
        '''Generates one sample of data'''
        # Select sample
        ID = self.list_IDs[index]
        recording_id = ID.split('_')[-1].split('.')[0]

        '''
        Example ID is Electronic_2_7.png. So we first split with '_' [Electronic, 2, 7.png], and get the last element and then split again with '.' [7,png]and get the first element
        '''
        if "Natural" in ID:
            image_path = os.path.join(S_PATH + recording_id, 'Natural', ID + '.png')
            print(image_path)
        else:
            image_path = os.path.join(S_PATH + recording_id, 'Electronic', ID + '.png')
            #label = 0

        image = skimage.io.imread(image_path)  # returns a RGB numpy array
        image = image  # .to(device)
        label = self.y[index]  # .to(device)
        return image, label
