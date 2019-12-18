#!/usr/bin/env python
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
from trainer import fit
from losses import ContrastiveLoss, TripletLoss
from siamese_network import SiameseNet, TripletNet, EmbeddingNet
from datasets import SiameseDataset, TripletDataset


S_PATH = '/home/shasvatmukes/project/audio_classification/All_Spectrograms/Mel_Spectrograms/Recording_'


def separate_data_by_mic_id_test(test_ids):  # pass a single test id
    test_list_IDs = []
    y = []
    for i in  range(len(test_ids)):
        for filename in glob.glob(
            os.path.join(S_PATH + str(test_id), 'Electronic', '*.png')):
            test_list_IDs.append(filename.split('/')[-1].split('.')[0])
            y.append(0)

        for filename in glob.glob(
            os.path.join(S_PATH + str(test_id), 'Natural', '*.png')):
            test_list_IDs.append(filename.split('/')[-1].split('.')[0])
            y.append(1)
    return test_list_IDs, y


def separate_data_by_mic_id_train(train_ids):  # pass a list of train ids
    train_list_IDs = []
    y = []
    for i in range(len(train_ids)):
        for filename in glob.glob(
            os.path.join(
                S_PATH + str(i), 'Electronic', '*.png')):
            # print(filename)
            train_list_IDs.append(filename.split('/')[-1].split('.')[0])
            y.append(0)

        for filename in glob.glob(
            os.path.join(S_PATH + str(i), 'Natural', '*.png')):
            train_list_IDs.append(filename.split('/')[-1].split('.')[0])
            y.append(1)
    return train_list_IDs, y


# device = 'cpu'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
cuda = True

def run_experiments():

    ###################################EXPERIMENT_1##############################################################
    '''
    DESCRIPTION
    Training and testing set both contain all the recordings. 80-20 split random state =  42
    '''

    '''
    ID: 5924295
    '''

    list_IDs = []
    train_list_IDs = []
    test_list_IDs = []
    y = []
    IDs = [1, 2, 3, 4]
    list_IDs, y = separate_data_by_mic_id_train(IDs)
    train_list_IDs, test_list_IDs, y_train, y_test = train_test_split(
        list_IDs, y, test_size=0.2, random_state=42)  # 100
    #print(train_list_IDs)
    ######HYPERPARAMETERS#############################################
    num_epochs = 10
    num_classes = 2
    learning_rate = 1e-3
    batch_size = 1
    contrastive_loss_margin = 1.0
    triplet_loss_margin = 1.0
    #################################################################

    siamese_training_set = SiameseDataset(train_list_IDs, y_train, True)
    siamese_train_loader = torch.utils.data.DataLoader(dataset=siamese_training_set,
                                               batch_size=batch_size,
                                               shuffle=True)

    siamese_test_set = SiameseDataset(test_list_IDs, y_test, False)  # test_list_Ids
    siamese_test_loader = torch.utils.data.DataLoader(dataset=siamese_test_set,
                                              batch_size=batch_size,
                                              shuffle=True) 
    embedding_net = EmbeddingNet(num_classes)
    model = SiameseNet(embedding_net)
    
    if cuda:
        model.cuda() 
    
    # Loss and optimizer

    criterion = ContrastiveLoss(contrastive_loss_margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training
    train_mode=True
    print('starting training')
    fit(siamese_train_loader, siamese_test_loader, model, criterion, optimizer, num_epochs, use_cuda, train_mode)

    PATH='/home/shasvatmukes/project/audio_classification/weights/vanilla_siamese_weights_log1.pth'  # unique names
    torch.save(model.state_dict(), PATH)

    model.load_state_dict(torch.load(PATH))
    # Test
    train_mode=False
    fit(train_loader, test_loader, model, criterion, optimizer, num_epochs, use_cuda, train_mode)

    '''
    RESULTS
    hyperparameters:
    result:
    exp id:
    '''

    ##################################xxxxxxxxxxxx###############################################################

if __name__ == "__main__":
    run_experiments()
