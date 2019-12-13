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
#import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchsummary import summary

#device = 'cpu'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
#cudnn.benchmark = True

list_IDs = []

train_list_IDs = []
test_list_IDs = []

y = []

for filename in glob.glob(os.path.join('../../../Log_Spectrogram/Electronic','*.png')):
	list_IDs.append(filename.split('/')[-1].split('.')[0])
        y.append(0)

for filename in glob.glob(os.path.join('../../../Log_Spectrogram/Natural','*.png')):
	list_IDs.append(filename.split('/')[-1].split('.')[0])
        y.append(1)

train_list_IDs, test_list_IDs, y_train, y_test = train_test_split(list_IDs, y, test_size=0.2, random_state=42) #100

print(train_list_IDs, 'train')
print(test_list_IDs, 'test')

#print(y)

#subset_list_IDs = []
#subset_list_IDs.extend(train_list_IDs[:10])
#subset_list_IDs.extend(train_list_IDs[-10:])

#print(subset_list_IDs)

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
                
                
		if "Natural" in ID:
		    image_path = os.path.join('../../../Log_Spectrogram','Natural',ID + '.png')
		    #label = 1
		else:
		    image_path = os.path.join('../../../Log_Spectrogram','Electronic',ID + '.png')   
		    #label = 0
                 
		image = skimage.io.imread(image_path) # returns a RGB numpy array
                image = image#.to(device)
                label = self.y[index]#.to(device)
            
		return image,label

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
                
		if "Natural" in ID:
		    image_path = os.path.join('../../../Log_Spectrogram','Natural',ID +'.png')
		    #label = 1
		else:
		    image_path = os.path.join('../../../Log_Spectrogram','Electronic',ID + '.png')   
		    #label = 0 
                
		image = skimage.io.imread(image_path) # returns a RGB numpy array
                image = image#.to(device)
                label = self.y[index]#.to(device)
		return image,label


# Parameters
num_epochs = 10
num_classes = 2
learning_rate = 1e-3
batch_size = 1

training_set = TrainDataset(train_list_IDs, y_train)
train_loader = torch.utils.data.DataLoader(dataset=training_set,
					    batch_size=batch_size,
					    shuffle=True)


test_set = TestDataset(test_list_IDs, y_test) #test_list_Ids
test_loader = torch.utils.data.DataLoader(dataset=test_set,
					  batch_size=batch_size,
					  shuffle=True)




class ConvNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3,7), stride=1),#Nxchxw
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,7)))
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


        #self.fc = nn.Linear(256*132*41, num_classes) #Mel
        self.fc = nn.Linear(256*133*41, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        #print(out.size())
        out = self.layer2(out)
        #print(out.size())
        out = self.layer3(out)
        #print(out.size())
        out = self.layer4(out)
        #print(out.size())
        out = self.layer5(out)
        #print(out.size())
        out = out.reshape(out.size(0), -1)
        #print(out.size())
        
        out = self.fc(out)
        return out

model = ConvNet(num_classes).cuda()#to(device)

summary(model, input_size=(4,6476, 4846))

torch.save(model,'/home/dghose/Voice_Classification/weights/simple_CNN_weights_log.pth' )

model = torch.load('/home/dghose/Voice_Classification/weights/simple_CNN_weights_log.pth')

#torch.save(model.state_dict(), '/home/dghose/Voice_Classification/weights/simple_CNN_weights')


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print('chimken')
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
    	images = np.swapaxes(images, 1, -1)
        
        #print(images.type)
        images = images.float()
        labels = labels.long()
        images = images.cuda()#to(device)
        labels = labels.cuda()#to(device)
        print(labels, 'labels')
        #i Forward pass
        outputs = model(images)
        print(torch.argmax(outputs),'outputs')
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 1 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))



# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = np.swapaxes(images, 1, -1) 
    	images = images.float()
        #labels = labels.float()
        #print(images.shape, 'IMAGES')
        #print(labels.shape,'LABELS')
        print(labels, 'labels')
    	images = images.cuda()#to(device)
        labels = labels.cuda()#to(device)
    
        outputs = model(images)
        print(torch.argmax(outputs),'output')
        _, predicted = torch.max(outputs.data, 1)
        print(predicted,'predicted')
        total += labels.size(0)
        print(total,'total')
        correct += (predicted == labels).sum().item()
        print(correct, 'correct')

    print('Test Accuracy of the model: {} %'.format(100 * correct / total))

'''
# Loop over epochs
for epoch in range(max_epochs):
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        # Model computations
        
        [...]

    # Validation
    with torch.set_grad_enabled(False):
        for local_batch, local_labels in validation_generator:
            # Transfer to GPU
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)

            # Model computations
            [...]

'''
'''
def load_dataset():
    data_path = '../../../Mel_Spectrogram/'
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=torchvision.transforms.ToTensor()
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=0,
        shuffle=True
    )_dir, transform=None:
        """
        Args:
            csv_file (string): Path
    return train_loader
'''
