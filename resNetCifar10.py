# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 12:26:13 2025

@author: asus
"""
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import os

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(((0.5,0.5,0.5)),(0.5,0.5,0.5))# not gray this datasset rgb 
    ])

 #download cfar10 dataset and create train test set 
train_set=datasets.CIFAR10(root="./data",train=True,download=True,
                                       transform=transform)
test_set=datasets.CIFAR10(root="./data",train=False,download=True,
                                      transform=transform)
batch_size=64
train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
#%%%residual block

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
#%%resNet create
class CustomResNet(nn.Module):
    def __init__(self, num_class=10):
        super(CustomResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = [ResidualBlock(in_channels, out_channels, stride, downsample)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
#%%
use_custom_model=True
if use_custom_model:
    model=CustomResNet().to(device)
else:
    model=models.resnet18(pretrained=True)#fine tuning with trained resnet18 model
    num_ftrs=model.fc.in_features#fully connected layer input size
    model.fc=nn.Sequential(
        nn.Linear(num_ftrs,256),
        nn.ReLU(),
        nn.Linear(256,10))
    model=model.to(device)
#loss func and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

#model training
num_epochs=1
for epoch in tqdm(range(num_epochs)):
    model.train()
    running_loss=0
    for images,labels in tqdm(train_loader):
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print(f"Epoch:{epoch+1}/{num_epochs},loss:{running_loss/len(train_loader)}")

#%%test
model.eval()
correct=0
total=0

with torch.no_grad():
    for images,labels in test_loader:
        images,labels=images.to(device),labels.to(device)
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        
print(f"TEst Accuracy:{100*correct/total}%")




































