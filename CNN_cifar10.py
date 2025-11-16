# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 07:57:20 2025

@author: asus
"""

import torch 
import torch.nn as nn 
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loader(batch_size=64):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(((0.5,0.5,0.5)),(0.5,0.5,0.5))# not gray this datasset rgb 
        ])
    #download cfar10 dataset and create train test set 
    train_set=torchvision.datasets.CIFAR10(root="./data",train=True,download=True,
                                           transform=transform)
    test_set=torchvision.datasets.CIFAR10(root="./data",train=False,download=True,
                                          transform=transform)
    
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
    
    return train_loader,test_loader
#%%visualize dataset
def imshow(img):
    #back to before data normalization
    img=img/2+0.5#rewerse normalization
    np_img=img.numpy()#tensor to numpy array trans
    plt.imshow(np.transpose(np_img,(1,2,0)))#correct color shape show

    plt.show()
    
def get_sample_images(train_loader):
    data_iter=iter(train_loader)
    images,labels=next(data_iter)
    return images,labels

def visualize(n):
    train_loader,test_loader=get_data_loader()
    
    images,labels=get_sample_images(train_loader)
    plt.figure()
    for i in range(n):
        plt.subplot(1,n,i+1)
        imshow(images[i])
        plt.title(f"label:{labels[i].item()}")
        plt.axis("off")
    plt.show()
#visualize(3)    
#%% build cnn model
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        
        self.conv1=nn.Conv2d(3,32,kernel_size=3,padding=1)#i_cahnnels=rgb 3,out_cahnnels filter account
        self.relu=nn.ReLU()#activation fonc
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)#2*2 size pooling,kernelsize=matrix size,stride is step by pixels        
        self.conv2=nn.Conv2d(32,64,kernel_size=3,padding=1)
        self.dropout=nn.Dropout(0.2)#%20 rate run
        self.fc1=nn.Linear(64*8*8,128)#fully connected layer input=4096 output=128
        self.fc2=nn.Linear(128,10)#output layer
        #image(3x32x32)->conv(32)->relu(32)->pool(16)
        #cov(16)->relu(16)->pool(8)->image(8x8)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
    def forward(self,x):
        x=self.relu(self.pool(self.bn1(self.conv1(x))))#conv block1
        x=self.pool(self.relu(self.bn2(self.conv2(x))))#conv block 2
        x=x.view(-1,64*8*8)#flatten layer
        x=self.dropout(self.relu(self.fc1(x)))
        x=self.fc2(x)#output layer
        return x

#model=CNN().to(device)

define_loss_and_optimizer=lambda model:(
    nn.CrossEntropyLoss(),#multi class classificaion problem
    optim.SGD(model.parameters(),lr=0.001,momentum=0.9)#stochastic gradient descend
    )    
        
    
#%%
def train_model(model,train_loader,criterion,optimizer,epochs=5):
    model.train()
    train_losses=[]
    
    for epoch in range(epochs):
        total_loss=0
        for images,labels in train_loader:
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            total_loss+=loss.item()
        avg_loss=total_loss/len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch:{epoch+1}/{epochs},loss:{avg_loss:.5f}")
        
    plt.figure()
    plt.plot(range(1,epochs+1),train_losses,marker="o",linestyle="-",label="train loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title("training loss")
    plt.legend()
    plt.show()
    
#train_loader,test_loader=get_data_loader()    
#model=CNN().to(device)
#criterion,optimizer=define_loss_and_optimizer(model)
#train_model(model,train_loader,criterion,optimizer,epochs=10)
    
#%%test
def test_model(model,test_loader,dataset_type):
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
    print(f"{dataset_type} accuracy:{100*correct/total}%")
#test_model(model,test_loader,dataset_type="test") 
   
#test_model(model,train_loader,dataset_type="training") 

    
   #you should examine dataset banchmark accuracy  values so you can should decide your accuracy good or bad 
#%%    
if __name__=="__main__":
    train_loader,test_loader=get_data_loader()
    
    visualize(3)  
    
    model=CNN().to(device)   
    criterion,optimizer=define_loss_and_optimizer(model)
    train_model(model,train_loader,criterion,optimizer,epochs=10)
    
    test_model(model,test_loader,dataset_type="test") 
       
    test_model(model,train_loader,dataset_type="training") 
    
    