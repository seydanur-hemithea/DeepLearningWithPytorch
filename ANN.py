# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 07:55:48 2025

@author: asus
"""

import torch #pytorch library for tensors
import torch.nn as nn #artificial neural network layers description
import torch.optim as optim #optimization algorithms modul
import torchvision#computer vision and pretrained models
import torchvision.transforms as transforms#vision transfroms
import matplotlib.pyplot as plt
#optional:device 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data_loader(batch_size=64):
    
    transform=transforms.Compose([
        transforms.ToTensor(),#scaling(standardization)
        #transforms.Normalize((0.5,), (0.5,))   #pixel value is  scaled between -1 & 1
        ])
#mnist dataset from pytorch
    train_set=torchvision.datasets.MNIST(root="./data",train=True,download=True,transform=transform)
    test_set=torchvision.datasets.MNIST(root="./data",train=False,download=True,transform=transform)
#create pytorch data loader
    train_loader=torch.utils.data.DataLoader(train_set,batch_size=batch_size,shuffle=True)
    test_loader=torch.utils.data.DataLoader(test_set,batch_size=batch_size,shuffle=False)
    return train_loader,test_loader
#train_loader,test_loader=get_data_loader()

#data visualization
def visualize_samples(loader,n):
    images,labels=next(iter(loader))
    print(images[0].shape)
    fig,axes=plt.subplots(1,n,figsize=(10,5))
    for i in range(n):
        axes[i].imshow(images[i].squeeze(),cmap="gray")
        axes[i].set_title(f"label:{labels[i].item()}")
        axes[i].axis("off")
    plt.show()

#visualize_samples(train_loader,4)

#define ann model
class NeuralNetwork(nn.Module):#inheritance from pytorch nn.module class
    def __init__(self):#build nn 
        super(NeuralNetwork,self).__init__()
        #vectorization (1D)
        self.flatten=nn.Flatten()
        #first fully connected layer
        self.fcl1=nn.Linear(28*28,128)#784=input size,128=output size
        #activation fonks
        self.relu=nn.ReLU()
        #second fully connected layer
        self.fcl2=nn.Linear(128,64)#128=input size, 64=output size
        self.fcl3=nn.Linear(64,10)#output layer 64=input size ,10=output size we are classing data 10 class
        
        
    def forward(self,x):#forward propagation,x=image
        #initial x=28*28=flatten 784
        x=self.flatten(x)
        x=self.fcl1(x) 
        x=self.relu(x)
        x=self.fcl2(x)
        x=self.relu(x)
        x=self.fcl3(x)
        return x
#create model and compile
#model=NeuralNetwork().to(device)      
#loss function and optimization algorithms
define_loss_and_optim=lambda model:(
    nn.CrossEntropyLoss(),#multi clas calssification problem loss function
    optim.Adam(model.parameters(),lr=0.001)#update weights with adam    
    )
#criterion,optimizer=define_loss_and_optim(model)

#training
def train_model(model,train_loader,criterion,optimizer,epochs=10):
    model.train() #mode training 
    train_losses=[]#result for per  epoch to save loss value
    for epoch in range(epochs):#training
        total_loss=0#total loss
        for images,labels in train_loader:#iteration
            images,labels=images.to(device),labels.to(device)#data is moved to device 
            optimizer.zero_grad()#get zero gradiant
            predictions=model(images)#appyl model,forward propogation,
            loss=criterion(predictions,labels)#loss calculating->y predction -y real
            loss.backward()#backward propogation new gradiant calculating
            optimizer.step()#update weights
            
            total_loss=total_loss+loss.item()
        avg_loss=total_loss/len(train_loader)#avarage loss 
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs},loss:{avg_loss:.3f}")
    #loss graph
    plt.figure()
    plt.plot(range(1,epochs+1),train_losses,marker="x",linestyle="-",labels="Train Losss")
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()
    
#train_model(model,train_loader,criterion,optimizer,epochs=5)
#%%test
def test_model(model,test_loader):
    model.eval()
    correct=0
    total=0#total data calculater
    with torch.no_grad():#gradiant calculating is unnecessary becuse this is test step
        for images,labels in test_loader:
            images,labels=images.to(device),labels.to(device)
            predictions=model(images)
            _,predicted=torch.max(predictions,1)#find highest probability calss label
            total+=labels.size(0)#update total data 
            correct+=(predicted==labels).sum().item()#calcualte correct pred
            
        print(f"Test accuracy:{100*correct/total:.3f}%")
        
#test_model(model,test_loader)
#%%
if __name__=="__main__":
    train_loader,test_loader=get_data_loader()
    visualize_samples(train_loader,5)
    model=NeuralNetwork().to(device)
    criterion,optimizer=define_loss_and_optim(model)
    train_model(model,train_loader,criterion,optimizer)
    test_model(model,test_loader)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



    
  