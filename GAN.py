# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 13:29:22 2025

@author: asus
"""

import torch 
import torch.nn as nn 
import torch.optim as optim
import torchvision 
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size=128
image_size=28*28
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
        ])

dataset=datasets.MNIST(root="./data",train=True,transform=transform,download=True)
dataLoader=DataLoader(dataset,batch_size=batch_size,shuffle=True)


#%%Discriminator
class Discriminator(nn.Module):
    def __init__(self):
       super(Discriminator,self).__init__()
       self.model=nn.Sequential(
           nn.Linear(image_size,1024),#input:image size
           nn.LeakyReLU(0.2),#activation func
           nn.Linear(1024,512),
           nn.LeakyReLU(0.2),
           nn.Linear(512,256),
           nn.LeakyReLU(0.2),
           nn.Linear(256,1),#real or fake 
           nn.Sigmoid()#0-1 
           )
      
           
    def forward(self,img):
        return self.model(img.view(-1,image_size))#Feed the model with the rectified image.
#%%generator
class Generator(nn.Module):
    def __init__(self,z_dim):
        super(Generator,self).__init__()
        self.model=nn.Sequential(
            nn.Linear(z_dim,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,image_size),
            nn.Tanh()#out activaton func
            )
    def forward(self,x):
        return self.model(x).view(-1,1,28,28)#28*28
    
#%%GAN training
lr=0.0002#learning rate
z_dim=100#random noise vector size
epochs=10


generator=Generator(z_dim).to(device)
discriminator=Discriminator().to(device)

criterion=nn.BCELoss()
g_optimizer=optim.Adam(generator.parameters(),lr=lr,betas=(0.5,0.999))
d_optimizer=optim.Adam(discriminator.parameters(),lr=lr,betas=(0.5,0.999))

for epoch in range(epochs):
    for i ,(real_imgs,_) in enumerate(dataLoader):
        real_imgs=real_imgs.to(device)
        batch_size=real_imgs.size(0)#The current batch size
        real_labels=torch.ones(batch_size,1).to(device)#to label real visual as 1
        fake_labels=torch.zeros(batch_size,1).to(device)#to lebl fake visual as 0
       #discriminator training 
        z=torch.randn(batch_size,z_dim).to(device)#random noisy vector
        fake_imgs=generator(z)#generator fake visual with generator
        real_loss=criterion(discriminator(real_imgs),real_labels)
        fake_loss=criterion(discriminator(fake_imgs.detach()),fake_labels)
        d_loss=real_loss+fake_loss
        
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        
        g_loss=criterion(discriminator(fake_imgs),real_labels)
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
    print(f"Epoch{epoch+1}/{epochs}d_loss:{d_loss.item():.3f},g_loss{g_loss.item():.3f}")
        
#%%
from torchvision.utils import make_grid



with torch.no_grad():
    z=torch.randn(16,z_dim).to(device)
    sample_imgs=generator(z).cpu()
    grid = np.transpose(
        make_grid(sample_imgs, nrow=4, normalize=True),
        (1, 2, 0)
    )
    
    plt.imshow(grid)
    plt.show()