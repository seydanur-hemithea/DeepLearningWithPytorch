# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 20:04:22 2025

@author: asus
"""

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import transforms,datasets
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

transform=transforms.Compose([transforms.ToTensor()])
#train test dataset load
train_dataset=datasets.FashionMNIST(root="./data",train=True,transform=transform,
                                    download=True)
test_dataset=datasets.FashionMNIST(root="./data",train=False,transform=transform,
                                    download=True)

batch_size=128
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)




#%%autoencoder developing
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder=nn.Sequential(            
        nn.Flatten(),#28*28 784
        nn.Linear(28*28,256),
        nn.ReLU(),#activation func
        nn.Linear(256,64),#fully connected layer
        nn.ReLU()       
        )
        self.decoder=nn.Sequential(
        nn.Linear(64,256),
        nn.ReLU(),
        nn.Linear(256,28*28),
        nn.Sigmoid(),
        nn.Unflatten(1,(1,28,28))      
   )
    def forward(self,x):
        encoded=self.encoder(x)
        decoded=self.decoder(encoded)
        return decoded
 #%%       
class EarlyStopping:
    def __init__(self,patience=5,min_delta=0.001):
        
      
       self.patience=patience # After how many epochs should we stop if the model is not improving?
       self.min_delta=min_delta #min change amount in loss
       self.best_loss=None#best loss value
       self.counter=0#stabil epoch counter
     
       
    def __call__(self,loss):
        if self.best_loss is None or loss< self.best_loss-self.min_delta:
            self.best_loss=loss 
            self.counter=0#get zero counter if it is improving
        else:
            self.counter+=1#increase counter if is not improving
        if self.counter >= self.patience:#stabile epoch count more than patience->stop
            return True
        
        return False     
#%% model trainig
#hperparamters
epochs=5
learning_rate=1e-3


model=Autoencoder()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
early_stopping=EarlyStopping(patience=5,min_delta=0.001)
    
#training function
def training(model,train_loader,optimizer,criterion,early_stopping,epochs):
    model.train()
    for epoch in range(epochs):
        total_loss=0
        for inputs,_ in train_loader:
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs,inputs)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
            
        avg_loss=total_loss/len(train_loader)
        print(f"Epoch{epoch+1}/{epochs},Loss:{avg_loss:.3f}")
        
        if early_stopping(avg_loss):
            print(f"Early stopping at epoch{epoch+1}")
            break
training(model,train_loader,optimizer,criterion,early_stopping,epochs)
#%%model testing
from scipy.ndimage import gaussian_filter
def compute_ssim(img1,img2,sigma=1.5):
    C1=(0.01*255)**2
    C2=(0.03*255)**2
    
    img1=img1.astype(np.float64)
    img2=img2.astype(np.float64)
    
    mu1=gaussian_filter(img1,sigma)
    mu2=gaussian_filter(img2,sigma)
    
    mu1_sq=mu1**2 
    mu2_sq=mu2**2
    mu1_mu2=mu1*mu2
    
    sigma1_sq=gaussian_filter(img1**2,sigma)-mu1_sq
    sigma2_sq=gaussian_filter(img2**2,sigma)-mu2_sq
    sigma12=gaussian_filter(img1*img2,sigma)-mu1_mu2
    
    #ssim map ,
    ssim_map=((2*mu1_mu2+C1)*(2*sigma12+C2))/((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
    
    return ssim_map.mean()
def evaluate(model,test_loader,n_images=10):
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            inputs,_=batch
            outputs=model(inputs)
            break
    inputs=inputs.numpy()
    outputs=outputs.numpy()
    fig, axes = plt.subplots(2, n_images, figsize=(n_images, 3))
    ssim_scores=[]
    
    for i in range(n_images):
        img1=np.squeeze(inputs[i])
        img2=np.squeeze(outputs[i])
        
        ssim_score=compute_ssim(img1,img2)
        ssim_scores.append(ssim_score)
        
        axes[0,i].imshow(img1,cmap="gray")
        axes[0,i].axis("off")
        axes[1,i].imshow(img2,cmap="gray")
        axes[1,i].axis("off")
      
    axes[0,0].set_title("original")
    axes[1,0].set_title("decoded image")
    plt.show()
    
    avg_ssim=np.mean(ssim_scores)
    print(f"avarage SSIM:{avg_ssim}")
evaluate(model,test_loader,n_images=10)










#%%
examples = iter(test_loader)
images, _ = next(examples)
output = model(images)

plt.figure(figsize=(10,4))
for i in range(6):
    plt.subplot(2,6,i+1)
    plt.imshow(images[i].squeeze(), cmap="gray")
    plt.axis("off")
    plt.subplot(2,6,i+7)
    plt.imshow(output[i].detach().squeeze(), cmap="gray")
    plt.axis("off")
plt.show()        