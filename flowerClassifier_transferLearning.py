# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 12:48:41 2025

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
import seaborn as sbn 
from sklearn.metrics import confusion_matrix,classification_report

#%%
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

transform_train=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2,contrast=0.2,
                           saturation=0.2,hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    )
transform_test=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
    
    )
train_dataset=datasets.Flowers102(root="./data",split="train",
                             transform=transform_train,download=True 
                             )
test_dataset=datasets.Flowers102(root="./data",split="val",
                             tranform=transform_test,download=True 
                             )
indices=torch.randint(len(train_dataset),(5,))
samples=[train_dataset[i] for i in indices]

fig,axes=plt.subplots(1,5,figsize=(15,5))
for i,(image,label) in enumerate(samples):
    image=image.numpy().transpose((1,2,0))
    image=(image*0.5)+0.5#reverse normalization
    axes[i].imshow(image)
    axes[i].set_title(f"label:{label}")
    axes[i].axis("off")
plt.show()
batch_size=32
train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)
#%%tranfer learning fine tuning model savinng
model=models.mobilenet_v2(pretrained=True)#  are allowing to model pretrained wweights is used?

#classifier layer add
num_ftrs=model.classifier[1].in_features#get input featrures of current classifier
model.classifier[1]=nn.Linear(num_ftrs,102)#last layeer is changed for flower 102
#loss func, optimizer

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.classifier[1].parameters(),lr=0.001) 
shedular=optim.lr_shedular.StepLR(optimizer,step_size=5,gamma=0.1)#decrease lr step by step 

#%%model training
epochs=3
for epoch in tqdm(range(epochs)):
    model.train()
    running_loss=0.0#total loss value
    for images,labels in tqdm(train_loader):
        images,labels=images.to(device),labels.to(device)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.bacward()
        optimizer.step()
        running_loss+=loss.item()
    shedular.step()
    print(f"Epoch:{epoch+1},Loss:{running_loss/len(train_loader):.4f}")

#model save

torch.save(model.state_dict(),"mobilenut_flowers102.pth")
#%%
model.eval()
all_pred=[]
all_labels=[]

with torch.no_grad():
    for images ,labels in images.to(device),labels.to(device):
        outputs=model(images)
        _,predicted=torch.max(outputs,1)
        all_pred.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
#confusion matrix
cm=confusion_matrix(all_labels,all_pred)
plt.figure(fig_size=(12,12))
sbn.heatmap(cm,annot=False,cmap="Blue")
plt.xlabel("predicted")
plt.ylabel("Real")
plt.title("confusion matrix")
plt.show()

print(classification_report(all_labels,all_pred))

        




 