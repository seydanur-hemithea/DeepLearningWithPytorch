# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 17:40:46 2025

@author: asus
"""

import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch.optim as optim

df=pd.read_csv("mantar_veriseti.csv")

#features


X = df.iloc[:, :-1]

# One-hot encoding: Habitat and color
X_encoded = pd.get_dummies(X, columns=["Habitat", "Renk"], dtype=int)


#labels
y,_ = pd.factorize(df.iloc[:, -1])#get pytorch tuple

y = y.astype("int64")  # sayısal hale getir

#standardization
scaler=StandardScaler()
X_scaler=scaler.fit_transform(X_encoded)



#train,test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaler, y, test_size=0.3, random_state=42,stratify=y

)
def to_tensor(data,target):
    return torch.tensor(data,dtype=torch.float32),torch.tensor(target,dtype=torch.long)
    
X_train,y_train=to_tensor(X_train,y_train)
X_test,y_test=to_tensor(X_test,y_test)

#%%RBFN model and rbf_kernel
def rbf_kernel(X,centers,beta):
    return torch.exp(-beta*torch.cdist(X,centers)**2)
    
class RBFN(nn.Module):
    def __init__(self,input_dim,num_centers,output_dim):
        super(RBFN,self).__init__()
        self.centers=nn.Parameter(torch.randn(num_centers,input_dim))#Initialize randomly rbf centers 
        self.beta=nn.Parameter(torch.ones(1)*2.0)#beta parameter will control rbf width
        self.linear=nn.Linear(num_centers,output_dim)#direct output to fully connected layer 
            
    def forward(self,x):
        #rbf kernel func calculate
        phi=rbf_kernel(x,self.centers,self.beta)
        return self.linear(phi)
#%%model training
num_centers=10
input_dim = X_train.shape[1]  # otomatik giriş boyutu

model=RBFN(input_dim=input_dim,num_centers=num_centers,output_dim=3)
#loss func and optim
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.01)

num_epochs=100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs=model(X_train)
    loss=criterion(outputs,y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1)%10==0:
        print(f"epoch{epoch+1}/{num_epochs},Loss:{loss.item():.4f}")
    
#%%
with torch.no_grad():
    y_pred = model(X_test)
    acc = (torch.argmax(y_pred,axis=1)==y_test).float().mean().item()
    print(f"Test Accuracy: {acc:.4f}")






