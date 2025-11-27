# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 08:31:39 2025

@author: asus
"""

import numpy as np
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def generate_data(seq_length=50, num_samples=1000):
    
    X = np.linspace(0, 100, num_samples)
    y = np.sin(X)

    sequence = []
    targets = []

    
    for i in range(len(X) - seq_length):
        sequence.append(y[i:i + seq_length])
        targets.append(y[i + seq_length])
        
    plt.figure(figsize=(8, 2))
    plt.plot(X,y,label='sin(t)', color='b',linewidth=2)
    
    plt.title('🎭 Sinüs Sekansı ve Hedef Noktası')
    plt.xlabel('Zaman adımı')
    plt.ylabel('Değer')
    plt.legend()
    plt.grid(True)
    plt.show()
    return np.array(sequence), np.array(targets)


sequence, targets = generate_data()
#%%
import torch 
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        
       super(RNN,self).__init__()
       self.RNN=nn.RNN(input_size,hidden_size,num_layers,batch_first=True)#
       self.fc1=nn.Linear(hidden_size,output_size)
       
       
        
    def forward(self,x):
        out, _ = self.RNN(x)
        out = out[:, -1, :]   # son zaman adımı
        out = self.fc1(out)
        return out

model=RNN(1,16,1,1)

#%%
seq_length=50
input_size=1
hidden_size=16
output_size=1
num_layers=1
epochs=20
batch_size=32
learning_rate=0.001
X,y=generate_data(seq_length)
X=torch.tensor(X,dtype=torch.float32).unsqueeze(-1)#translate torch tensor andd add dimension,
y=torch.tensor(y,dtype=torch.float32).unsqueeze(-1)
dataset=torch.utils.data.TensorDataset(X,y)#pytorch dataset
dataloader=torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)

#model
model=RNN(input_size,hidden_size,output_size,num_layers)
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
for epoch in range(epochs):
    for batch_x,batch_y in dataloader:
        optimizer.zero_grad()#get zeros grad
        pred_y=model(batch_x)#model prediction
        loss=criterion(pred_y,batch_y)#compare real and pred and calculate loss 
        loss.backward()#bcalculate grad with bachward propogation 
        optimizer.step()#update weights
    print(f"epoch:{epoch+1}/{epochs},loss:{loss.item():.4f}")
    
#%%test data
X_test=np.linspace(100,110,seq_length).reshape(1,-1)
y_test=np.sin(X_test)
x_test2=np.linspace(120,130,seq_length).reshape(1,-1)
y_test2=np.sin(x_test2)

#from numpy to tensor

X_test=torch.tensor(y_test,dtype=torch.float32).unsqueeze(-1)
x_test2=torch.tensor(y_test2,dtype=torch.float32).unsqueeze(-1)   

model.eval()
prediction1=model(X_test).detach().numpy()
prediction2=model(x_test2).detach().numpy()

plt.figure()
plt.plot(np.linspace(0,100,len(y)),y,marker="o",label="Training dataset")
plt.plot(X_test.numpy().flatten(),marker="o",label="test1")
plt.plot(x_test2.numpy().flatten(),marker="o",label="test2")

plt.plot(np.arange(seq_length,seq_length+1),prediction1.flatten(),"ro",label="prediction1")
plt.plot(np.arange(seq_length,seq_length+1),prediction2.flatten(),"ro",label="prediction2")
plt.legend()
plt.show()
                  
    
        


