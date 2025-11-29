# -*- coding: utf-8 -*-
"""
Created on Sat Nov 29 12:14:09 2025

@author: asus
"""
import torch 
import torch.nn as nn 
import torch.optim as optim
from collections import Counter
from itertools import product


import matplotlib.pyplot as plt
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 


text = """
Bugün ne hissettiğimi bilmiyorum, ama hissetmediğimi çok iyi biliyorum.
Kapı kapalıydı, yine de içeri bir şeyler sızdı.
Aynı cümleyi üçüncü kez düşündüm, ilk ikisini unuttum.
Zaman bazen hızlı değil, sadece dikkatsiz.
Kelimeler beni terk etti, ben de onları aramıyorum artık.
Gözlerimi kapadım, dünya sesini kısmadı.
Bir şeyler yolunda ama hangi yol olduğunu bilmiyorum.
Kafamın içinde biri konuşuyor, ama ben değilim.
Bugün güçlü görünmekten yoruldum, keşke sadece görünsem.
Düşüncelerim birbirine çarpıyor, kıvılcımlar çıkıyor.
Bazen susuyorum çünkü kelimeler benden utanıyor.
Pencere açık ama içimdeki hava hiç değişmiyor.
Unutmak istiyorum ama unuttuğumu da unutuyorum.
Kendime kızmadım bugün, bu da bir ilerleme belki.
Saat ilerliyor ama ben aynı yerde sayıyorum.
Bir şey söylemek istiyorum ama cümle hâlâ hazırlanıyor.
Gülmeye çalıştım, yüzüm anlamadı.
Bugün içimde bir ağırlık var, sebebi yok, kendisi var.
Düşüncelerim yavaş değil, sadece birbirine takılıyor.
Her şey yolunda gibi ama ben o yolda değilim.
"""
words=text.replace(".","").replace("!","").lower().split()

word_counts=Counter(words)
vocab=sorted(word_counts,key=word_counts.get,reverse=True)
word_to_ix={word: i for i,word in enumerate(vocab)}
ix_to_word={i:word for i,word in enumerate(vocab)}

data=[(words[i],words[i+1])for i in range(len(words)-1)]

#%%
class LSTM(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim):
        
        super(LSTM,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)#embedding layer
        self.lstm=nn.LSTM(embedding_dim,hidden_dim)#LSTM layer
        self.fc=nn.Linear(hidden_dim,vocab_size)#flly connected layer and output layer
        
    def forward(self,x):
        x=self.embedding(x)
        lstm_out,_=self.lstm(x.view(1,1,-1))
        output=self.fc(lstm_out.view(1,-1))
        return output
    
model=LSTM(len(vocab),embedding_dim=8,hidden_dim=32)

#%%hiperparamter tuning
#tensor
def prepare_sequence(seq,to_ix):
    return torch.tensor([to_ix[w]for w in seq],dtype=torch.long)

embedding_sizes=[0,16]
hidden_sizes=[32,64]
learning_rates=[0.01,0.005]
best_loss=float("inf")
best_params={}
print("hiperparamter tuning start")
#grid search
for emb_size,hidden_size,lr in product(embedding_sizes,hidden_sizes,learning_rates):
    print(f"deneme:Embedding:{emb_size},Hidden size:{hidden_size},leaarning rate:{lr}")

    model=LSTM(len(vocab),emb_size,hidden_size)
    loss_function=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)
    
    epochs=50
    total_loss=0
    for epoch in range(epochs):
        epoch_loss=0
        for word,next_word in data:
            model.zero_grad()
            input_tensor=prepare_sequence([word],word_to_ix)
            target_tensor=prepare_sequence([next_word],word_to_ix)
            output=model(input_tensor)#prediction
            loss=loss_function(output,target_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss+=loss.item()
            
        if epoch %10==0:
            print(f"Epopch:{epoch},Loss:{epoch_loss:.5f}")
        total_loss=epoch_loss
        
    if total_loss<best_loss:
        best_loss=total_loss
        best_params={"embedding_dim":emb_size,"hidden_dim":hidden_size,"learning rate":lr}
    print()
    
print(f"Best params:{best_params}")
"""Epopch:0,Loss:674.48332
Epopch:10,Loss:84.43510
Epopch:20,Loss:73.56607
Epopch:30,Loss:68.71065
Epopch:40,Loss:66.06445

Best params:{'embedding_dim': 16, 'hidden_dim': 64, 'learning rate': 0.005}"""

#%%lstm training
final_model=LSTM(len(vocab),best_params['embedding_dim'],best_params['hidden_dim'])
optimizer=optim.Adam(final_model.parameters(),lr=best_params['learning rate'])
loss_function=nn.CrossEntropyLoss()
print("final_model training")
epochs=100
for epoch in range(epochs):
    epoch_loss=0
    for word,next_word in data:
        final_model.zero_grad()
        input_tensor=prepare_sequence([word], word_to_ix)
        target_tensor=prepare_sequence([next_word], word_to_ix)
        output=final_model(input_tensor)
        loss=loss_function(output,target_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss+=loss.item()
    if epoch %10==0:
        print(f"final model epoch :{epoch},loss:{epoch_loss:.5f}")
        
        
"""final_model training
final model epoch :0,loss:675.31297
final model epoch :10,loss:84.45691
final model epoch :20,loss:73.43382
final model epoch :30,loss:68.74530
final model epoch :40,loss:65.46422
final model epoch :50,loss:62.30637
final model epoch :60,loss:59.99450
final model epoch :70,loss:58.36262
final model epoch :80,loss:56.47224
final model epoch :90,loss:55.03164"""
#%%evaluation
def predict_sequence(start_word,num_words):
    current_word=start_word
    output_sequence=[current_word]
    
    for _ in range(num_words):
        with torch.no_grad():
            input_tensor=prepare_sequence([current_word], word_to_ix)
            output=final_model(input_tensor)
            predicted_idx=torch.argmax(output).item()
            predicted_word=ix_to_word[predicted_idx]
            output_sequence.append(predicted_word)
            current_word=predicted_word
       
    return output_sequence
    
start_word="bugün"
num_prediction=9
predicted_sequence=predict_sequence(start_word,num_prediction)
print(" ".join(predicted_sequence))

"""bugün içimde bir şeyler yolunda ama ben aynı yerde sayıyorum"""



