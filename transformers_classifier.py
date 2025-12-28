# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 13:01:00 2025

@author: asus
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import string
from collections import Counter


#%%
positive_sentences = [
    "Bugün çok verimli geçti, kendimi enerjik hissediyorum.",
    "Projeyi başarıyla tamamladım, sonuçtan gurur duyuyorum.",
    "Yeni şeyler öğrenmek bana ilham veriyor.",
    "Takım çalışması çok keyifliydi.",
    "Portföyüm her geçen gün daha güçlü hale geliyor.",
    "Arkadaşlarım bana destek oldu, çok mutlu oldum.",
    "Hedefime adım adım yaklaşıyorum.",
    "Yaptığım işten keyif alıyorum.",
    "Bugün kendimi çok motive hissediyorum.",
    "Başarılarım bana özgüven veriyor.",
    "Yeni bir şey denemek heyecan verici.",
    "Küçük adımlar bile beni ileri taşıyor.",
    "Çalışmalarımın karşılığını görmek harika.",
    "Kendimi geliştirmek bana mutluluk veriyor.",
    "İlham dolu bir gün geçirdim.",
    "Zorlukları aşmak bana güç veriyor.",
    "Bugün çok şey öğrendim.",
    "Deneyimlerim beni daha iyi yapıyor.",
    "Yaptığım işin değerli olduğunu hissediyorum.",
    "Her gün biraz daha ilerliyorum.",
    "Kendime güvenim arttı.",
    "Yeni fırsatlar beni heyecanlandırıyor.",
    "Çalışmalarımın takdir edilmesi güzel.",
    "Bugün çok üretken oldum.",
    "Başarıya giden yolda ilerliyorum.",
    "Küçük başarılar bile beni mutlu ediyor.",
    "Yeni projeler beni motive ediyor.",
    "Arkadaşlarımla çalışmak keyifliydi.",
    "Bugün kendimi çok huzurlu hissediyorum.",
    "Başarılarım bana umut veriyor.",
    "Yeni şeyler denemek bana cesaret veriyor.",
    "Bugün çok pozitif bir gün geçirdim.",
    "Kendimi güçlü hissediyorum.",
    "Çalışmalarımın sonuçlarını görmek harika.",
    "Yeni bilgiler öğrenmek beni mutlu ediyor.",
    "Bugün çok motive oldum.",
    "Başarıya ulaşmak bana ilham veriyor.",
    "Kendimi çok üretken hissediyorum.",
    "Yeni projeler beni heyecanlandırıyor.",
    "Bugün çok şey başardım.",
    "Çalışmalarımın değerli olduğunu hissediyorum.",
    "Kendime güvenim arttı.",
    "Yeni fırsatlar beni mutlu ediyor.",
    "Bugün çok huzurlu bir gün geçirdim.",
    "Başarılarım bana cesaret veriyor.",
    "Yeni şeyler öğrenmek bana umut veriyor.",
    "Bugün çok pozitif hissettim.",
    "Kendimi güçlü ve motive hissediyorum.",
    "Çalışmalarımın karşılığını almak harika.",
    "Yeni projeler bana ilham veriyor."
]

negative_sentences = [

    "Bu proje beni çok yordu, motivasyonum düştü.",
    "Beklediğim sonucu alamadım, biraz hayal kırıklığı yaşadım.",
    "Bugün işler istediğim gibi gitmedi.",
    "Hataları düzeltmek çok zaman aldı.",
    "Kendimi biraz başarısız hissediyorum.",
    "Çalışmalarımın karşılığını göremedim.",
    "Bugün çok verimsiz geçti.",
    "Motivasyonum tamamen kayboldu.",
    "Projede ilerleme sağlayamadım.",
    "Kendimi çok yorgun hissediyorum.",
    "Başarıya ulaşamadım.",
    "Bugün çok stresli geçti.",
    "Çalışmalarım sonuç vermedi.",
    "Kendimi mutsuz hissediyorum.",
    "Yeni şeyler öğrenemedim.",
    "Bugün çok zor geçti.",
    "Motivasyonum düştü.",
    "Projede hata yaptım.",
    "Kendimi başarısız hissediyorum.",
    "Çalışmalarımın değeri yokmuş gibi hissediyorum.",
    "Bugün çok kötü geçti.",
    "Başarıya ulaşamadım.",
    "Motivasyonum kayboldu.",
    "Projede ilerleme olmadı.",
    "Kendimi çok stresli hissediyorum.",
    "Çalışmalarım sonuçsuz kaldı.",
    "Bugün çok verimsiz geçti.",
    "Motivasyonum düştü.",
    "Projede hata yaptım.",
    "Kendimi başarısız hissediyorum.",
    "Çalışmalarımın değeri yokmuş gibi hissediyorum.",
    "Bugün çok kötü geçti.",
    "Başarıya ulaşamadım.",
    "Motivasyonum kayboldu.",
    "Projede ilerleme olmadı.",
    "Kendimi çok stresli hissediyorum.",
    "Çalışmalarım sonuçsuz kaldı.",
    "Bugün çok verimsiz geçti.",
    "Motivasyonum düştü.",
    "Projede hata yaptım.",
    "Kendimi başarısız hissediyorum.",
    "Çalışmalarımın değeri yokmuş gibi hissediyorum.",
    "Bugün çok kötü geçti.",
    "Başarıya ulaşamadım.",
    "Motivasyonum kayboldu.",
    "Projede ilerleme olmadı.",
    "Kendimi çok stresli hissediyorum.",
    "Çalışmalarım sonuçsuz kaldı.",
    "Bugün çok verimsiz geçti."
]


def preprocess(text):
    text=text.lower()
    text=text.translate(str.maketrans("","",string.punctuation))
    return text

#%%daatset create

data=positive_sentences+negative_sentences
labels=[1]*len(positive_sentences)+[0]*len(negative_sentences)

data=[preprocess(sentence) for sentence in data]#data is processed in preprocess function with for loop over the sentences in data
#create vocab
all_words=" ".join(data).split()
word_counts=Counter(all_words)
vocab={word:idx+1 for  idx,(word,_) in enumerate(word_counts.items())}
vocab["<PAD>"]=0#tokenizing   
#pytorch tensor
max_len=15
def sentence_to_tensor(sentence,vocab,max_len=15):
    tokens=sentence.split()
    indices=[vocab.get(word,0) for word in tokens]# get indices
    indices=indices[:max_len]
    indices+=[0]*(max_len-len(indices))
    return torch.tensor(indices)
    
X=torch.stack([sentence_to_tensor(sentence,vocab,max_len)for sentence in data])
y=torch.tensor(labels)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

    

#%% transformer model
class TransformerClass(nn.Module):
    def __init__(self,vocab_size,embedding_dim,num_heads,num_layers,hidden_dim,num_classes):
        super(TransformerClass,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.positional_encoding=nn.Parameter(torch.randn(1,max_len,embedding_dim))
        self.transformer=nn.Transformer(d_model=embedding_dim,#embedding vector size
                                        nhead=num_heads,#multi head attention mecanizm 
                                        num_encoder_layers=num_layers,#number of transformer encode layer
                                        dim_feedforward=hidden_dim)#encoding hidden layer size
            
        self.fc=nn.Linear(embedding_dim*max_len,hidden_dim)
        self.out=nn.Linear(hidden_dim, num_classes)
        self.sigmoid=nn.Sigmoid()
        
        
    def forward(self,x):
        embedded=self.embedding(x)+self.positional_encoding
        output=self.transformer(embedded,embedded)
        output=output.view(output.size(0),-1)
        output=torch.relu(self.fc(output))
        output=self.out(output)
        output=self.sigmoid(output)
        return output
#model=TransformerClass(len(vocab),32,4,4,64,1)
#%%training

vocab_size=len(vocab)
embedding_dim=32
num_heads=4
num_layers=4
hidden_dim=64
num_classes=1

model=TransformerClass(vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, num_classes)

#loss optimizer
criterion=nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.0005)

number_epochs=100
model.train()
for epoch in range(number_epochs):
    optimizer.zero_grad()
    output=model(X_train.long()).squeeze()
    loss=criterion(output,y_train.float())
    loss.backward()
    optimizer.step()
    
    print(f"Epochs:{epoch+1}/{number_epochs} loss:{loss}")


#%%
model.eval()
with torch.no_grad():
    y_pred=model(X_test.long()).squeeze()
    y_pred=(y_pred>0.5).float()
accuracy=accuracy_score(y_test, y_pred)
print(f"Test accuracy:{accuracy}")
















 
    