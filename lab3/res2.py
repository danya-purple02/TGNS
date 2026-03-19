# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:34:30 2026

@author: izada
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

if(torch.cuda.is_available()):
    device = "cuda:0"
else:
    device = "cpu"

print("device = ", device, "\n")

filename = 'iris.csv'
df = pd.read_csv(filename)

print("samples:\n", df.head(10))

x = df.iloc[:, [0,1,2,3]].values
y = df.iloc[:, 4].values

y = np.where(y == "Setosa", 0,
    np.where(y == "Versicolor", 1, 2))

x = torch.tensor(x, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)

y_new = torch.zeros((y.shape[0], 3), dtype = torch.float32).to(device)

for i in range(y.shape[0]):
    if y[i] == 0:
        y_new[i][0] = 1
    elif y[i] == 1:
        y_new[i][1] = 1
    else:
        y_new[i][2] = 1

y = y_new

print(x.shape)
#print(x)

print(y.shape)
#print(y)


linear = nn.Linear(4, 3).to(device)

print ('w: ', linear.weight)
print ('b: ', linear.bias)

lossFn = nn.MSELoss() # MSE - среднеквадратичная ошибка, вычисляется как sqrt(sum(y^2 - yp^2))

optimizer = torch.optim.SGD(linear.parameters(), lr=0.01) # lr - скорость обучения

yp = linear(x)

# имея предсказание можно вычислить ошибку
#loss = lossFn(yp, y)
#print('Ошибка: ', loss.item())

# и сделать обратный проход, который вычислит градиенты (по ним скорректируем веса)
#loss.backward()

for i in range(0, 150):
    optimizer.zero_grad()
    
    pred = linear(x)
    loss = lossFn(pred, y)
    
    loss.backward()
    optimizer.step()
    
    if((i+1) % 10 == 0):
        print('Ошибка на ' + str(i+1) + ' итерации: ', loss.item())

with torch.no_grad():
    pred_class = torch.argmax(pred, dim=1)
    true_class = torch.argmax(y, dim=1)
    acc = (pred_class == true_class).float().mean()

print("предсказанные классы:\n", pred_class)
print("истинные классы:\n", true_class)
print("точность =", acc.item())
