# задача классификации "купит" - "не купит"

import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class ClassificationNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(nn.Linear(input_size, hidden_size),
                                    nn.Tanh(),
                                    nn.Linear(hidden_size, output_size),
                                    nn.Tanh())
    
    def forward(self, x):
        pred = self.layers(x)
        return pred

if(torch.cuda.is_available()):
    device = "cuda:0"
else:
    device = "cpu"



filename = "dataset_simple.csv"

df = pd.read_csv(filename)


x_np = df.iloc[:, [0, 1]].values.astype(np.float32)
x_np = (x_np - x_np.mean(axis=0)) / x_np.std(axis=0)
x = torch.Tensor(x_np).to(device)

y = df.iloc[:,[2]].values
y = torch.Tensor(np.where(y == 1, 1, -1).reshape(-1,1)).to(device)

input_size = 2
hidden_size = 3
output_size = 1
print("hidden_size = ", hidden_size)

net = ClassificationNet(input_size, hidden_size, output_size).to(device)



# проверка работы до обучения:
#print("*проверка до обучения*")
with torch.no_grad():
    pred = net.layers(x)

pred = (pred >= 0).float() * 2 - 1
#print(pred)

err = sum(abs(y-pred))/2
print("ошибки: ", err)


# обучение:
print("*обчение*")
lossFn = nn.MSELoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.005)

epochs = 6000
for i in range(0, epochs):
    pred = net.layers(x)
    loss = lossFn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(i % 500 == 0):
        print("ошибка на ", i, "операции: ", loss.item())
        
#проверка работы после обучения:
print("*проверка обученной сети*")
    
with torch.no_grad():
    pred = net.layers(x)

pred = (pred >= 0).float() * 2 - 1
err = sum(abs(y-pred))/2


# вывод результатов
print("ошибки:", err)
#print("предсказание - истина")
#for i in range(0, pred.shape[0]):
#    print(pred[i].item(), " - ", y[i].item())

print("pred: ", pred.squeeze())
print("y:    ", y.squeeze())