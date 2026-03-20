import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class RegressNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        nn.Module.__init__(self)
        
        self.layers = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, output_size)
                                   )
    
    def forward(self, x):
        pred = self.layers(x)
        return pred


if(torch.cuda.is_available()):
    device = "cuda:0"
else:
    device = "cpu"
    
    

filename = "dataset_simple.csv"

df = pd.read_csv(filename)

x = torch.Tensor(df.iloc[:, [0]].values).to(device)
y = torch.Tensor(df.iloc[:, [1]].values).to(device)


plt.figure()
plt.scatter(df.iloc[:, [0]].values, df.iloc[:, 1].values, marker='o')

input_size = x.shape[1]
hidden_size = 3
output_size = y.shape[1]

print("\nhidden_size = ", hidden_size, "\n")

net = RegressNet(input_size, hidden_size, output_size).to(device)

lossFn = nn.L1Loss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.001)



epochs = 10000
for i in range(0, epochs):
    pred = net.layers(x)
    loss = lossFn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(i % 1000 == 0):
        print("ошибка на ", i, "операции: ", loss.item())
        
#проверка работы после обучения:
print("\n*проверка обученной сети*")
    
with torch.no_grad():
    pred = net.layers(x)


err = (abs(y.max()-pred.max()))/2

print("\nмаксимальная ошибка: ", err)

print("\npred: ", pred.squeeze())
print("\ny:    ", y.squeeze())

x_plot = df.iloc[:, 0].values
y_plot = df.iloc[:, 1].values
pred_plot = pred.detach().cpu().numpy()
sort_idx = np.argsort(x_plot)

plt.figure()
plt.scatter(x_plot, y_plot, marker='o', label='Реальные данные')
plt.plot(x_plot[sort_idx], pred_plot[sort_idx], label='Предсказание сети')
plt.legend()
plt.title("Результат регрессии")
plt.show()