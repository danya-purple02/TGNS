import torch 
import numpy as np
##import pandas as pd

import torch.nn as nn

import random

if(torch.cuda.is_available()):
    device = "cuda:0"
else:
    device = "cpu"

device

z = np.random.randint(0, 10, size=(1), dtype=np.int32)

x = torch.from_numpy(z).to(device)
print(x)

x = x.to(dtype=torch.float32)
x.requires_grad=True
print(x)

id_in_group = 26

if(id_in_group % 2 == 0):
    n = 3;
else:
    n = 2;
print("n = ", n, "\n")

y = x ** n
print(y)

y = y * random.randint(1, 10)
print(y)


y = torch.exp(y)
print(y)

y.backward()

print(x.grad)
print(y.grad_fn)