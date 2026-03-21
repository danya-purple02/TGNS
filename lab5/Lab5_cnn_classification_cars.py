# -*- coding: utf-8 -*-
"""
1) mazda_mx5
2) corvette_c6
3) volkswagen_passat_b3
4) porsche_cayenne_1_rest
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import alexnet, AlexNet_Weights

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Устройство:', device)

# Пути к данным
train_dir = './prepared_data/train'
test_dir = './prepared_data/test'
print('Обучающий набор:', train_dir)
print('Тестовый набор:', test_dir)

# Ожидаемые классы в наборе данных
expected_classes = [
    'mazda_mx5',
    'corvette_c6',
    'volkswagen_passat_b3',
    'porsche_cayenne_1_rest'
]

# 1. Простая сверточная сеть
# Так как простые сети работают с изображениями фиксированного размера,
# то наши изображения необходимо смасштабировать и преобразовать в тензор
data_transforms = transforms.Compose([
                        transforms.Resize(68),
                        transforms.CenterCrop(64),
                        transforms.ToTensor()])

# Создадим обучающий и тестовый наборы
train_dataset = torchvision.datasets.ImageFolder(root=train_dir,
                                                 transform=data_transforms)
test_dataset = torchvision.datasets.ImageFolder(root=test_dir,
                                                transform=data_transforms)

# Посмотрим какие классы содержатся в наборе
class_names = train_dataset.classes
print('Найденные классы:', class_names)
print('Ожидались:', expected_classes)

# Список изображений можно получить следующим образом
train_set = train_dataset.samples
print('Первый элемент обучающего набора:', train_set[0])
print('Размер обучающего набора:', len(train_set))

batch_size = 10

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                    shuffle=True,  num_workers=2)


test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                    shuffle=False, num_workers=2) 


# Загрузим одну порцию данных
# Каждое обращение к DataLoader возвращает изображения и их классы
inputs, classes = next(iter(train_loader))
print('Размер batch:', inputs.shape) # 10 изображений, 3 канала (RGB), размер каждого 224х224 пикселя

# Построим сетку из изображений
img = torchvision.utils.make_grid(inputs, nrow=5) # метод делает сетку из картинок
img = img.numpy().transpose((1, 2, 0)) # для отображения через matplotlib 
plt.figure()
plt.imshow(img)
plt.title('Примеры изображений из train')
plt.axis('off')


# Теперь можно переходить к созданию сети
# Для этого будем использовать как и ранее метод Sequential
# который объединит несколько слоев в один стек
class CnNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.layer1 = nn.Sequential(
        # первый сверточный слой с ReLU активацией и maxpooling-ом
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=2), # 3 канала, 16 фильтров, размер ядра 7
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # второй сверточный слой 
        # количество каналов второго слоя равно количеству фильтров предыдущего слоя
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # третий сверточный слой 
        # ядро фильтра от слоя к слою уменьшается
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # классификационный слой имеет нейронов: количество фильтров * размеры карты признаков
        self.fc = nn.Linear(8 * 8 * 64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1) # флаттеринг
        out = self.fc(out)
        return out


# Количество классов
num_classes = len(class_names)

# создаем экземпляр сети
net = CnNet(num_classes).to(device)

# Задаем функцию потерь и алгоритм оптимизации
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# создаем цикл обучения и замеряем время его выполнения
num_epochs = 50
save_loss = []

t = time.time()
for epoch in range(num_epochs):
    net.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # прямой проход
        outputs = net(images)
        # вычисление значения функции потерь
        loss = lossFn(outputs, labels)

        # Обратный проход (вычисляем градиенты)
        optimizer.zero_grad()
        loss.backward()
        # делаем шаг оптимизации весов
        optimizer.step()

        # сохраняем loss
        save_loss.append(loss.item())

        # выводим немного диагностической информации
        if i % 50 == 0:
            print('Эпоха ' + str(epoch + 1) + ' из ' + str(num_epochs) +
                  ' Шаг ' + str(i) + ' Ошибка: ' + str(loss.item()))

print('Время обучения:', time.time() - t)

# Посмотрим как уменьшался loss в процессе обучения
plt.figure()
plt.plot(save_loss)
plt.title('График loss')
plt.xlabel('Шаг обучения')
plt.ylabel('Loss')

# Посчитаем точность нашей модели: количество правильно классифицированных картинок
# поделенное на общее количество тестовых примеров
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad(): # отключим вычисление граиентов, т.к. будем делать только прямой проход
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images) # делаем предсказание по пакету
        _, pred_class = torch.max(pred.data, 1) # выбираем класс с максимальной оценкой
        correct_predictions += (pred_class == labels).sum().item() # сравниваем ответ с правильной меткой

print('Точность модели: ' + str(100 * correct_predictions / num_test_samples) + '%')

# сохраняем нашу модель в файл для дальнейшего использования
torch.save(net.state_dict(), 'CnNet_cars.ckpt')


# 2. Предобученная сеть AlexNet
# Так как сеть, которую мы планируем взять за базу натренирована на изображениях 
# определенного размера, то наши изображения необходимо к ним преобразовать
alexnet_transforms = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225] )
    ])

# Пересоздадим датасеты с учетом новых размеров и нормировки яркости
train_dataset = torchvision.datasets.ImageFolder(root=train_dir,
                                                 transform=alexnet_transforms)
test_dataset = torchvision.datasets.ImageFolder(root=test_dir,
                                                transform=alexnet_transforms)

batch_size = 10

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=2)

# В качестве донора возьмем преобученную на ImageNet наборе сеть AlexNet
# Список доступных предобученных сетей можно посмотреть тут https://pytorch.org/vision/main/models.html
net = torchvision.models.alexnet(pretrained=True)


# можно посмотреть структуру этой сети
print(net)


# Так как веса feature_extractor уже обучены, нам нужно их "заморозить", чтобы 
# быстрее научился наш классификатор
#  для этого отключаем у всех слоев (включая слои feature_extractor-а) градиенты
for param in net.parameters():
    param.requires_grad = False

# Меняем только последний слой классификатора под 4 класса

# Выходной слой AlexNet содержит 1000 нейронов (по количеству классов в ImageNet).
# Нам нужно его заменить на слой, содержащий только len(class_names) класса.

class_names = train_dataset.classes
num_classes = len(class_names)

new_classifier = net.classifier[:-1] # берем все слой классификатора кроме последнего
new_classifier.add_module('fc', nn.Linear(4096, num_classes))# добавляем последним слой с двумя нейронами на выходе
net.classifier = new_classifier # меняем классификатор сети

net = net.to(device)

# проверим эффективность новой сети
net.eval()
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad(): # отключим вычисление граиентов, т.к. будем делать только прямой проход
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images) # делаем предсказание по пакету
        _, pred_class = torch.max(pred.data, 1) # выбираем класс с максимальной оценкой
        correct_predictions += (pred_class == labels).sum().item()

print('Точность AlexNet до обучения: ' + str(100 * correct_predictions / num_test_samples) + '%')
# явно требуется обучение

# Перейдем к обучению.
# Зададим количество эпох обучения, функционал потерь и оптимизатор.
num_epochs = 50
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.classifier.parameters(), lr=0.001)

save_loss = []
#save_acc = []

# создаем цикл обучения и замеряем время его выполнения
t = time.time()
for epoch in range(num_epochs):
    net.train()
    correct_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        # прямой проход
        outputs = net(images)
        # вычисление значения функции потерь
        loss = lossFn(outputs, labels)
         # Обратный проход (вычисляем градиенты)

        optimizer.zero_grad()
        loss.backward()
        # делаем шаг оптимизации весов
        optimizer.step()
        save_loss.append(loss.item())

        #_, predicted = torch.max(outputs.data, 1)
        #total_train += labels.size(0)
        #correct_train += (predicted == labels).sum().item()

        # выводим немного диагностической информации
        if i % 20 == 0:
            print('AlexNet | Эпоха ' + str(epoch + 1) + ' из ' + str(num_epochs) +
                  ' | Шаг ' + str(i) + ' | Ошибка: ' + str(loss.item()))


print('Время обучения AlexNet:', time.time() - t)

# График loss
plt.figure()
plt.plot(save_loss)
plt.title('График loss для AlexNet')
plt.xlabel('Шаг обучения')
plt.ylabel('Loss')

# График accuracy
#plt.figure()
#plt.plot(save_acc)
#plt.title('График accuracy по эпохам для AlexNet')
#plt.xlabel('Эпоха')
#plt.ylabel('Accuracy, %')

# Еще раз посчитаем точность нашей модели (после обучения)
correct_predictions = 0
num_test_samples = len(test_dataset)

with torch.no_grad(): # отключим вычисление граиентов, т.к. будем делать только прямой проход
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images) # делаем предсказание по пакету
        _, pred_class = torch.max(pred.data, 1) # выбираем класс с максимальной оценкой
        correct_predictions += (pred_class == labels).sum().item()

print('Точность AlexNet после обучения: ' + str(100 * correct_predictions / num_test_samples) + '%')
# уже лучше

# Сохраняем обученную модель
torch.save(net.state_dict(), 'alexnet_cars_4classes.pth')

# Реализуем отображение картинок и их класса, предсказанного сетью
inputs, classes = next(iter(test_loader))
pred = net(inputs.to(device))
_, pred_class = torch.max(pred.data, 1)

for i, j in zip(inputs, pred_class.cpu()):
    img = i.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.figure()
    plt.imshow(img)
    plt.title('Предсказанный класс: ' + class_names[j])
    plt.axis('off')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

print('Соответствие индексов классам:')
for idx, name in enumerate(class_names):
    print(idx, '->', name)

plt.show()
