#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import cv2
import numpy as np
import gc
import matplotlib.pyplot as plt


from models import *

batch_sizes = [64, 128]
num_classes = 10
learning_rates = np.logspace(-4, -3, 10)
reg_strengths = np.logspace(-4, 1, 10)
num_epochs = 1
print_every_num_iters = 100000

best_params = {}
combinations = []
for bs in batch_sizes:
    for rs in reg_strengths:
        for lr in learning_rates:
            combinations.append({'batch_size': bs, 'learning_rate': lr, 'reg_strength': rs})

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)


dataset = torchvision.datasets.FashionMNIST(root = './FMNIST',
                                                  train = True,
                                                  transform = transforms.Compose([
                                                          transforms.RandomCrop(28, padding=2), 
                                                          transforms.RandomHorizontalFlip(),
                                                          transforms.RandomHorizontalFlip(),
                                                          transforms.ToTensor()]), 
                                                          download=False)

test_set = torchvision.datasets.FashionMNIST(root = './FMNIST',
                                                  train = False,
                                                  transform = transforms.Compose([
                                                          transforms.ToTensor()]), 
                                                          download=False)

train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

writer = SummaryWriter()

max_val_acc = 0

for comb_idx, comb in enumerate(combinations):
    print(comb_idx, comb)
    train_loader = torch.utils.data.DataLoader(dataset = train_set,
                                           batch_size = comb['batch_size'],
                                           shuffle = True)

    val_loader = torch.utils.data.DataLoader(dataset = val_set,
                                            batch_size = comb['batch_size'],
                                            shuffle = False)

    test_loader = torch.utils.data.DataLoader(dataset = test_set,
                                            batch_size = comb['batch_size'],
                                            shuffle = False)
    model = CNN_2(num_classes).to(device)

    # print(summary(model, (1, 28, 28)))

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=comb['learning_rate'], \
                                weight_decay=comb['reg_strength'])

    train_loss_epochs = []
    val_loss_epochs = []

    n_iter = 0
    for epoch in range(num_epochs):
        loss_history = []
        total_count = 0
        correct_count = 0
        for i, (images, labels) in enumerate(train_loader):
                n_iter += 1
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_count += labels.size(0)
                correct_count += (predicted == labels).sum().item() 
                
                loss = loss_function(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                gc.collect()
                loss_history.append(loss.item())

                writer.add_scalar('Loss/train_iter', loss.item(), n_iter)
                if i % print_every_num_iters == 0:
                        print(loss.item())
                del images, labels, outputs, loss

        train_loss = np.mean(np.array(loss_history))
        print(f'Epoch {epoch}: Loss: {np.mean(np.array(loss_history))}, Acc: {correct_count / total_count}')


        # validation after each epoch
        total_count = 0
        correct_count = 0
        loss_history = []
        for i, (images, labels) in enumerate(val_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_count += labels.size(0)
                correct_count += (predicted == labels).sum().item() 
                loss = loss_function(outputs, labels)
                loss_history.append(loss.item())
                del images, labels, outputs, loss
        val_loss = np.mean(np.array(loss_history))
        val_acc = correct_count / total_count
        print(f'Validation Epoch {epoch}: Loss: {np.mean(np.array(loss_history))}, Acc: {correct_count / total_count}')
    
    print("Best params till now: ", best_params)
    if val_acc > max_val_acc:
        best_params = comb
        best_params['val_acc'] = val_acc
        max_val_acc = val_acc
        print("Best params till now: ", best_params)

print("Best params: ", best_params)
