#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import cv2
import numpy as np
import gc
import matplotlib.pyplot as plt
import shutil
import os
import argparse
import time

from models import *

ap = argparse.ArgumentParser()
ap.add_argument("--model", default='CNN_1', help="Which model to choose.")
ap.add_argument("--batch_size", type=int, default='128', help="Batch size")
ap.add_argument("--lr", type=float, default=4e-3, help="Learning rate")
ap.add_argument("--rs", type=float, default=1e-3, help="Regularization strength")
ap.add_argument("--num_epochs", type=int, default=10, help="Num epochs")
ap.add_argument("--augment_data", type=bool, default=False, help="Include augmentation.")
ap.add_argument("--loss_type", type=str, default='softmax', help='Type of loss function, \
                SOFTMAX cross entropy or SIGMOID binary cross entropy')
ap.add_argument("--verbose", type=bool, default=True, help="Verbose")

args = ap.parse_args()

batch_size = args.batch_size
num_classes = 10
learning_rate = args.lr 
reg_strength = args.rs
num_epochs = args.num_epochs
print_every_num_iters = 10000

if os.path.exists('runs'):
        shutil.rmtree('runs')

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

if args.model == 'CNN_1':
        model = CNN_1(num_classes).to(device)
elif args.model == 'CNN_2':
        model = CNN_2(num_classes).to(device)
elif args.model == 'CNN_3':
        model = CNN_3(num_classes).to(device)
elif args.model == 'CNN_4':
        model = CNN_4(num_classes).to(device)
elif args.model == 'CNN_5':
        model = CNN_5(num_classes).to(device)
else:
        print("Specified model does not exist. Exiting...")
        quit(0)

print(summary(model, (1, 28, 28)))


#Loading the dataset and preprocessing
if args.augment_data:
        dataset = torchvision.datasets.FashionMNIST(root = './FMNIST',
                                                        train = True,
                                                        transform = transforms.Compose([
                                                                transforms.RandomCrop(28, padding=2), 
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.ToTensor()]), 
                                                                download=False)
else:
        dataset = torchvision.datasets.FashionMNIST(root = './FMNIST',
                                                train = True,
                                                transform = transforms.Compose([
                                                        transforms.ToTensor()]), 
                                                        download=False)

test_set = torchvision.datasets.FashionMNIST(root = './FMNIST',
                                                  train = False,
                                                  transform = transforms.Compose([
                                                          transforms.ToTensor()]), 
                                                          download=False)

train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

train_loader = torch.utils.data.DataLoader(dataset = train_set,
                                           batch_size = batch_size,
                                           shuffle = True)

val_loader = torch.utils.data.DataLoader(dataset = val_set,
                                         batch_size = batch_size,
                                         shuffle = False)

test_loader = torch.utils.data.DataLoader(dataset = test_set,
                                        batch_size = batch_size,
                                        shuffle = False)

# for i, (images, labels) in enumerate(val_loader):
#         print(np.max(np.array(images[0][0])))
#         cv2.imshow('neka slika', np.array(images[0][0]))
#         cv2.waitKey(0)
#         break
writer = SummaryWriter()

if args.loss_type == 'softmax':
        loss_function = nn.CrossEntropyLoss()
elif args.loss_type == 'sigmoid':
        loss_function = nn.BCEWithLogitsLoss()
else: 
        print("Specified loss type not exists. Exiting...")
        quit(0)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg_strength)

train_loss_epochs = []
val_loss_epochs = []

n_iter = 0
for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
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
                
                if args.loss_type == 'softmax':
                        loss = loss_function(outputs, labels)
                else:
                        labels_one_hot = F.one_hot(labels, num_classes=num_classes).type_as(outputs)
                        loss = loss_function(outputs, labels_one_hot)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                gc.collect()
                loss_history.append(loss.item())

                writer.add_scalar('Loss/train_iter', loss.item(), n_iter)
                if i % print_every_num_iters == 0:
                        print(loss.item())
                del images, labels, outputs, loss

        end_time = time.time()
        train_acc = correct_count / total_count
        print("Epoch time: ", end_time - start_time)
        train_loss = np.mean(np.array(loss_history))
        print(f'Epoch {epoch}: Loss: {np.mean(np.array(loss_history))}, Acc: {correct_count / total_count}')
        train_loss_epochs.append(np.mean(np.array(loss_history)))
        writer.add_scalar('Loss/train_epoch', np.mean(np.array(loss_history)), epoch+1)

        # validation after each epoch
        model.eval()
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
                if args.loss_type == 'softmax':
                        loss = loss_function(outputs, labels)
                else:
                        labels_one_hot = F.one_hot(labels, num_classes=num_classes).type_as(outputs)
                        loss = loss_function(outputs, labels_one_hot)
                loss_history.append(loss.item())
                del images, labels, outputs, loss
        val_loss = np.mean(np.array(loss_history))
        print(f'Validation Epoch {epoch}: Loss: {np.mean(np.array(loss_history))}, \
                Acc: {correct_count / total_count}')
        val_acc = correct_count / total_count
        val_loss_epochs.append(np.mean(np.array(loss_history)))
        writer.add_scalar('Loss/val_epoch', np.mean(np.array(loss_history)), epoch+1)
        writer.add_scalars('Loss/all', {'train_epoch_loss': train_loss, \
                                     'val_epoch_loss': val_loss}, epoch+1)
        writer.add_scalars('Accuracy/',{'train_acc': train_acc, 'val_acc': val_acc}, epoch+1)
        # test after each epoch
        model.eval()
        total_count = 0
        correct_count = 0
        for i, (images, labels) in enumerate(test_loader):
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_count += labels.size(0)
                correct_count += (predicted == labels).sum().item() 
                if args.loss_type == 'softmax':
                        loss = loss_function(outputs, labels)
                else:
                        labels_one_hot = F.one_hot(labels, num_classes=num_classes).type_as(outputs)
                        loss = loss_function(outputs, labels_one_hot)
                loss_history.append(loss.item())
                del images, labels, outputs, loss
        test_acc = correct_count / total_count
        print("Test acc: ", test_acc)
        writer.add_scalar('Test accuracy', test_acc, epoch+1)

torch.save(model.state_dict(), f'saved_model_{args.model}.pt')

