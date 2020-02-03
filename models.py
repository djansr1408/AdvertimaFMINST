import torch
import torch.nn as nn
from torchsummary import summary


class CNN_1(nn.Module):
    def __init__(self, num_classes):
        super(CNN_1, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(nn.Linear(3200, 1024), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.layer7 = nn.Linear(512, num_classes)
        
    
    def forward(self, X):
        res = self.layer1(X)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.layer4(res)
        
        res = res.reshape(res.size(0), -1)
        res = self.layer5(res)
        res = self.layer6(res)
        res = self.layer7(res)

        return res


class CNN_2(nn.Module):
    def __init__(self, num_classes):
        super(CNN_2, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64), 
            nn.ReLU()
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(nn.Linear(3200, 1024), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.layer7 = nn.Linear(512, num_classes)
        
    
    def forward(self, X):
        res = self.layer1(X)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.layer4(res)
        res = res.reshape(res.size(0), -1)
        res = self.layer5(res)
        res = self.layer6(res)
        res = self.layer7(res)

        return res


class CNN_3(nn.Module):
    def __init__(self, num_classes):
        super(CNN_3, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64), 
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(nn.Linear(3200, 1024), nn.Dropout(p=0.5), nn.ReLU())
        self.layer6 = nn.Sequential(nn.Linear(1024, 512), nn.Dropout(p=0.5), nn.ReLU())
        self.layer7 = nn.Linear(512, num_classes)
        
    
    def forward(self, X):
        res = self.layer1(X)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.layer4(res)
        
        res = res.reshape(res.size(0), -1)
        res = self.layer5(res)
        res = self.layer6(res)
        res = self.layer7(res)

        return res



class CNN_4(nn.Module):
    def __init__(self, num_classes):
        super(CNN_4, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(64), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Linear(512, num_classes)
        
    
    def forward(self, X):
        res = self.layer1(X)
        res = self.layer2(res)
        res = self.layer3(res)
        res = res.reshape(res.size(0), -1)
        res = self.layer5(res)
        
        return res


class CNN_5(nn.Module):
    def __init__(self, num_classes):
        super(CNN_5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64), 
            nn.ReLU()
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(128), 
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(256), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer6 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.layer7 = nn.Linear(512, num_classes)
        
    
    def forward(self, X):
        res = self.layer1(X)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.layer4(res)
        res = self.layer5(res)
        res = res.reshape(res.size(0), -1)
        res = self.layer6(res)
        res = self.layer7(res)

        return res