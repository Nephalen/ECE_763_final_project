import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .custom_layer import Maxout

class BaseModel(nn.Module):
    #for CIFAR 10 data
    def __init__(self, input_dim, output_dim):
        super(BaseModel, self).__init__()
        
        #define model components
        self.conv1 = nn.Conv2d(input_dim[0], 60, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.maxout_pool = 5
        self.maxout = Maxout(self.maxout_pool)
        self.conv2 = nn.Conv2d(60//self.maxout_pool, 160, 5)
        
        self.fc1 = nn.Linear(160//self.maxout_pool*24*24, 120)
        #self.fc2 = nn.Linear(120, 120)
        self.fc3 = nn.Linear(120//self.maxout_pool, output_dim)
        
        return
    
    def forward(self, x):
        x = self.maxout(self.conv1(x))
        x = self.maxout(self.conv2(x))
        x = x.view(-1, 160//self.maxout_pool*24*24)
        x = self.maxout(self.fc1(x))
        x = F.softmax(self.fc3(x))
        
        return x

class MagicModel1(nn.Module):
    #for CIFAR 10 data
    def __init__(self, input_dim, output_dim):
        super(MagicModel1, self).__init__()
        
        #define model components
        #self.conv1 = nn.Conv2d(input_dim[0], 6, 5)
        self.maxpool = nn.MaxPool2d(2, 1)
        self.dropout2d = nn.Dropout2d()
        self.dropout = nn.Dropout()
        
        self.conv1 = nn.ModuleList([nn.Conv2d(input_dim[0], 60, i) for i in range(1, input_dim[1]-5)])
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(60) for i in range(1, input_dim[1]-5)])
        
        self.conv2 = nn.ModuleList([nn.Conv2d(60, 30, 5) for _ in range(1, input_dim[1]-5)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(30) for i in range(1, input_dim[1]-5)])
        #self.conv2 = nn.Conv2d(20, 60, 5)
        #self.bn2 = nn.BatchNorm2d(60)
        
        self.flat_sizes = np.array([(input_dim[1]-i-5)*(input_dim[1]-i-5) for i in range(1, input_dim[1]-5)])
        self.fc1 = nn.ModuleList([nn.Linear(30*size, 60) for size in self.flat_sizes])
        #self.fc1 = nn.Linear(60*np.sum(self.flat_sizes), 240)
        self.fc2 = nn.Linear(60*len(self.flat_sizes), 480)
        #self.fc2 = nn.Linear(30*len(self.flat_sizes), output_dim)
        self.fc480s = nn.ModuleList([nn.Linear(480, 480) for _ in np.arange(10)])
        self.fc3 = nn.Linear(480, output_dim)
        
        return
    
    def forward(self, x):
        xs = [self.maxpool(F.relu(b(c(x)))) for c, b in zip(self.conv1, self.bn1)]
        xs = [self.maxpool(F.relu(b(c(xx)))) for xx, c, b in zip(xs, self.conv2, self.bn2)]
        #xs = [self.dropout(self.maxpool(F.relu(self.bn2(self.conv2(x))))) for x in xs]
        xs = [xx.view(-1, 30*size) for xx, size in zip(xs, self.flat_sizes)]
        xs = [F.relu(fc(xx)) for xx, fc in zip(xs, self.fc1)]
        x = torch.cat(xs, dim=1)
        x = F.relu(self.fc2(x))
        
        x = F.softmax(self.fc3(x))
        
        return x
    
class MagicModel2(nn.Module):
    #for CIFAR 10 data
    def __init__(self, input_dim, output_dim):
        super(MagicModel2, self).__init__()
        
        #define model components
        #self.conv1 = nn.Conv2d(input_dim[0], 6, 5)
        self.maxpool = nn.MaxPool2d(2, 1)
        self.maxout_pool = 5
        self.maxout = Maxout(self.maxout_pool)
        
        self.conv1 = nn.ModuleList([nn.Conv2d(input_dim[0], 60, i) for i in range(1, input_dim[1]-5)])
        self.bn1 = nn.ModuleList([nn.BatchNorm2d(60) for i in range(1, input_dim[1]-5)])
        
        self.conv2 = nn.ModuleList([nn.Conv2d(60//self.maxout_pool, 30, 5) for _ in range(1, input_dim[1]-5)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(30) for i in range(1, input_dim[1]-5)])
        #self.conv2 = nn.Conv2d(20, 60, 5)
        #self.bn2 = nn.BatchNorm2d(60)
        
        self.flat_sizes = np.array([(input_dim[1]-i-3)*(input_dim[1]-i-3) for i in range(1, input_dim[1]-5)])
        self.fc1 = nn.ModuleList([nn.Linear(30//self.maxout_pool*size, 60) for size in self.flat_sizes])
        #self.fc1 = nn.Linear(60*np.sum(self.flat_sizes), 240)
        self.fc2 = nn.Linear(60//self.maxout_pool*len(self.flat_sizes), 480)
        #self.fc2 = nn.Linear(30*len(self.flat_sizes), output_dim)
        self.fc3 = nn.Linear(480//self.maxout_pool, output_dim)
        
        return
    
    def forward(self, x):
        xs = [self.maxout(F.relu(b(c(x)))) for c, b in zip(self.conv1, self.bn1)]
        xs = [self.maxout(F.relu(b(c(xx)))) for xx, c, b in zip(xs, self.conv2, self.bn2)]
        #xs = [self.dropout(self.maxpool(F.relu(self.bn2(self.conv2(x))))) for x in xs]
        xs = [xx.view(-1, 30//self.maxout_pool*size) for xx, size in zip(xs, self.flat_sizes)]
        xs = [self.maxout(F.relu(fc(xx))) for xx, fc in zip(xs, self.fc1)]
        x = torch.cat(xs, dim=1)
        x = self.maxout(F.relu(self.fc2(x)))
        
        x = F.softmax(self.fc3(x))
        
        return x
    
    