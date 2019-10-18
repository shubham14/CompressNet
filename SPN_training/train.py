# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:50:38 2019

@author: Shubham
"""

import os, time, random, argparse
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import torch.optim
import cv2
from torch.utils import data
from PIL import Image
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
import time
import pickle as pkl
import h5py
import copy
import glob
import torchvision.transforms as trans

class h5_loader(data.Dataset):
    def __init__(self, file_path):
        self.file_list = [f for f in glob.glob(os.path.join(file_path, '*.h5'))]
        self.len = len(self.file_list)

    def __getitem__(self, index):
        self.file = h5py.File(self.file_list[index], 'r')
        self.data = self.file.get('Feature')
        self.data = torch.tensor(np.array(self.data))     
        self.label = self.file.get('label')
        return np.array(self.data), np.array(self.label)

    def __len__(self):
        return self.len

class Index_pred(nn.Module):
    def __init__(self, in_channels=60, out_channels=60):
        super(Index_pred, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        out = self.conv1(x)
        max_val, _ = torch.max(out, dim = 1)
        max_val, _ = torch.max(max_val, dim = 1) 
        out = F.sigmoid(out)
        return out
    
def train(trainloader, net, criterion, optimizer, device, scheduler, epochs=2):
    for epoch in range(epochs):
        scheduler.step()
        start = time.time()
        running_loss = 0.0
        for i, (images, target) in enumerate(trainloader):
            images = images.to(device)
            target = target.to(device).float()
            target = target/4
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:    
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')
   
def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            pred, labels = data
            _, c, h, w = pred.shape
            pred = pred.to(device)
            labels = labels.to(device)
            labels = labels/4
            outputs = net(pred)
            # out_c = copy.deepcopy(outputs)
            out_c = outputs
            outputs[(out_c >= 0.0) & (out_c < 0.25)] = 0
            outputs[(out_c >= 0.25) & (out_c < 0.5)] = 1
            outputs[(out_c >= 0.5) & (out_c < 0.75)] = 2
            outputs[(out_c >= 0.75) & (out_c <= 1.0)] = 3
            print(outputs)
            total = c * h * w
            labels = labels * 4
            correct = (outputs.long() == labels.long()).sum().item()
            print('Accuracy of the network on the 1 sample: %d %%' % (
                100 * correct / total))
    
def main(trainloader, testloader):
    os.environ['CUDA_VISIBLE_DEVICES']='2'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on device : {}'.format(device))
    #device = torch.device('cpu')
    net = Index_pred().to(device)
    saved_state_dict = torch.load('index_pred.pth')
    net.load_state_dict(saved_state_dict.state_dict())
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(net.parameters() ,lr=0.01, eps=1e-08)
    scheduler = MultiStepLR(optimizer, milestones=[50, 100], gamma=0.1)
    # train(trainloader, net, criterion, optimizer, device, scheduler)
    # torch.save(net, 'index_pred1.pth') # save the model
    test(testloader, net, device) # testing
    
if __name__ == "__main__":
    # obg = h5_loader('./SWWAE_dataset/')
    # print(obg[0])
    train_loader = torch.utils.data.DataLoader(h5_loader('./SPN_dataset/'),batch_size=2, shuffle=True, num_workers=1, pin_memory=True) # Data loader for train set
    main(train_loader, train_loader)