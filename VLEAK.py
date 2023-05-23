#!/usr/bin/env python
# coding: utf-8

# In[1]:


import snntorch as snn
import torch
from torch import nn

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from IPython import display
from IPython.display import clear_output, HTML, Video

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch.nn as nn

from tqdm.auto import tqdm
from datetime import datetime

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.nn.functional import pad, softmax

from math import sqrt

import tonic
import random
from tonic import DiskCachedDataset
import tonic.transforms as transforms

import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
print(f"Running on {device}")

#RUN ONCE
#!echo '/data' >> .gitignore


# ### Load DVSGesture data, Utilities functions

# In[2]:


class SpikeAnimation():
    def __init__(self, num_plots, color_map = 'coolwarm', frames = None):
        self.to_plot = []
        self.sizes = []
        self.ims = []
        self.color_map = color_map
        self.calc_frames = True
        self.fig, self.axs = plt.subplots(1, num_plots)
        if frames is not None:
            self.frames = frames
            self.calc_frames = False
        return
            
                       
    def add_to_plot(self, three_dim_tensor, size, title):
        assert len(three_dim_tensor.shape) == 3, "Not a three dimensional tensor"
        self.to_plot.append(three_dim_tensor.squeeze(1).detach().numpy())
        self.axs[len(self.to_plot)-1].set_title(title)
        self.sizes.append(size)
        self.ims.append( self.axs[len(self.to_plot)-1].imshow(self.to_plot[-1][0].reshape(size), 
                                      cmap = self.color_map) )
        return
        
    def blit(self, n):
        for iii, image in enumerate(self.ims):
            image.set_array(self.to_plot[iii][n].reshape(self.sizes[iii]))
        return self.ims
    
    def show(self, return_obj=False):
        assert len(self.to_plot) != 0, "No spikes loaded"
        ani = matplotlib.animation.FuncAnimation(self.fig, self.blit, frames = self.frames)
        if return_obj:
            return ani
        return HTML(ani.to_jshtml())

def quick_animate(three_dim_tensor, frames, titles=None, return_obj=False,):
    temp = SpikeAnimation(max(2, len(three_dim_tensor)), frames=frames)
    titles = [" "]*len(three_dim_tensor) if titles is None else titles
    for count, iii in enumerate(three_dim_tensor):
        temp.add_to_plot(iii.cpu(), (iii.shape[-1], iii.shape[-2]), f"{titles[count]}")
    return temp.show(return_obj)

def add_to_class(Class):
    """Register functions as methods in created class."""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
    return wrapper

def pos_neg_to_frame(loader_tensor):
    # combine positive and negative channels of neuromorphic dataset
    return (loader_tensor[:,:,1]-loader_tensor[:,:,0]).movedim(0,1).unsqueeze(2)


# In[3]:


class snn_hmax(nn.Module):
    def __init__(self):
        
        super().__init__()
        self.counter = 0
        self.num_ker1 = 4
        self.num_ker2 = 8
        self.num_classes = 11
        self.num_capsule_channels = 16
        self.feature_connections = torch.randint(4, (self.num_capsule_channels,5))
        self.stdp_inshallah = nn.Conv2d(in_channels=10, out_channels=self.num_capsule_channels, kernel_size=31, stride=1, padding=1, groups=1, bias=True)
        self.s1 = nn.Conv2d(in_channels=1, out_channels=self.num_ker1, 
                                     kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.c1 = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.c2 = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.fully_connected1 = nn.Linear(8064, 512)
        self.relu = nn.ReLU()
        self.fully_connected2 = nn.Linear(512, self.num_classes)
        


# In[4]:


dataset = tonic.datasets.DVSGesture(save_to='./data',
                               train=True)
sensor_size = tonic.datasets.DVSGesture.sensor_size

frame_transform = transforms.Compose([transforms.CropTime(max=2_000_000),
                                      transforms.Downsample(spatial_factor=0.5),
                                      transforms.ToFrame(sensor_size=(64,64,2),
                                                         time_window=10_000)
                                     ])
trainset = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=True)
testset = tonic.datasets.DVSGesture(save_to='./data', transform=frame_transform, train=False)

random.seed(seed)
batch_size = 1

train_subsample_percent = 30
test_subsample_percent = 10

################# DELETE IF USING WHOLE SET
train_subsample_idx = random.sample(range(0, len(trainset)), int(len(trainset)*train_subsample_percent//100))
test_subsample_idx = random.sample(range(0, len(testset)), int(len(testset)*test_subsample_percent//100))
trainset = Subset(trainset,  train_subsample_idx)
testset= Subset(testset,  test_subsample_idx)
##########################################

cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/DVSGesture/train')
cached_testset = DiskCachedDataset(testset, cache_path='./cache/DVSGesture/test')

cached_dataloader = DataLoader(cached_trainset, batch_size=batch_size)

train_loader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=True), shuffle=True)
test_loader = DataLoader(cached_testset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=True))


# In[6]:


inp = input("Enter model accuracy to load (or \"None\"): ")
if inp=="None":
    model = snn_hmax().to(device)
else:
    model = torch.load(f"modelF{inp}.pt").to(device)


# ### Static model methods for step-by-step demonstration

# In[7]:


weights_updates = []
yy_record = torch.zeros(323,24-10, batch_size, 10, 31,31).cpu()

@add_to_class(snn_hmax)        
def forward(self, input_spikes, record=True):
    
    num_steps, bat_size= input_spikes.shape[0:2]
    im_size=31
    container = torch.zeros(num_steps-10, bat_size, self.num_capsule_channels, 6,6).to(device)
    
    for cap_chan in range(self.num_capsule_channels):
        capsule = torch.zeros((bat_size,10,im_size,im_size))
        for jjj, batch_step in enumerate(input_spikes):

            s1_reduced = self.s1(batch_step.to(device)) 
            c1_reduced = self.c1(s1_reduced)
            capsule = capsule.roll(-1,1)
            chann = c1_reduced[:, self.feature_connections[cap_chan,(jjj%5)]]
            capsule[:len(c1_reduced),-1] = chann
            if jjj>10: #wait for capsule to fill up
                if record:
                    yy_record[self.counter, jjj-10, :len(c1_reduced)] = capsule

                xx = self.stdp_inshallah(torch.softmax(capsule.to(device), dim=-1))
                xx = self.c2(xx.reshape(len(xx),12,12))
                #cond = (xx.mean((1,2))>xx.mean((0,1,2)))
                #container[jjj, cond, cap_chan] = xx[cond]
                container[jjj-10, :, cap_chan] = xx
    if record:
        self.counter = self.counter+1
    #container = container[(container.sum([1,2,3,4]).sort(descending=True)[1]).sort()[0]] #iff selecting x most salient
    container = container.movedim(0,1)
    out = container.reshape(container.size(0), -1).to(device)
    out = model.fully_connected1(out)
    out = model.relu(out)
    out = model.fully_connected2(out)

    return out

@add_to_class(snn_hmax)
def __call__(self, input_spikes, **kwargs):
    return self.forward(input_spikes, **kwargs)


# ## Initial training

# In[8]:


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.99, 0.999))


# Want to show relationship between weight updates and timing

# In[ ]:


model.counter = 0
num_epochs = 1
loss_hist = []
#weights = list(weights)
# Outer training loop
num_evaluate = 25
switch_flag = False
for epoch in range(num_epochs):
    weights_updates = []
    #Load in the data in batches using the train_loader object
    for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", position=0, leave=True)):  
        
        labels = labels.to(device)
        ims = pos_neg_to_frame(images)[50:(74-50)*3+50:3].to(device)

        w = model.stdp_inshallah.weight.clone().detach()
        outputs = model(ims)
        
        loss = criterion(outputs, labels)
        loss_hist.append(loss)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        weights_updates.append(model.stdp_inshallah.weight-w)
            
        if i == 0:
            with torch.no_grad():
                correct = 0
                total = 0
                predictions = []
                true = []
                for j, (test_images, test_labels) in enumerate(tqdm(test_loader, desc="Testing")):
                    if j == num_evaluate:
                        break
                    ims = pos_neg_to_frame(test_images)[50:(74-50)*3+50:3].to(device)
                    
                    #ims = ims[:, (test_labels == 1) | (test_labels==9)]
                    #test_labels = test_labels[(test_labels == 1) | (test_labels==9)]
                    #if ims.shape[1] == 0:
                    #    continue

                    test_labels = test_labels.to(device)
                    outputs = model(ims, record=False)
                    _, predicted = torch.max(outputs.data, 1)
                    total += test_labels.size(0)
                    correct += (predicted == test_labels).sum().item()
                    true.append(int(test_labels))
                    predictions.append(int(predicted))

                print(f"Predicted: {predictions}")
                print(f"Labels: {true}")
                    
                print(f'Accuracy of the network: {100*correct/total} %')
                
    model.counter=0
    weights_updates = torch.stack(weights_updates).cpu()
    torch.save(weights_updates, f"weight_delta{100*correct/total}.pt")
    torch.save(yy_record, f"inputs{100*correct/total}.pt")
    torch.save(model, f"modelF{100*correct/total}.pt")  
    weights_updates = []
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    #torch.save(model, f"model{100*correct/total}.pt")

