#!/usr/bin/env python
# coding: utf-8

# In[1]:


import snntorch as snn
import torch
from torchvision.datasets import MNIST
from torch import nn

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from IPython.display import HTML

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import utils
import torch.nn as nn

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from tqdm import tqdm
from math import sqrt

#!echo '/data' >> .gitignore


# In[2]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 0

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
print(f"Running on {device}")


# ## vvvvv Utilities vvvvv

# In[3]:


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
    
    def show(self):
        assert len(self.to_plot) != 0, "No spikes loaded"
        ani = matplotlib.animation.FuncAnimation(self.fig, self.blit, frames = self.frames)
        return HTML(ani.to_jshtml())

def quick_animate(three_dim_tensor, size, frames):
    temp = SpikeAnimation(max(2, len(three_dim_tensor)), frames=frames)
    for iii in three_dim_tensor:
        temp.add_to_plot(iii, size, " ")
    return temp.show()


# ## Building the network (no stdp)

# In[4]:


class snn_hmax(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        self.num_ker1 = 4
        self.num_ker2 = 4
        
        self.s1 = nn.Conv2d(in_channels=1, out_channels=self.num_ker1, 
                                     kernel_size=3, stride=1, padding=0, groups=1, bias=True)
        self.c1 = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.s2 = nn.Conv2d(in_channels=self.num_ker1, out_channels = self.num_ker2,
                                      kernel_size=4, stride=1, padding=0, bias=True)
        self.c2 = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.fully_connected1 = nn.Linear(self.num_ker2*25, 128)
        self.relu = nn.ReLU()
        self.fully_connected2 = nn.Linear(128, 10)
        self.alph = nn.Parameter(1+torch.rand(self.num_ker2))
        self.bet = nn.Parameter(1+torch.rand(self.num_ker2))
        self.bet.requires_grad_(False)
        self.stdp_bool = False
        self.past_beta = torch.zeros_like(self.bet).to(device)
        self.past_alph = torch.zeros_like(self.alph).to(device)
        
    def forward(self, input_spikes):
    
        self.s1_records = []
        self.c1_records = []
        self.s2_records = []
        self.c2_records = []
        self.output_records = []
        
        if self.stdp_bool == True:
            print(f"Alpha: {self.alph}, DD {self.past_alph/self.alph}"); self.past_alph = self.alph
            print(f"Beta: {self.bet}, DD {self.past_beta/self.bet}"); self.past_beta = self.bet
            pos_delts = torch.zeros(50, 100, 1, 13, 13)
            neg_delts = torch.zeros(50, 100, 1, 13, 13)
            temp =  nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
            for im in range(images.shape[1]):
                xx = torch.stack([images[iii+1, im, 0]-images[iii, im, 0] for iii in range(images.shape[0]-1)])
                pos_delts[:-1, im, 0] = temp(xx*(xx>0))
                maxx = torch.max(xx)-xx
                neg_delts[:-1, im, 0] = temp(maxx*(maxx>0))
            #shape [49, 100, 1, 13, 13]
        
        #[NumSteps, BatSize, 1, 28, 28]
        for iii, step in enumerate(input_spikes):
            #[BatSize, 1, 28, 28]
            s1_currents = self.s1(step) #X kernels*image_dim1*image_dim2
            #[BatSize, Ker1, 2X, 2X]
            self.s1_records.append(s1_currents)

            max_pooled = self.c1(s1_currents).to(device)
            #[BatSize, Ker1, 1X, 1X]

            if self.stdp_bool == True:
                #xxx_delts ~([50, 100, 1, 13, 13])
                #max_pooled ~([100, 4, 13, 13])
                for ker in range(max_pooled.shape[1]):
                    max_pooled = 0.5*max_pooled + \
                            self.alph[ker]*(max_pooled*pos_delts[iii])
                    max_pooled = 0.5*max_pooled + \
                            self.bet[ker]*(max_pooled*neg_delts[iii])
                    
            self.c1_records.append(max_pooled)
                
            s2_convoluteds = self.s2(max_pooled).to(device)
            self.s2_records.append(s2_convoluteds)
            #[BatSize, Ker2, 28, 28]
                                    
            maxpooled2 = self.c2(s2_convoluteds)
            self.c2_records.append(maxpooled2)
            
            #shape [10, 16, 5, 5]
            out = maxpooled2.reshape(maxpooled2.size(0), -1)
            out = self.fully_connected1(out)
            out = self.relu(out)
            out = self.fully_connected2(out)
            self.output_records.append(out)

        self.output_records = torch.stack(self.output_records, dim=0)
        return self.output_records.sum(dim=0)/len(input_spikes)

    def apply_stdp(self, x=True):
        self.stdp_bool = x 
        self.alph.requires_grad_(x)
        self.bet.requires_grad_(x)
    
    def __call__(self, spikes_in):
        return self.forward(spikes_in)


# ## Initial training

# In[5]:


# dataloader arguments
batch_size = 100
data_path="./data/mnist"

dtype = torch.float

transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)


train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


# In[6]:


model = snn_hmax().to(device)


# In[7]:


criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))


# In[8]:


from tqdm import tqdm

num_epochs = 4
loss_hist = []
counter = 0
num_steps = 5
# Outer training loop
num_evaluate = 50

model.stdp_bool = False

for epoch in range(1):
    #Load in the data in batches using the train_loader object
    for i, (images, labels) in tqdm(enumerate(train_loader)):  
        labels = labels.to(device)
        images = spikegen.rate(images, num_steps=50).to(device)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_hist.append(loss)
        # Backward and optimize
        optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        
        if i%50 == 0:
            with torch.no_grad():
                correct = 0
                total = 0
                for j, (images, labels) in enumerate(train_loader):
                    if j == num_evaluate:
                        break
                    images = images.to(device)
                    images = spikegen.rate(images, num_steps=50).to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


# In[220]:


z = torch.stack(model.c1_records, 0)
print(z.shape)
out = z.repeat(1,1,2,1,1)
for im in range(z.shape[1]):
    for ker in range(z.shape[2]):
        spike_delta = torch.stack([z[iii+1, im, ker]-z[iii, im, ker] for iii in range(49)], 0)
        neg_delta = spike_delta*(spike_delta<0)
        pos_delta = spike_delta*(spike_delta>0)
        out[:-1, im, 2*ker] = z[:-1, im, ker]*pos_delta
        out[:-1, im, (2*ker)+1] = z[:-1, im, ker]*neg_delta
        
quick_animate([z[:-1, 1, 0].cpu(), out[:-1, 1, 0].cpu(), out[:-1, 1, 1].cpu()], (13,13),49)


# In[204]:


for parameter in model.parameters():
    parameter.requires_grad_(False)

model.apply_stdp()
print([parameter.requires_grad for parameter in model.parameters()])
print([parameter for parameter in model.parameters()])


# In[38]:


xx = images

#for image in images
pos_delts = torch.zeros(49, 100, 1, 13, 13)
neg_delts = torch.zeros(49, 100, 1, 13, 13)
temp =  nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

for im in range(images.shape[1]):
    xx = torch.stack([images[iii+1, im, 0]-images[iii, im, 0] for iii in range(images.shape[0]-1)])
    pos_delts[:, im, 0] = temp(xx*(xx>0))
    maxx = torch.max(xx)-xx
    neg_delts[:, im, 0] = temp(maxx*(maxx>0))

print(pos_delts.shape)
print(neg_delts.shape)
xx = pos_delts[:, 0, 0]
yy = neg_delts[:, 0, 0]
sa = SpikeAnimation(num_plots=3, frames=49)
sa.add_to_plot(xx.to('cpu'), (13,13), "inp")
sa.add_to_plot(yy.to('cpu'), (13,13), "delta")
sa.add_to_plot(images[:-1, 0, 0].to('cpu'), (28,28), "delta")
#sa.add_to_plot( (14,14), "minus delta")
sa.show()


# In[29]:


xx = [float(item) for item in loss_hist]
plt.plot(xx)


# In[16]:


spike_anim = SpikeAnimation(num_plots=6, frames=4)
spike_anim.add_to_plot(images[:, 1, 0].to('cpu'), (28,28), "Inputs")
spike_anim.add_to_plot(model.s1_records[0][1].to('cpu'), (26,26), "Conv1")
spike_anim.add_to_plot(model.s1_records[:][1][0].to('cpu'), (26,26), "Conv1")
spike_anim.add_to_plot(model.c1_records[0][1].to('cpu'), (13, 13), "Max1")
spike_anim.add_to_plot(model.s2_records[0][1].to('cpu'), (10, 10), "Conv2")
spike_anim.add_to_plot(model.c2_records[0][1].to('cpu'), (5, 5), "Max2")
spike_anim.show()


# In[ ]:


xxx = spikegen.rate(datum, num_steps=50).to('cpu')

print(xxx.shape)

spike_anim = SpikeAnimation(num_plots=2, frames=50)
spike_anim.add_to_plot(xxx, (28,28), 'test')
spike_anim.show()


# In[ ]:


data, targets = next(iter(train_loader))
print(data.shape)
print(targets.shape)
data = data.to(device)
targets = targets.to(device)

# forward pass
model.train()

sum_recordings = torch.zeros(data.shape[0],1,10, device=device)
spikes_out = torch.zeros(128, 1, 10, device=device)
for iii, datum in tqdm(enumerate(data)):
    xxx = spikegen.rate(datum, num_steps=50).flatten(2,3).to(device)
    _,_,_,spikes,mem_pots = model(xxx)
    sum_recordings[iii, 0, :] = mem_pots
    spikes_out[iii, 0, :] = spikes

# initialize the loss & sum over time
loss_val = torch.zeros((1), dtype=dtype, device=device)

pred = sum_recordings.squeeze(1)
loss_val += loss(pred, targets)

# Gradient calculation + weight update
optimizer.zero_grad()
loss_val.backward()
optimizer.step()

# Store loss history for future plotting
loss_hist.append(loss_val.item())

# Test set
with torch.no_grad():
    model.eval()
    test_data, test_targets = next(iter(test_loader))
    test_data = test_data.to(device)
    test_targets = test_targets.to(device)

    test_mem = torch.zeros(test_data.shape[0],1,10, device=device)
    test_spikes = torch.zeros(128, 1, 10, device=device)
    for iii, datum in tqdm(enumerate(test_data)):
        xxx = spikegen.rate(datum, num_steps=50).flatten(2,3)
        s1_convoluteds, max_pooled, s2_convoluteds, spikes,mem_pots = model(xxx)
        test_mem[iii, 0, :] = mem_pots
        test_spikes[iii, 0, :] = spikes

    # Test set loss
    pred = test_mem.squeeze(1)
    test_loss = torch.zeros((1), dtype=dtype, device=device)

    test_loss += loss(pred, test_targets)
    test_loss_hist.append(test_loss.item())

    # Print train/test loss/accuracy
    if counter % 1 == 0:
        train_printer(
            data, targets, epoch,
            counter, iter_counter,
            loss_hist, test_loss_hist,
            test_data, test_targets)
    counter += 1
    iter_counter +=1


# In[ ]:


spikes_anim = SpikeAnimation(num_plots=2)
spikes_anim.add_to_plot(spikes, (2,5), f"S1 K{1}")


# In[ ]:


torch.autograd.set_detect_anomaly(True)


# In[ ]:


import tonic
import tonic.transforms as transforms
from tqdm import tqdm


# In[ ]:


dataset = tonic.datasets.NMNIST(save_to='./data',
                               train=True)
events, target = dataset[0]
print(events)


sensor_size = tonic.datasets.NMNIST.sensor_size

# Denoise removes isolated, one-off events
# time_window
frame_transform = transforms.Compose([transforms.Denoise(filter_time=10000),
                                      transforms.ToFrame(sensor_size=sensor_size,
                                                         time_window=1000)
                                     ])

trainset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=True)
testset = tonic.datasets.NMNIST(save_to='./data', transform=frame_transform, train=False)


# In[ ]:


from torch.utils.data import DataLoader
from tonic import DiskCachedDataset


cached_trainset = DiskCachedDataset(trainset, cache_path='./cache/nmnist/train')
cached_dataloader = DataLoader(cached_trainset)

batch_size = 128
trainloader = DataLoader(cached_trainset, batch_size=batch_size, collate_fn=tonic.collation.PadTensors())

def load_sample_batched():
    events, target = next(iter(cached_dataloader))


# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=2e-2, betas=(0.9, 0.999))
loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

num_epochs = 1
num_iters = 50

loss_hist = []
acc_hist = []

def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(data.size(0)):  # data.size(0) = number of time steps
      spk_out = net(data[step])
      spk_rec.append(spk_out)

    return torch.stack(spk_rec)

# training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(tqdm(iter(trainloader))):
        data = data.to(device)
        data = data.flatten(3,4)
        targets = targets.to(device)
        
        model.train()
        spk_rec = forward_pass(model, data)
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

        acc = SF.accuracy_rate(spk_rec, targets)
        acc_hist.append(acc)
        print(f"Accuracy: {acc * 100:.2f}%\n")

        # training loop breaks after 50 iterations
        if i == num_iters:
          break


# In[ ]:


data[0,4].sum()


# In[ ]:


torch.cuda.empty_cache()

