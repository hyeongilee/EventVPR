#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append(CURRENT_TEST_DIR + "/../../src")


# In[2]:


import sys
print(sys.version)


# In[3]:


from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import slayerSNN as snn
from learningStats import learningStats
from IPython.display import HTML
import zipfile


# In[4]:


netParams = snn.params('network.yaml')


# In[5]:


# Dataset definition
class nmnistDataset(Dataset):
    def __init__(self, datasetPath, sampleFile, samplingTime, sampleLength):
        self.path = datasetPath 
        self.samples = np.loadtxt(sampleFile).astype('int')
        self.samplingTime = samplingTime
        self.nTimeBins    = int(sampleLength / samplingTime)

    def __getitem__(self, index):
        inputIndex  = self.samples[index, 0]
        classLabel  = self.samples[index, 1]

        inputSpikes = snn.io.readNpSpikes(
                        self.path + str(inputIndex.item()) + '.npy'
                        ).toSpikeTensor(torch.zeros((2,320,240,self.nTimeBins)),
                        samplingTime=self.samplingTime)
        desiredClass = torch.zeros((2, 1, 1, 1))
        desiredClass[classLabel,...] = 1
        return inputSpikes, desiredClass, classLabel

    def __len__(self):
        return self.samples.shape[0]


# In[6]:


class Network(torch.nn.Module):
    def __init__(self, netParams):
        super(Network, self).__init__()
        # initialize slayer
        slayer = snn.layer(netParams['neuron'], netParams['simulation'])
        self.slayer = slayer
        # define network functions
        self.conv1 = slayer.conv(2, 8, 5, stride=2, padding=1)
        self.conv2 = slayer.conv(8, 16, 3, stride=2, padding=1)
        self.conv3 = slayer.conv(16, 32, 3, stride=2, padding=1)
        self.pool1 = slayer.pool(2)
        self.pool2 = slayer.pool(2)
        self.fc1   = slayer.dense((8, 8, 64), 2)

    def forward(self, spikeInput):
        spikeLayer1 = self.slayer.spike(self.conv1(self.slayer.psp(spikeInput ))) # 32, 32, 16
        spikeLayer2 = self.slayer.spike(self.pool1(self.slayer.psp(spikeLayer1))) # 16, 16, 16
        spikeLayer3 = self.slayer.spike(self.conv2(self.slayer.psp(spikeLayer2))) # 16, 16, 32
        spikeLayer4 = self.slayer.spike(self.pool2(self.slayer.psp(spikeLayer3))) #  8,  8, 32
        spikeLayer5 = self.slayer.spike(self.conv3(self.slayer.psp(spikeLayer4))) #  8,  8, 64
        spikeOut    = self.slayer.spike(self.fc1  (self.slayer.psp(spikeLayer5))) #  10

        return spikeOut


# In[7]:


# Define the cuda device to run the code on.
# device = torch.device('cuda')
# Use multiple GPU's if available
device = torch.device('cuda:0') # should be the first GPU of deviceIDs
deviceIds = [0, 1]

# Create network instance.
# net = Network(netParams).to(device)
# Split the network to run over multiple GPUs
net = torch.nn.DataParallel(Network(netParams).to(device), device_ids=deviceIds)

# Create snn loss instance.
error = snn.loss(netParams).to(device)

# Define optimizer module.
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01, amsgrad = True)
    
# Dataset and dataLoader instances.
trainingSet = nmnistDataset(datasetPath =netParams['training']['path']['in'], 
                            sampleFile  =netParams['training']['path']['train'],
                            samplingTime=netParams['simulation']['Ts'],
                            sampleLength=netParams['simulation']['tSample'])
trainLoader = DataLoader(dataset=trainingSet, batch_size=8, shuffle=False, num_workers=4)

testingSet = nmnistDataset(datasetPath  =netParams['training']['path']['in'], 
                            sampleFile  =netParams['training']['path']['test'],
                            samplingTime=netParams['simulation']['Ts'],
                            sampleLength=netParams['simulation']['tSample'])
testLoader = DataLoader(dataset=testingSet, batch_size=8, shuffle=False, num_workers=4)

# Learning stats instance.
stats = learningStats()


# ## Visualize

# In[8]:


#input, target, label = trainingSet[0]
#anim = snn.io.animTD(snn.io.spikeArrayToEvent(input.reshape((2, 320, 320, -1)).cpu().data.numpy()))
#HTML(anim.to_jshtml())


# # Train the network

# In[ ]:


for epoch in range(100):
    # Reset training stats.
    stats.training.reset()
    tSt = datetime.now()

    # Training loop.
    for i, (input, target, label) in enumerate(trainLoader, 0):
        # Move the input and target to correct GPU.
        input  = input.to(device)
        target = target.to(device) 

        # Forward pass of the network.
        output = net.forward(input)

        # Gather the training stats.
        stats.training.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
        stats.training.numSamples     += len(label)

        # Calculate loss.
        loss = error.numSpikes(output, target)

        # Reset gradients to zero.
        optimizer.zero_grad()

        # Backward pass of the network.
        loss.backward()

        # Update weights.
        optimizer.step()

        # Gather training loss stats.
        stats.training.lossSum += loss.cpu().data.item()

        # Display training stats. (Suitable for normal python implementation)
        # stats.print(epoch, i, (datetime.now() - tSt).total_seconds())

    # Update training stats.
    stats.training.update()
    # Reset testing stats.
    stats.testing.reset()

    # Testing loop.
    # Same steps as Training loops except loss backpropagation and weight update.
    for i, (input, target, label) in enumerate(testLoader, 0):
        input  = input.to(device)
        target = target.to(device) 

        output = net.forward(input)

        stats.testing.correctSamples += torch.sum( snn.predict.getClass(output) == label ).data.item()
        stats.testing.numSamples     += len(label)

        loss = error.numSpikes(output, target)
        stats.testing.lossSum += loss.cpu().data.item()
        # stats.print(epoch, i)

    # Update testing stats.
    stats.testing.update()
    if epoch%10==0:  stats.print(epoch, timeElapsed=(datetime.now() - tSt).total_seconds())


# ## Plot

# In[ ]:


plt.figure(1)
plt.semilogy(stats.training.lossLog, label='Training')
plt.semilogy(stats.testing .lossLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.figure(2)
plt.plot(stats.training.accuracyLog, label='Training')
plt.plot(stats.testing .accuracyLog, label='Testing')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

