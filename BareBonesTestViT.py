from __future__ import print_function
import torch
import numpy
import glob
from itertools import chain
import os
import random
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from tqdm import tqdm
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from vit_pytorch.efficient import ViT
from train import train
# Training settings
batch_size = 15
epochs = 5
lr = 3e-5
gamma = 0.7
seed = 42


train_dataset = torchvision.datasets.MNIST(root='./data/',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data/',
                                          train=False,
                                          transform=transforms.ToTensor())
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)
indices = torch.arange(0,1000)
indices_test = torch.arange(0,20)
train_dataset_short = torch.utils.data.Subset(train_dataset,indices)

test_dataset_short = torch.utils.data.Subset(test_dataset,indices_test)


train_loader_short = torch.utils.data.DataLoader(dataset=train_dataset_short,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader_short = torch.utils.data.DataLoader(dataset=test_dataset_short,
                                                batch_size=batch_size,
                                                shuffle=False)

#Defining network parameters




patch_size=4
embed_dim=96 # dimensionality of the latent space
n_attention_heads=4 #number of heads to be used")
forward_mul=2 #forward multiplier
n_layers=6 #number of encoder layers
num_channels = 1

# We know that MNIST images are 28 pixels in each dimension.
img_size = 28
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
# Number of colour channels for the images: 1 channel for gray-scale.
n_channels = 1
# Number of classes, one class for each of 10 digits.
n_classes = 10
#Using GPU or CPU depending on what is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

efficient_transformer = Linformer(
    dim=96,
    seq_len=49+1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=4,
    k=64
)

model = ViT(
    dim=96,
    image_size=28,
    patch_size=4,
    num_classes=10,
    transformer=efficient_transformer,
    channels=1,
).to(device)

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

train(model,optimizer, criterion, train_loader_short,epochs,device)

correctPred = 0

for data, label in tqdm(test_loader_short):
    output = model(data)
    acc = []
    correctPred = 0
    for i in range(len(label)):
        if(torch.argmax(output[i])==label[i]):
            correctPred += 1
        print("Prediction: ", torch.argmax(output[i]), "True value: ", label[i])
    acc.append(correctPred/len(label))

print(sum(acc)/len(acc))

