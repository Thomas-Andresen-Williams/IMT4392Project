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
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomHorizontalFlip, RandomCrop


batch_size = 4
epochs = 5
lr = 3e-5
gamma = 0.7
seed = 42

transform_training_data = Compose([RandomCrop(32, padding=4), Resize((224)), RandomHorizontalFlip(), ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

train_dataset = torchvision.datasets.CIFAR10(root='./dataCIFAR10/',
                                           train=True,
                                           transform=transform_training_data,
                                           download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./dataCIFAR10/',
                                          train=False,
                                          transform=transform_training_data)
# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                         shuffle=False)
indices = torch.arange(0,100)
indices_test = torch.arange(0,10)
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

# We know that MNIST images are 28 pixels in each dimension.
img_size = 32
# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size
# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)
# Number of colour channels for the images: 1 channel for gray-scale.
n_channels = 3
# Number of classes
n_classes = 10
#Using GPU or CPU depending on what is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

efficient_transformer = Linformer(
    dim=128,
    seq_len=49 + 1,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=64
)

model = ViT(
    dim=128,
    image_size=224,
    patch_size=32,
    num_classes=10,
    transformer=efficient_transformer,
    channels=3,
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


