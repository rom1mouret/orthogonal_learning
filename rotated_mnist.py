#!/usr/bin/env python3

import os
import sys
from tempfile import gettempdir
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from orthocl import (
    GradProjLinear,
    GradProjConv2d,
    proj_computation,
    gradproj_layers,
    GPM,
    NullSpace,
    LowPlasticity,
    ZeroPlasticity
)

dim = 28 ** 2
angles = [0, 90, -45, 45]  # i.e. the tasks

# set 'type' to 0, 1, 2 or 3 to switch between the experiments:
type = 1
# switch between projection algorithm here:
proj_algorithm = lambda: NullSpace(R=0.01)
#proj_algorithm = lambda: LowPlasticity()
#proj_algorithm = lambda: GPM(R=R)
#proj_algorithm = lambda : ZeroPlasticity()

if type == 0:
    print("** flat  **")
    classifier = nn.Sequential(
        nn.Flatten(),
        GradProjLinear(dim, 60, proj_algorithm()),
        nn.LeakyReLU(0.1),
        GradProjLinear(60, 32, proj_algorithm()),
        nn.LeakyReLU(0.1),
        GradProjLinear(32, 10, proj_algorithm())
    )
elif type == 1:
    print("** Conv2d **")
    classifier = nn.Sequential(
        GradProjConv2d(1, 32, 5, proj_algorithm()),
        nn.AdaptiveMaxPool2d((4, 4)),
        nn.Flatten(),
        GradProjLinear(32 * 4 * 4, 32, proj_algorithm()),
        nn.LeakyReLU(0.1),
        GradProjLinear(32, 10, proj_algorithm())
    )
elif type == 2:
    print("** Flat zero stability **")
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(dim, 60),
        nn.LeakyReLU(0.1),
        nn.Linear(60, 32),
        nn.LeakyReLU(0.1),
        nn.Linear(32, 10)
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
else:
    print("** Conv zero stability **")
    classifier = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=5, bias=False),
        nn.AdaptiveMaxPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 32),
        nn.LeakyReLU(0.1),
        nn.Linear(32, 10)
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)


tr = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.15], std=[0.3]),
])
prefix = os.path.join(gettempdir(), "mnist_")
train_set = datasets.MNIST(prefix+"training", download=True, train=True, transform=tr)
test_set = datasets.MNIST(prefix+"testing", download=True, train=False, transform=tr)

# training
loss_func = nn.CrossEntropyLoss()
for angle in tqdm(angles, total=len(angles)):
    # take a snapshot at the beginning of each task
    for layer in gradproj_layers(classifier):
        layer.take_snapshot()
    for epoch in range(1):
        loader = torch.utils.data.DataLoader(train_set, batch_size=8, shuffle=True)
        for images, labels in loader:
            images = TF.rotate(images, angle)
            y_pred = classifier(images)
            loss_func(y_pred, labels).backward()
            nn.utils.clip_grad_value_(classifier.parameters(), 15)
            if type in (0, 1):
                for layer in gradproj_layers(classifier):
                    layer.sgd_step(lr=0.005)
            else:
                optimizer.step()
                optimizer.zero_grad()

    loader = torch.utils.data.DataLoader(train_set, batch_size=512, shuffle=False)
    if type in (0, 1):
        with proj_computation(classifier):
            for images, labels in loader:
                images = TF.rotate(images, angle)
                y_pred = classifier(images)

# evaluation
total_acc = 0
with torch.no_grad():
    for angle in tqdm(angles, total=len(angles)):
        loader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False)
        acc = 0
        total = 0
        for images, labels in loader:
            images = TF.rotate(images, angle)
            _, y_pred = classifier(images).max(dim=1)
            acc += (y_pred == labels).float().sum()
            total += images.size(0)
        acc /= total
        print("acc", acc)
        total_acc += acc

total_acc /= len(angles)
print("total_acc", total_acc)
