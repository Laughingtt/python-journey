#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： tian
# datetime： 2023/1/28 10:39 AM 
# ide： PyCharm

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import torch, torchvision

model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

print(data)

prediction = model(data)  # forward pass

loss = (prediction - labels).sum()
loss.backward()  # backward pass

optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

optim.step()  # gradient descent

