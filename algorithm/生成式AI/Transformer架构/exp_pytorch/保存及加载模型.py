"""
保存和加载模型权重
Pytorch 模型会将所有参数存储在一个状态字典 (state dictionary) 中，可以通过 Model.state_dict() 加载。Pytorch 通过 torch.save() 保存模型权重：

"""
import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')

"""
保存和加载完整模型
上面存储模型权重的方式虽然可以节省空间，但是加载前需要构建一个结构完全相同的模型实例来承接权重。如果我们希望在存储权重的同时，也一起保存模型结构，就需要将整个模型传给 torch.save() ：

"""

import torch
import torchvision.models as models

model = models.vgg16(pretrained=True)
torch.save(model, 'model.pth')
# 这样就可以直接从保存的文件中加载整个模型（包括权重和结构）：

model = torch.load('model.pth')
