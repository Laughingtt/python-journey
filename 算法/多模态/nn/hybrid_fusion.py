
"""
混合融合（Hybrid Fusion）是指将多个模态的输入数据（如图像和文本）在模型的不同层级进行融合，最终得到联合特征，再送入分类器进行分类。与早期融合和晚期融合相比，混合融合可以更好地捕捉不同模态之间的互动关系，从而提高模型的性能。

在 PyTorch 中实现混合融合，可以通过定义多个模型来分别处理不同的数据类型，然后在模型的不同层级将它们的输出特征进行融合，最终送入分类器进行分类。

以下是一个示例代码：
"""


import torch
import torch.nn as nn

# 定义图像模型
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 6 * 6, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 64 * 6 * 6)
        x = self.fc1(x)
        return x

# 定义文本模型
class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(1000, 128)
        self.lstm = nn.LSTM(128, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(256 * 2, 512)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        return x

# 定义联合模型
class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.image_model = ImageModel()
        self.text_model = TextModel()
        self.fc1 = nn.Linear(512 + 512, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x1, x2):
        x1 = self.image_model(x1)
        x2 = self.text_model(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

# 定义数据集和数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
