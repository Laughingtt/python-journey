"""
早期融合（Early Fusion）是指将不同模态的输入数据在输入层级进行融合，然后将联合特征送入分类器进行分类。早期融合的优点是计算简单，但缺点是可能会将一些不相关的特征合并在一起，导致模型性能下降。

在 PyTorch 中实现早期融合，可以通过将不同模态的输入数据在输入层级进行合并，然后将联合特征送入模型进行训练。
"""

import torch
import torch.nn as nn


# 定义联合模型
class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 16 * 16 + 1000, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = nn.functional.relu(x1)
        x1 = self.pool1(x1)
        x1 = x1.view(-1, 32 * 16 * 16)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x


# 定义数据集和数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 定义优化器和损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(10):
    for i, (images, texts, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # 在测试集上测试模型
    correct = 0
    total = 0
    with torch.no_grad():
        for images, texts, labels in test_loader:
            outputs = model(images, texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Epoch [{}/{}], Test Accuracy: {:.2f}%'
          .format(epoch + 1, 10, 100 * correct / total))

"""
在早期融合中，可以将不同模态的数据在数据集层级进行合并，然后将联合数据集传递给模型进行训练。在 PyTorch 中，可以通过自定义 Dataset 类来实现这个过程。

以下是一个示例代码，其中我们假设有两个模态的数据：图像数据和文本数据。
"""
import torch
from torch.utils.data import Dataset


# 定义早期融合数据集
class EarlyFusionDataset(Dataset):
    def __init__(self, image_data, text_data, labels):
        self.image_data = image_data
        self.text_data = text_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.image_data[index]
        text = self.text_data[index]
        label = self.labels[index]
        return (image, text, label)


# 加载数据
image_data = ...
text_data = ...
labels = ...
dataset = EarlyFusionDataset(image_data, text_data, labels)

# 划分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
