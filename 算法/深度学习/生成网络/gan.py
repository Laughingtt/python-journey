import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

# 定义生成器网络
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

# 定义超参数
input_size = 784 # MNIST图像大小为28x28，因此输入大小为784
hidden_size = 256
output_size = 1 # 判别器的输出为一个二进制值（真/假）
num_epochs = 200
batch_size = 100
learning_rate = 0.0002

# 加载MNIST数据集
train_dataset = dsets.MNIST(root='./data/',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# 定义生成器和判别器
G = Generator(input_size, hidden_size, output_size)
D = Discriminator(input_size, hidden_size, output_size)

# 定义损失函数和优化器
criterion = nn.BCELoss() # 二元交叉熵损失函数
G_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)
D_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)

# 训练GAN
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        # 生成真样本和假样本的标签
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # 训练判别器
        real_images = Variable(images.view(batch_size, -1))
        real_outputs = D(real_images)
        real_loss = criterion(real_outputs, real_labels)

        z = Variable
        # 训练生成器
        z = Variable(torch.randn(batch_size, input_size)) # 生成随机噪声
        fake_images = G(z)
        fake_outputs = D(fake_images)
        fake_loss = criterion(fake_outputs, fake_labels)

        G_loss = fake_loss

        D.zero_grad()
        real_loss.backward()
        D_optimizer.step()

        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [%d/%d], Step [%d/%d], d_loss: %.4f, g_loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, real_loss.data[0], G_loss.data[0]))

# 保存生成器模型
torch.save(G.state_dict(), 'generator.pkl')
