import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os

"""
这个简单的GAN网络使用MNIST数据集训练，可以在PyTorch上轻松实现。在训练过程中，首先初始化生成器和判别器网络，并定义优化器和损失函数。然后通过循环迭代训练数据集，每个迭代中先训练判别器，再训练生成器，最后输出训练信息并保存模型。

"""

# 定义生成器网络
class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)


# 定义判别器网络
class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x)


# 定义训练函数
def train(gen, disc, dataloader, epochs, z_dim, lr, device):
    # 初始化优化器和损失函数
    opt_gen = optim.Adam(gen.parameters(), lr=lr)
    opt_disc = optim.Adam(disc.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            # 初始化数据
            real = real.view(-1, 784).to(device)
            batch_size = real.shape[0]
            # 定义真假标签
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)

            # 训练判别器
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_real = disc(real)
            disc_fake = disc(fake.detach())
            loss_disc_real = criterion(disc_real, real_label)
            loss_disc_fake = criterion(disc_fake, fake_label)
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # 训练生成器
            noise = torch.randn(batch_size, z_dim).to(device)
            fake = gen(noise)
            disc_fake = disc(fake)
            loss_gen = criterion(disc_fake, real_label)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            # 输出训练信息
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx}/{len(dataloader)}] Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f}")

    # 保存模型
    if not os.path.exists("models/"):
        os.makedirs("models/")
    torch.save(gen.state_dict(), "models/generator.pth")
    torch.save(disc.state_dict(), "models/discriminator.pth")


if __name__ == '__main__':
    # 超参数定义
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z_dim = 64
    lr = 0.0002
    batch_size = 128
    epochs = 10
    img_dim = 784

    # 下载数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 初始化网络
    gen = Generator(z_dim, img_dim).to(device)
    disc = Discriminator(img_dim).to(device)

    # 训练网络
    train(gen, disc, dataloader, epochs, z_dim, lr, device)

