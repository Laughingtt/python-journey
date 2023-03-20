import torch
import torch.nn as nn
import numpy as np


class LinearRegressionModel(nn.Module):
    def __init__(self, input_shape, linear1, linear2, output_shape):
        super(LinearRegressionModel, self).__init__()
        self.linear1 = nn.Linear(input_shape, linear1)
        self.linear2 = nn.Linear(linear1, linear2)
        self.linear3 = nn.Linear(linear2, output_shape)

    def forward(self, x):
        l1 = self.linear1(x)
        l2 = self.linear2(l1)
        l3 = self.linear3(l2)
        return l3


def train_model(x_train, y_train, linear1, linear2):
    # 指定参数与损失函数
    model = LinearRegressionModel(x_train.shape[1], linear1, linear2, 1)
    epochs = 1000  # 迭代1000次
    learning_rate = 0.01  # 学习率
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 优化函数
    criterion = nn.MSELoss()  # Loss使用MSE值，目标是使MSE最小

    loss_list = []
    for epoch in range(epochs):
        epoch += 1
        optimizer.zero_grad()  # 梯度清零
        outputs = model(x_train)  # 前向传播
        loss = criterion(outputs, y_train)  # 计算损失
        loss.backward()  # 返向传播
        loss_list.append(loss.detach().numpy())
        optimizer.step()  # 更新权重参数
        if epoch%100 == 0:
            print("loss: ", loss)
    mean_loss = np.mean(loss_list)
    print("loss: ", mean_loss)


if __name__ == '__main__':
    x_train = torch.randn(100, 4)  # 生成100个4维的随机数，作为训练集的 X
    y_train = torch.randn(100, 1)  # 作为训练集的label
    train_model(x_train, y_train, 32, 8)
