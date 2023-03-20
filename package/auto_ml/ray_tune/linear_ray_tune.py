import torch
import torch.nn as nn
import numpy as np
from ray import tune


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


def train_model(config):  # 修改1：修改参数，所有的参数都要借助config传递
    # 指定参数与损失函数
    model = LinearRegressionModel(x_train.shape[1], config['linear1'], config['linear2'], 1)
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
    mean_loss = np.mean(loss_list)
    tune.report(my_loss=mean_loss)


if __name__ == '__main__':
    x_train = torch.randn(100, 4)  # 生成100个4维的随机数，作为训练集的 X
    y_train = torch.randn(100, 1)  # 作为训练集的label
    # train_model(x_train, y_train, 32, 8) # 修改2：就不需要这样启动了，注释掉这一行

    # 修改3：下面就是封装的方法
    config = {
        "linear1": tune.sample_from(lambda _: np.random.randint(2, 5)),  # 自定义采样
        "linear2": tune.choice([2, 4, 8, 16]),  # 从给定值中随机选择
    }
    results = tune.run(  # 执行训练过程，执行到这里就会根据config自动调参了
        train_model,  # 要训练的模型
        resources_per_trial={"cpu": 8, },  # 指定训练资源
        config=config,
        num_samples=10,  # 调参个数
    )
    # 得到最后的结果
    print("======================== Result =========================")
    print(results.results_df)
