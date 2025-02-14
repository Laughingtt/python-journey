import numpy as np


class Optimizers():
    def __init__(self, alpha=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def adam(self, dx):
        if self.m is None:
            self.m = np.zeros_like(dx)
            self.v = np.zeros_like(dx)
        self.t += 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * dx
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * dx ** 2
        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)
        delta_x = -self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return delta_x

    def adagrad(self, dx):
        if self.m is None:
            self.m = np.zeros_like(dx)
        self.m += dx ** 2
        delta_x = -self.alpha * dx / (np.sqrt(self.m) + self.epsilon)
        return delta_x

    def rmsprop(self, dx):
        if self.m is None:
            self.m = np.zeros_like(dx)
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * dx ** 2
        delta_x = -self.alpha * dx / (np.sqrt(self.m) + self.epsilon)
        return delta_x

    def sgd(self, dx):
        delta_x = -self.alpha * dx
        return delta_x

    def gd(self, dx):
        return self.sgd(dx)


if __name__ == '__main__':
    w = np.random.randn(10, 1)
    x = np.random.randn(100, 10)
    y = np.random.randn(100, 1)

    optimizers = Optimizers(alpha=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

    for i in range(100):
        # 计算梯度
        dx = np.dot(x.T, np.dot(x, w) - y)

        # 使用优化器更新参数
        delta_x = optimizers.adam(dx)
        w += delta_x
