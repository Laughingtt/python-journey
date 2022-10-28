#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:sgd.py
@time:2021/04/22

"""
# 随机梯度下降SGD
# 以 y=x1+2*x2为例

import numpy as np


# 多元数据
def sgd():
    # 训练集，每个样本有2个分量
    x = np.array([(1, 1), (1, 2), (2, 2), (3, 1), (1, 3), (2, 4), (2, 3), (3, 3)])
    y = np.array([3, 5, 6, 5, 7, 10, 8, 9])

    # 初始化
    m, dim = x.shape
    theta = np.zeros(dim)  # 参数
    alpha = 0.01  # 学习率
    threshold = 0.0001  # 停止迭代的错误阈值
    iterations = 1500  # 迭代次数
    error = 0  # 初始错误为0

    # 迭代开始
    for i in range(iterations):

        error = 1 / (2 * m) * np.dot((np.dot(x, theta) - y).T, (np.dot(x, theta) - y))
        # 迭代停止
        if abs(error) <= threshold:
            break

        j = np.random.randint(0, m)

        theta -= alpha * (x[j] * (np.dot(x[j], theta) - y[j]))

    print('迭代次数：%d' % (i + 1), 'theta：', theta, 'error：%f' % error)


if __name__ == '__main__':
    sgd()
