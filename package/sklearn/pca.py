"""可解释性方差贡献率曲线"""
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 加载数据集
iris = load_iris()
X = iris.data
y = iris.target
# 2. 调用PCA
pca = PCA().fit(X)  # 默认n_components为特征数目4
pca_info = pca.explained_variance_ratio_
# print("每个特征在原始数据信息占比：\n", pca_info)
pca_info_sum = np.cumsum(pca_info)
# print("前i个特征总共在原始数据信息占比：\n", pca_info_sum)

plt.plot([1, 2, 3, 4], pca_info_sum)  # [1, 2, 3, 4]表示选1个特征、2个特征...
plt.xticks([1, 2, 3, 4])  # 限制坐标长度
plt.xlabel('The number of features after dimension')
plt.ylabel('The sum of explained_variance_ratio_')
plt.show()


