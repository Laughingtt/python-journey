from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import numpy as np

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义基本模型
k = "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"
estimators = [
    ('dt', DecisionTreeClassifier(max_depth=5)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svm', SVC(kernel='rbf', probability=True, random_state=42))
]

# 定义元模型
meta_estimator = LogisticRegression(random_state=42)

# 定义 k-fold 交叉验证折数
n_splits = 5

# 定义训练集和测试集的预测结果保存列表
train_preds, test_preds = [], []

# 使用 KFold 进行交叉验证
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

for train_index, test_index in kf.split(X_train):
    # 将数据集分为训练集和验证集
    X_train_cv, X_valid_cv = X_train[train_index], X_train[test_index]
    y_train_cv, y_valid_cv = y_train[train_index], y_train[test_index]

    # 训练基本模型
    for estimator_name, estimator in estimators:
        estimator.fit(X_train_cv, y_train_cv)

        # 使用基本模型预测验证集
        y_valid_pred = estimator.predict(X_valid_cv)
        train_preds.append(y_valid_pred)

        # 使用基本模型预测测试集
        y_test_pred = estimator.predict(X_test)
        test_preds.append(y_test_pred)

# 将预测结果转化为二维数组
train_preds = np.asarray(train_preds).T
test_preds = np.asarray(test_preds).T

# 训练元模型
meta_estimator.fit(train_preds, y_train)

# 预测训练集和测试集
train_meta_pred = meta_estimator.predict(train_preds)
test_meta_pred = meta_estimator.predict(test_preds)

# 输出训练集和测试集的准确率
print('Train Accuracy: {:.2f}%'.format(accuracy_score(y_train, train_meta_pred) * 100))
print('Test Accuracy: {:.2f}%'.format(accuracy_score(y_test, test_meta_pred) * 100))
