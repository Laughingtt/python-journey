
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# 加载数据集
iris = load_iris()
X, y = iris.data, iris.target

# 定义三个基分类器
clf1 = LogisticRegression(random_state=1)
clf2 = GaussianNB()
clf3 = SVC(kernel='rbf', probability=True, random_state=1)

# 定义 VotingClassifier 元分类器
ensemble = VotingClassifier(estimators=[('lr', clf1), ('nb', clf2), ('svm', clf3)], voting='soft')

# 训练元分类器
ensemble.fit(X, y)

# 预测结果
y_pred = ensemble.predict(X)
