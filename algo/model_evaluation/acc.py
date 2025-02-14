from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,precision_recall_curve

# 预测标签和真实标签
y_true = [0, 1, 0, 1, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1, 0, 1]

# 计算准确率
accuracy = accuracy_score(y_true, y_pred)
print("Accuracy:", accuracy)

# 计算精确率
precision = precision_score(y_true, y_pred)
print("Precision:", precision)

# 计算召回率
recall = recall_score(y_true, y_pred)
print("Recall:", recall)

# 计算F1得分
f1 = f1_score(y_true, y_pred)
print("F1 score:", f1)
