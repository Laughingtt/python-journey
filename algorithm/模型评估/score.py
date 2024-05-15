# 预测标签和真实标签
import numpy as np

import matplotlib.pyplot as plt

y_true = [0, 1, 0, 1, 1, 0, 1, 1]
y_pred = [0, 1, 0, 0, 1, 1, 0, 1]

"""
预测/真实	        1（Postive）	            0（Negative）
1 (Postive)	    TP（True Postive:真阳）	FP (False Postive:假阳)
0（Negative）	FN (False Negative:假阴)	TN (True Negative:真阴)

精确度（precision）/查准率：TP/（TP+FP）=TP/P    预测为真中，实际为正样本的概率

召回率（recall）/查全率：TP/（TP+FN）  正样本中，被识别为真的概率

假阳率（False positive rate）：FPR = FP/(FP+TN)  负样本中，被识别为真的概率

真阳率（True positive rate）：TPR = TP/（TP+FN）  正样本中，能被识别为真的概率

准确率（accuracy）：ACC =（TP+TN）/(P+N) 所有样本中，能被正确识别的概率
"""

"""
PR-曲线

召回率 Recall = TP/(TP+FN)  X轴
精准率 Precision = TP/(TP+FP)  Y轴

指标优势:
越靠近右上角，模型效果越好
"""


def get_tp_fp() -> object:
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for pred, label in zip(y_pred, y_true):
        if (label == 1) and (pred == 1):
            tp = tp + 1
        elif (label == 0) and (pred == 1):
            fp = fp + 1
        elif (label == 0) and (pred == 0):
            tn = tn + 1
        elif (label == 1) and (pred == 0):
            fn += 1
    return tp, fp, tn, fn


def get_recall_and_pre() -> object:
    tp, fp, tn, fn = get_tp_fp()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)

    print("recall", recall)
    print("precision", precision)

    return recall, precision


def get_pr_curve():
    # 计算PR曲线
    precision, recall, thresholds = [], [], []
    # 遍历不同阈值，计算精确率和召回率
    for threshold in np.linspace(0, 1, num=100):
        # 根据阈值将预测概率转换为类别
        y_pred2 = np.where(y_pred >= threshold, 1, 0)

        # 计算精确率、召回率和阈值
        tp = np.sum(np.logical_and(y_pred2 == 1, y_true == 1))
        fp = np.sum(np.logical_and(y_pred2 == 1, y_true == 0))
        fn = np.sum(np.logical_and(y_pred2 == 0, y_true == 1))

        if tp + fp == 0:
            precision.append(0)
        else:
            precision.append(tp / (tp + fp))

        if tp + fn == 0:
            recall.append(0)
        else:
            recall.append(tp / (tp + fn))

        thresholds.append(threshold)

    # 计算PR曲线下的面积
    pr_auc = np.trapz(precision[::-1], recall[::-1])

    # 绘制PR曲线
    plt.figure()
    plt.plot(recall, precision, lw=2, color='navy', label='PR curve (AUC = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Precision-Recall curve')
    plt.legend(loc="lower right")
    plt.show()


"""
F1score 指标

F1 score 是精确率和召回率的一个加权平均

F1_score = 2*(precision*recall)/(precision+recall)

指标优势：
Precision体现了模型对负样本的区分能力，Precision越高，模型对负样本的区分能力越强；
Recall体现了模型对正样本的识别能力，
Recall越高，模型对正样本的识别能力越强。
F1 score是两者的综合，F1 score越高，说明模型越稳健。
"""


def get_f1_score():
    recall, precision = get_recall_and_pre()

    f1_score = 2 * (precision * recall) / (precision + recall)

    print("f1_score", f1_score)


if __name__ == '__main__':
    get_f1_score()
    get_pr_curve()
