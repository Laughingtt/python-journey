import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plt_scatter(score_path):
    score_df = pd.read_csv(score_path)

    # 生成x和y值
    x = score_df["train_id"].to_numpy()
    y = score_df["score"].to_numpy()

    # 绘制图形
    plt.scatter(x, y, s=10, alpha=0.8)

    plt.scatter(x[0], y[0], color='red', s=20)
    plt.annotate("max score:{}".format(round(y[0], 2)), xy=(x[0], y[0]), xytext=(x[0], y[0] + 0.05))

    plt.title('Test Score')
    plt.xlabel('train id')
    plt.ylabel('score')

    plt.xlim([0, len(score_df) + 2])
    plt.ylim([0, 1])

    plt.xticks(np.arange(0, len(score_df) + 2, 5))

    plt.show()


def plt_scatter_all_model(result_path,is_sort=True):
    score_path = os.listdir(result_path)
    color_lis = ["green", "blue", "violet", "coral"]
    type_lsit = []
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    for c, score_p in enumerate(score_path):
        score_df = pd.read_csv(os.path.join(result_path, score_p))

        # 生成x和y值
        x = score_df["train_id"].to_numpy()
        y = score_df["score"].to_numpy()
        if is_sort is True:
            x = range(len(y))

        # 绘制图形
        _type = ax.scatter(x, y, s=8, alpha=0.8, c=color_lis[c])
        ax.annotate("{}".format(round(y[0], 2)), xy=(x[0], y[0]), xytext=(x[0]-0.1, y[0]))

        type_lsit.append(_type)

    # ax.legend((type_lsit[0], type_lsit[1], type_lsit[2], type_lsit[3]),
    #           ("DidntLike", "SmallDoses", "LargeDoses", "223"), loc=0)

    ax.legend(tuple(type_lsit), tuple(score_path), loc=4)

    # ax.title('Test Score')
    # ax.xlabel('train id')
    # ax.ylabel('score')

    # ax.xlim([0, len(score_df) + 2])
    # ax.ylim([0, 1])

    # ax.xticks(np.arange(0, len(score_df) + 2, 5))
    plt.show()


def plt_nn_learning_curve(results_grad):
    ax = None
    for train_id, result in enumerate(results_grad):
        if result.metrics_dataframe is None:
            continue
        _label = f"id:{train_id}\n" \
                 f"lr={result.config['lr']:.4f}, " \
                 f"momentum={result.config['momentum']:.4f}," \
                 f" hidden_size={[result.config['hidden_size'] for i in range(result.config['hidden_length'])]}"
        print(_label)
        label = f"id:{train_id}"
        if ax is None:
            ax = result.metrics_dataframe.plot("training_iteration", "score", label=label)
        else:
            result.metrics_dataframe.plot("training_iteration", "score", ax=ax, label=label)

    ax.set_title("Mean Accuracy vs. Training Iteration for All Trials")
    ax.set_ylabel("Mean Test Accuracy")

    import matplotlib.pyplot as plt

    # 将label平移
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.show()


if __name__ == '__main__':
    # plt_scatter("../example/score_df.csv")
    plt_scatter_all_model("/Users/tianjian/Projects/python-BasicUsage2/package/auto_ml/ray_tune/ml/example/tmp_result")
