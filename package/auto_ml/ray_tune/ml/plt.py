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

    plt.scatter(x[0], y[0], color='red', s=50)

    plt.title('Mean Test Accuracy')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim([0, len(score_df) + 2])
    plt.ylim([0, 1])

    plt.xticks(np.arange(0, len(score_df) + 2, 5))

    plt.show()


if __name__ == '__main__':
    plt_scatter("score_df.csv")