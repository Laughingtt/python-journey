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
    plt_scatter("example/score_df.csv")
