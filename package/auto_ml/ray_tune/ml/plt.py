import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

score_df = pd.read_csv("score_df.csv")

# 生成x和y值
x = score_df["train_id"]
y = score_df["score"]

# 绘制图形
plt.scatter(x, y, s=10, alpha=0.8)
plt.title('Mean Test Accuracy')
plt.xlabel('x')
plt.ylabel('y')

plt.xlim([0, len(score_df) + 2])
plt.ylim([0, 1])

plt.xticks(np.arange(0, len(score_df) + 2, 5))

plt.show()
