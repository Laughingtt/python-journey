from ml.datasets import TabularMinimal
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# 重写Pytorch Dataset 类里面的 init len getitem 方法
class DatasetFromNumPy(Dataset):
    def __init__(self, x_data, label):
        self.x_data = x_data
        self.label = label

    def __len__(self):
        return len(self.label)  # 返回长度

    def __getitem__(self, idx):
        # 这里可以对数据进行处理,比如讲字符数值化
        features = self.x_data[idx]  # 索引到第idx行的数据
        label = self.label[idx]  # 最后一项为指标数据
        return features, label  # 返回特征和指标


minimal_data = TabularMinimal()

# DatasetFromNumPy 实例化，生成一个 DataLoader
train_dataset = DatasetFromNumPy(minimal_data.x_train, minimal_data.y_train)
train_loader = DataLoader(train_dataset, batch_size=minimal_data.train_size, shuffle=True)  # batch_size为每次读取的样本数量，shuffle是否选择随机读取数据
# 测试 len 方法
len(train_dataset)
# 测试 getitem 方法
# 打印 DataLoader 里面的内容
for data in train_loader:
    print(data[0])
    print(data[1])
    break
