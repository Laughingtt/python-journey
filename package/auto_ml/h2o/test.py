from h2o.automl import H2OAutoML
import h2o
import pandas as pd

h2o.init()


# pandas导入方式
train = pd.read_csv('/Users/tianjian/Projects/python-BasicUsage/算法/data/my_data_guest.csv')
test = pd.read_csv('/Users/tianjian/Projects/python-BasicUsage/算法/data/my_data_guest.csv')
hf = h2o.H2OFrame(train)
test_hf = h2o.H2OFrame(test)

hf.head()

# 选择预测变量和目标
hf['bad'] = hf['bad'].asfactor()
predictors = hf.drop('bad').columns
response = 'bad'

# 切分数据集，添加停止条件参数为最大模型数和最大时间，然后训练
train_hf, valid_hf = hf.split_frame(ratios=[.8], seed=1234)
aml = H2OAutoML(
    max_models=10,
    max_runtime_secs=30,
    seed=1234,
)

aml.train(x=predictors,
          y=response,
          training_frame=hf,
          )
