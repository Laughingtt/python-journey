import pandas as pd
import numpy as np
import datetime

# 参数设置
num_records = 10 ** 7  # 生成1000万条记录
num_users = 10 ** 6  # 100万用户
num_products = 10 ** 5  # 10万种商品

# 生成数据
np.random.seed(42)  # 保证结果可重复
transaction_ids = np.arange(1, num_records + 1)
user_ids = np.random.randint(1, num_users + 1, num_records)
product_ids = np.random.randint(1, num_products + 1, num_records)
amounts = np.round(np.random.uniform(1, 1000, num_records), 2)  # 交易金额在1到1000之间，保留两位小数
transaction_times = pd.to_datetime(np.random.randint(
    (datetime.datetime(2023, 1, 1) - datetime.datetime(1970, 1, 1)).total_seconds(),
    (datetime.datetime(2024, 1, 1) - datetime.datetime(1970, 1, 1)).total_seconds(),
    num_records
), unit='s')

# 创建DataFrame
df = pd.DataFrame({
    'transaction_id': transaction_ids,
    'user_id': user_ids,
    'product_id': product_ids,
    'amount': amounts,
    'transaction_time': transaction_times
})

# 保存为CSV文件
df.to_csv('/mnt/data/transactions.csv', index=False)
