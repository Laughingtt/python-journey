import csv
import random
from datetime import datetime, timedelta


# 生成随机字符串
def generate_random_string(length):
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join(random.choice(letters) for _ in range(length))


# 生成随机日期
def generate_random_date(start_date, end_date):
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    days = (end - start).days
    random_days = random.randint(0, days)
    return (start + timedelta(days=random_days)).strftime('%Y-%m-%d')


# 生成随机小数
def generate_random_float(min_value, max_value):
    return round(random.uniform(min_value, max_value), 2)


# 生成随机整数
def generate_random_integer(min_value, max_value):
    return random.randint(min_value, max_value)


# 生成数据集
def generate_dataset(num_records):
    dataset = []
    for _ in range(num_records):
        name = generate_random_string(8)
        dob = generate_random_date('1980-01-01', '2003-12-31')
        credit_score = generate_random_integer(300, 850)
        balance = generate_random_float(0, 10000)

        record = {
            'Name': name,
            'Date of Birth': dob,
            'Credit Score': credit_score,
            'Balance': balance
        }

        dataset.append(record)

    return dataset


# 设置要生成的记录数
num_records = 100

# 生成数据集
data = generate_dataset(num_records)

# 设置CSV文件的列名
fieldnames = ['Name', 'Date of Birth', 'Credit Score', 'Balance']

# 写入数据到CSV文件
with open('data.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # 写入列名
    writer.writeheader()

    # 写入数据记录
    for record in data:
        writer.writerow(record)

print("数据已成功生成并保存到 data.csv 文件中。")
