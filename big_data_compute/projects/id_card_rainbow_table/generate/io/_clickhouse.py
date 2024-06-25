import clickhouse_driver

# 连接到ClickHouse数据库
conn = clickhouse_driver.connect(
    host='localhost',
    port=9000,
    user='default',
    database='ck_test',
    password=''
)

# 创建表
create_table_query = '''  
CREATE TABLE if not exists my_table (  
    column1 UInt8,  
    column2 String,  
    column3 Float32  
) ENGINE = Memory  
'''
conn.cursor().execute(create_table_query)

# 将数据写入表
data = [
    (1, 'apple', 1.2),
    (2, 'banana', 2.3),
    (3, 'orange', 3.4),
]
insert_query = 'INSERT INTO my_table (column1, column2, column3) VALUES'
insert_query += ','.join(['({0}, \'{1}\', {2})'.format(row[0], row[1], row[2]) for row in data])
conn.cursor().execute(insert_query)

# 提交更改并关闭连接
conn.commit()
conn.close()