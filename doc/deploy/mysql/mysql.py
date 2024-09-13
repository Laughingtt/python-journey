import pymysql


conn = pymysql.connect(
    host='172.27.27.139',  # MySQL服务器地址
    port=3306,         # MySQL服务器端口号
    user='test',       # 用户名
    password='test', # 密码
    database='rmjr', # 数据库名称
    charset='utf8'     # 字符编码
)

# 创建连接
conn = pymysql.connect(
    host='172.27.27.139',  # MySQL服务器地址
    port=13306,         # MySQL服务器端口号
    user='datahub',       # 用户名
    password='datahub', # 密码
    database='datahub', # 数据库名称
    charset='utf8'     # 字符编码
)

# 创建游标
cursor = conn.cursor()

# 执行SQL语句
cursor.execute("SELECT VERSION()")

# 获取单条数据
data = cursor.fetchone()
print('Database version:', data)

# 关闭游标和连接
cursor.close()
conn.close()


创建用户
CREATE USER 'test'@'%' IDENTIFIED BY 'test';

赋权
GRANT ALL PRIVILEGES ON *.* TO 'test'@'%' WITH GRANT OPTION;

刷新
FLUSH PRIVILEGES;

修改root密码

修改配置
sudo vim /etc/mysql/mysql.conf.d/mysqld.cnf

增加 跳过安全验证
skip-grant-tables

重启服务

sudo service mysql restart

root进入
mysql -u root   回车

use mysql
alter user 'root'@'localhost' identified with mysql_native_password by '123456';      --修改密码为123456

