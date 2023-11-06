import mysql.connector


# sql_alchemy_conn = mysql+mysqlconnector://root:JS@$wj921@kdf@192.168.111.27:3306/airflow
# 创建一个MySQL连接
conn = mysql.connector.connect(
  host="localhost",
  user="yourusername",
  password="yourpassword",
)

# 打印连接状态
print(conn)

cursor = conn.cursor()

cursor.execute('show databases')

conn.close()