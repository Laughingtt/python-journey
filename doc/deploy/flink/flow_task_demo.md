## kafka 接收csv数据

1. 创建topic

```shell

bin/kafka-topics.sh --create --topic user_behavior --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1
```

2. 发送测试数据到topic

```shell
echo "1,1001,buy" | bin/kafka-console-producer.sh --topic user_behavior --bootstrap-server localhost:9092
echo "2,1002,browse" | bin/kafka-console-producer.sh --topic user_behavior --bootstrap-server localhost:9092
echo "3,1003,rse" | bin/kafka-console-producer.sh --topic user_behavior --bootstrap-server localhost:9092
```

```shell
bin/kafka-console-consumer.sh --topic user_behavior --from-beginning --bootstrap-server localhost:9092
```

3. 启动flink

flink加载kafka数据时,需要flink-kafka-connector,flink对应版本官网可下载,启动时加载即可

```shell
# 启动集群
./bin/start-cluster.sh

# 启动sql client
./bin/sql-client.sh -j lib/flink-sql-connector-kafka-1.17.2.jar
```

4. 创建flink流表接收kafka数据

```sql
CREATE TABLE KafkaTable
(
    `user_id` BIGINT,
    `item_id` BIGINT,
    `behavior` STRING
)
WITH ( 'connector' = 'kafka',
    'topic' = 'user_behavior',
    'properties.bootstrap.servers' = 'localhost:9092',
    'properties.group.id' = 'testGroup',
    'scan.startup.mode' = 'earliest-offset',
    'format' = 'csv');

-- 查询统计
select user_id,behavior, count(*) as cnt
from KafkaTable
group by user_id,behavior
```

5. 创建JDBC接收flink sink数据

mysql 创建表
```sql
CREATE TABLE sink_user
(
    `user_id` int,
    `behavior` varchar(30),
    `cnt` int
)

-- 设定主键
CREATE TABLE sink_user
(
    `user_id` int,
    `behavior` varchar(30),
    `cnt` int,
    primary key (user_id,behavior)
)

```

flink sql client 新建连接表
```sql
CREATE TABLE sink_user (
    `user_id` BIGINT,
    `behavior` STRING,
    `cnt` BIGINT,
    PRIMARY KEY (user_id,behavior) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:mysql://localhost:3306/test?useUnicode=true&characterEncoding=utf-8&serverTimezone=GMT%2B8&useSSL=false',
   'table-name' = 'sink_user',
    'username' = 'test', -- 用户名
    'password' = 'test' -- 密码
);
```

flink消费kafka的数据写入到mysql中
```sql
insert into sink_user
select user_id,behavior, count(*) as cnt
from KafkaTable
group by user_id,behavior;
```

查看mysql表数据
```sql
mysql> select * from sink_user;
+---------+----------+------+
| user_id | behavior | cnt  |
+---------+----------+------+
|       3 | chlis    |    1 |
|       1 | buy      |    4 |
|       2 | browse   |    1 |
|       1 | buy      |    5 |
|       1 | buy      |    6 |
+---------+----------+------+
```
没有建立主键表所以统计的数据以新增的方式进行累计,按照分组统计，设定主键