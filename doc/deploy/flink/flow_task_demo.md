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
select behavior, user_id, count(*) as cnt
from KafkaTable
group by behavior, user_id
```