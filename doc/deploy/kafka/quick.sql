// --------------------------------------------------------------------------------------------
//  TODO 通过flinksql向kafka写入数据(写入时指定 timestamp)
// --------------------------------------------------------------------------------------------
drop table kafka_table_source_test_startup_mode;
CREATE TABLE kafka_table_source_test_startup_mode (
  `order_id` BIGINT,
  `price` DOUBLE
) WITH (
  'connector' = 'kafka',
  'topic' = 'my-replicated-topic',
  'properties.bootstrap.servers' = 'localhost:9092',
  'properties.group.id' = 'FlinkConsumer',
  'scan.startup.mode' = 'earliest-offset',
  'value.format' = 'csv'
);

insert into kafka_table_source_test_startup_mode(order_id, price, ts)
SELECT *
FROM (VALUES (1, 2.0, TO_TIMESTAMP_LTZ(1000, 3))
           , (2, 4.0, TO_TIMESTAMP_LTZ(2000, 3))
           , (3, 6.0, TO_TIMESTAMP_LTZ(3000, 3))
           , (4, 7.0, TO_TIMESTAMP_LTZ(4000, 3))
           , (5, 8.0, TO_TIMESTAMP_LTZ(5000, 3))
           , (6, 10.0, TO_TIMESTAMP_LTZ(6000, 3))
           , (7, 12.0, TO_TIMESTAMP_LTZ(7000, 3))) AS book (order_id, price, ts);


insert into kafka_table_source_test_startup_mode(order_id, price)
SELECT *
FROM (VALUES (1, 2.0)) AS book (order_id, price);

-- 触发读取kafka操作
select * from kafka_table_source_test_startup_mode;



CREATE TABLE kafka_sink (
    id STRING,
    val STRING,
    ts TIMESTAMP(3),
    WATERMARK FOR ts AS ts - INTERVAL '5' SECOND
) WITH (
  'connector' = 'kafka',
  'topic' = 'my-replicated-topic',
  'properties.bootstrap.servers' = 'localhost:9092',
  'properties.group.id' = 'FlinkConsumer',
  'scan.startup.mode' = 'earliest-offset',
  'value.format' = 'csv'
);


INSERT INTO kafka_sink
VALUES ('1', 'data1', TIMESTAMP '2024-09-14 10:00:00');
