"""
1.根据下方条件写出SQL
有一张浏览表user_visit_log，有三个字段user_id，page_id，visit_timestamp
计算每个用户访问每个页面的时间：
1. 默认同一个用户不同页面访问记录之间的时间为访问时间
2. 如果是最后一条记录，则按15分钟算
3. 如果访问时间超过15分钟，则算15分钟
"""

SELECT user_id, page_id, visit_timestamp,
       IF(
           (
               SELECT COUNT(*) FROM user_visit_log AS uv2
               WHERE uv2.user_id = uv1.user_id AND uv2.page_id = uv1.page_id AND uv2.visit_timestamp > uv1.visit_timestamp
           ) = 0,
           15,
           IF(
               TIMESTAMPDIFF(MINUTE, visit_timestamp, LEAD(visit_timestamp) OVER (PARTITION BY user_id, page_id ORDER BY visit_timestamp)) >= 15,
               15,
               TIMESTAMPDIFF(MINUTE, visit_timestamp, LEAD(visit_timestamp) OVER (PARTITION BY user_id, page_id ORDER BY visit_timestamp))
           )
       ) AS visit_time
FROM user_visit_log AS uv1;



SELECT user_id, page_id, visit_timestamp,
       IF(
           TIMESTAMPDIFF(MINUTE, visit_timestamp, LEAD(visit_timestamp) OVER (PARTITION BY user_id, page_id ORDER BY visit_timestamp)) >= 15,
           15,
           TIMESTAMPDIFF(MINUTE, visit_timestamp, LEAD(visit_timestamp) OVER (PARTITION BY user_id, page_id ORDER BY visit_timestamp))
       ) AS visit_time
FROM user_visit_log;































1.

select use_id, para_id,times
    if (timeStamdiff(times,Minute,lead(times) over(partiton by user_id,pages order by times)) >= 15,
        15,
        timeStamdiff(times,Minute,lead(times) over(partiton by user_id,pages order by times)) >= 15,)
from  time_log


2. 数据库执行慢原因

    1. 数据库有没有加索引，缺少索引速度比较慢
    2. 锁定的问题，多个用户访问一张表，操作同一数据会有锁的情况
    3。 数据量比较大
    4。 sql 优化问题，语句写的不好


3。 分表分库后id 怎么处理

    1。 首先分库分表后，首先要保证的是id全局统一
    2。 uuid,时间，自增id都可以
    3。 分布式id生成器去生成


4。 雪花模型和星模型


主要区别在维度表

    1。 中心事实表与其他部分表，通过一个字段的表的维度关联起来
    2。 查询速度比较快，易于理解，表结构不好维护

    1。 维度可以进一步分化成多个维度表
    2。 表结构复杂，
    3。 查询慢，好维护


5。 缓慢变换维的处理方式
    1。 直接替换
    2。 保留历史
    3。 保留部分历史数据


6。
    reduce
    reducebykey
    groupbykey
    aggrefate
    等
    注意groupbykey 会把相同的key拉到统一分区，导致负载过重

7。
    1。 shuflle优化
    2。 数据预处理
    3. 重分区，
    4。增加并行读
    5。桶排序等
    6。 对key加随机

