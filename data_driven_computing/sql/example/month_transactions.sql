

Create table If Not Exists Transactions (id int, country varchar(4), state enum('approved', 'declined'), amount int, trans_date date)
Truncate table Transactions
insert into Transactions (id, country, state, amount, trans_date) values ('121', 'US', 'approved', '1000', '2018-12-18')
insert into Transactions (id, country, state, amount, trans_date) values ('122', 'US', 'declined', '2000', '2018-12-19')
insert into Transactions (id, country, state, amount, trans_date) values ('123', 'US', 'approved', '2000', '2019-01-01')
insert into Transactions (id, country, state, amount, trans_date) values ('124', 'DE', 'approved', '2000', '2019-01-07')


--Table: Transactions
--
--+---------------+---------+
--| Column Name   | Type    |
--+---------------+---------+
--| id            | int     |
--| country       | varchar |
--| state         | enum    |
--| amount        | int     |
--| trans_date    | date    |
--+---------------+---------+
--id 是这个表的主键。
--该表包含有关传入事务的信息。
--state 列类型为 “[”批准“，”拒绝“] 之一。
--
--
--编写一个 sql 查询来查找每个月和每个国家/地区的事务数及其总金额、已批准的事务数及其总金额。
--
--以 任意顺序 返回结果表。
--
--查询结果格式如下所示。
--
--
--
--示例 1:
--
--输入：
--Transactions table:
--+------+---------+----------+--------+------------+
--| id   | country | state    | amount | trans_date |
--+------+---------+----------+--------+------------+
--| 121  | US      | approved | 1000   | 2018-12-18 |
--| 122  | US      | declined | 2000   | 2018-12-19 |
--| 123  | US      | approved | 2000   | 2019-01-01 |
--| 124  | DE      | approved | 2000   | 2019-01-07 |
--+------+---------+----------+--------+------------+
--输出：
--+----------+---------+-------------+----------------+--------------------+-----------------------+
--| month    | country | trans_count | approved_count | trans_total_amount | approved_total_amount |
--+----------+---------+-------------+----------------+--------------------+-----------------------+
--| 2018-12  | US      | 2           | 1              | 3000               | 1000                  |
--| 2019-01  | US      | 1           | 1              | 2000               | 2000                  |
--| 2019-01  | DE      | 1           | 1              | 2000               | 2000                  |
--+----------+---------+-------------+----------------+--------------------+-----------------------+

# Write your MySQL query statement below

select
    DATE_FORMAT(trans_date, '%Y-%m') as month,
    # data_format 对日期进行格式化
    country,
    count(id) as trans_count,
    # 统计按月份，国家分组后的交易数量
    sum(if (state = 'approved',1,0)) as approved_count,
    # 统计approved 的交易数量，首先使用if 语句将，state字段置为 1和0，对此结果进行求和即可得到approved交易的数量
    sum(amount) as trans_total_amount,
    # 统计交易总量
    sum(if (state = 'approved',1,0) *amount) as approved_total_amount
    # 先找出approved 的交易 乘以 对应的 交易金额 得到了 approved 交易金额，再求和得到总金额
    # id,
    # state,
    # amount,
    # trans_date
from
    Transactions
group by month(trans_date), country