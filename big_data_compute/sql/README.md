# SQL语法

<!-- TOC -->
* [SQL语法](#sql语法)
  * [类型解释](#类型解释)
  * [一、DDL](#一ddl)
  * [二、DML](#二dml)
  * [三、DCL](#三dcl)
  * [四、DQL](#四dql)
    * [4.1条件查询](#41条件查询)
    * [4.2模糊查询](#42模糊查询)
    * [4.3字段控制查询](#43字段控制查询)
    * [4.4 排序](#44-排序)
    * [4.5 聚合函数](#45-聚合函数)
    * [4.6 分组(GROUP BY)查询](#46-分组group-by查询)
    * [4.7 HAVING子句](#47-having子句)
    * [4.8 LIMIT](#48-limit)
    * [4.9 完整性约束](#49-完整性约束)
      * [4.9.1 主键 ：primary key](#491-主键-primary-key)
      * [4.9.2 主键自增长 ：auto_increment（主键必须是整型才可以自增长）](#492-主键自增长-autoincrement主键必须是整型才可以自增长)
      * [4.9.3 非空：NOT NULL](#493-非空not-null)
      * [4.9.4 唯一：UNIQUE](#494-唯一unique)
      * [4.9.5 外键](#495-外键)
    * [5 完整性约束](#5-完整性约束)
      * [5.1 Union](#51-union)
      * [5.2 连接查询](#52-连接查询)
        * [5.2.1. 内连接：](#521-内连接)
        * [5.2.2. 左连接：](#522-左连接)
        * [5.2.3. 右连接:](#523-右连接)
        * [5.2.4. 全连接](#524-全连接)
        * [5.2.5. 子查询](#525-子查询)
    * [6 常用函数](#6-常用函数)
      * [6.1 窗口函数](#61-窗口函数)
      * [6.2 INSERT INTO SELECT](#62-insert-into-select)
      * [6.3 regexp_replace](#63-regexpreplace)
      * [6.4 Partition by 和 group by的区别](#64-partition-by-和-group-by的区别)
      * [6.5 SUM和COUNT是两个不同的聚合函数](#65-sum和count是两个不同的聚合函数)
      * [6.6 TRUNCATE TABLE](#66-truncate-table)
      * [6.7 SUBSTRING](#67-substring)
      * [6.8 CONCAT](#68-concat)
      * [6.9 coalesce 用法](#69-coalesce-用法)
      * [6.10 rank](#610-rank)
<!-- TOC -->
## 类型解释

* DDL（Data Definition Language）：数据定义语言，用来定义数据库对象：库、表、列等；
* DML（Data Manipulation Language）:数据操作语言，用来定义数据库记录（数据）；
* DCL（Data Control Language）：数据控制语言，用来定义访问权限和安全级别；
* DQL（Data Query Language）：数据查询语言，用来查询记录（数据）。

## 一、DDL

```sql
-- 查看所有数据库名称：
SHOW
DATABASES
；
-- 切换数据库：
        USE mydb1;
-- 创建数据库：
        CREATE
DATABASE [IF NOT EXISTS] mydb1
；
-- 删除数据库：
        DROP
DATABASE [IF EXISTS] mydb1
；
-- 修改数据库编码：
       ALTER
DATABASE mydb1 CHARACTER SET utf8
```

```sql
-- 创建表
CREATE TABLE stu
(
    sid    CHAR(6),
    sname  VARCHAR(20),
    age    INT,
    gender VARCHAR(10)
);

CREATE TABLE emp
(
    eid      CHAR(6),
    ename    VARCHAR(50),
    age      INT,
    gender   VARCHAR(6),
    birthday DATE,
    hiredate DATE,
    salary   DECIMAL(7, 2),
    resume   VARCHAR(1000)
);
```

```sql
--查看当前数据库中所有表名称:
SHOW
TABLES;
--查看指定表的创建语句：
    SHOW
CREATE TABLE emp;
--查看表结构:
DESC emp;
--删除表：
DROP TABLE emp;
--修改表：
-- 1.修改之添加列：给stu表添加classname列：
ALTER TABLE stu ADD (classname varchar(100));
-- 2.修改之修改列类型：修改stu表的gender列类型为CHAR(2)：
ALTER TABLE stu MODIFY gender CHAR (2);
-- 3.修改之修改列名：修改stu表的gender列名为sex：
ALTER TABLE stu change gender sex CHAR (2);
-- 4.修改之删除列：删除stu表的classname列：
ALTER TABLE stu DROP classname;
-- 5.修改之修改表名称：修改stu表名称为student：
ALTER TABLE stu RENAME TO student;
```

## 二、DML

```sql
-- 插入数据
INSERT INTO stu(sid, sname)
VALUES ('s_1001', 'zhangSan');
INSERT INTO stu
VALUES ('s_1002', 'liSi', 32, 'female');
-- 修改数据
UPDATE stu
SET sname=’liSi’,
    age=’20’
WHERE age > 50
  AND gender =’male’;
-- 删除数据
DELETE
FROM stu
WHERE sname =’chenQi’
   OR age > 30;
DELETE
FROM stu;
-- truncate 是先DROP TABLE，再CREATE TABLE。而且TRUNCATE删除的记录是无  法回滚的，但DELETE删除的记录是可以回滚的
TRUNCATE TABLE stu;
```

## 三、DCL

```sql
-- 创建用户: CREATE USER 用户名@地址 IDENTIFIED BY '密码';
CREATE
USER user1@localhost IDENTIFIED BY
‘123
’; 
CREATE
USER user2@
’%
’ IDENTIFIED BY
‘123
’; 
-- 给用户授权: GRANT 权限1, … , 权限n ON 数据库.* TO 用户名
GRANT CREATE
,ALTER
,DROP,INSERT,UPDATE,DELETE,SELECT ON mydb1.* TO user1@localhost;
GRANT ALL
ON mydb1.* TO user2@localhost;
-- 撤销授权: REVOKE权限1, … , 权限n ON 数据库.* FORM 用户名
REVOKE CREATE,ALTER,DROP ON mydb1.* FROM user1@localhost;
-- 查看用户权限:SHOW GRANTS FOR 用户名
SHOW
GRANTS FOR user1@localhost;
-- 删除用户:DROP USER 用户名
DROP
USER user1@localhost;
-- 修改用户密码
USE
mysql;
UPDATE USER
SET PASSWORD=PASSWORD(‘密码’)
WHERE User =’用户名’
  and Host =’IP’;
FLUSH
PRIVILEGES;
--------------------
UPDATE USER
SET PASSWORD=PASSWORD('1234')
WHERE User = 'user2'
  and Host =’localhost’;
FLUSH
PRIVILEGES;
```

## 四、DQL

```sql
-- 语法:
SELECT selection_list /*要查询的列名称*/
FROM table_list /*要查询的表名称*/
WHERE condition /*行条件*/
GROUP BY grouping_columns /*对结果分组*/
HAVING condition /*分组后的行条件*/
ORDER BY sorting_columns /*对结果分组*/
    LIMIT offset_start, row_count /*结果限定*/
```

### 4.1条件查询

* =、!=、<>、<、<=、>、>=；
* BETWEEN…AND；
* IN(set)；
* IS NULL；
* AND；
* OR；
* NOT；

```sql
SELECT *
FROM stu
WHERE sid IN ('S_1001', 'S_1002', 'S_1003');
SELECT *
FROM stu
WHERE sname IS NOT NULL;

```

### 4.2模糊查询

* “_”:匹配任意一个字母，5个“\_”表示5个任意字母
* “%”:匹配0~n个任何字母 “

```sql
-- 查询姓名中第2个字母为“i”的学生记录
SELECT *
FROM stu
WHERE sname LIKE '_i%';
```

### 4.3字段控制查询

* 去除重复记录 :distinct

```SELECT DISTINCT sal FROM emp;```

* 给列名添加别名

```SELECT *, sal+IFNULL(comm,0) AS total FROM emp;```

### 4.4 排序

```sql
SELECT *
FROM emp
ORDER BY sal DESC, empno ASC;
```

### 4.5 聚合函数

* COUNT()：统计指定列不为NULL的记录行数；
* MAX()：计算指定列的最大值，是字符串类型，那么使用字符串排序运算；
* MIN()：计算指定列的最小值，是字符串类型，那么使用字符串排序运算；
* SUM()：计算指定列的数值和，不是数值类型，计算结果为0；
* AVG()：计算指定列的平均值，不是数值类型，那么计算结果为0；

### 4.6 分组(GROUP BY)查询

```sql
SELECT deptno, COUNT(*)
FROM emp
WHERE sal > 1500
GROUP BY deptno;
```

### 4.7 HAVING子句

```sql
SELECT deptno, SUM(sal)
FROM emp
GROUP BY deptno
HAVING SUM(sal) > 9000;
-- 注：WHERE是对分组前记录的条件，如果某行记录没有满足WHERE子句的条件，那么这行记录不会参加分组；而HAVING是对分组后数据的约束
```

### 4.8 LIMIT

limit 起始行 , 查询行数 //起始行从0开始，为开区间

```sql
--  查询从第四行开始的10行记录
SELECT *
FROM emp LIMIT 3, 10;
```

### 4.9 完整性约束

#### 4.9.1 主键 ：primary key

创建表：定义列时指定主键

创建表：定义列之后独立指定主键

修改表时指定主键

```ALTER TABLE stu ADD PRIMARY KEY(sid);```

删除主键

```ALTER TABLE stu DROP PRIMARY KEY;```

#### 4.9.2 主键自增长 ：auto_increment（主键必须是整型才可以自增长）

创建表时设置主键自增长

```sql
CREATE TABLE stu
(
    sid    INT PRIMARY KEY AUTO_INCREMENT,
    sname  VARCHAR(20),
    age    INT,
    gender VARCHAR(10)
);
```

修改表时设置主键自增长

```ALTER TABLE stu CHANGE sid sid INT AUTO_INCREMENT;```

修改表时删除主键自增长

```ALTER TABLE stu CHANGE sid sid INT;```

#### 4.9.3 非空：NOT NULL

字段设为非空后，插入记录时必须给值

#### 4.9.4 唯一：UNIQUE

字段指定唯一约束后，字段的值必须是唯一的

#### 4.9.5 外键

外键是另一张表的主键 ！！

外键就是用来约束这一列的值必须是另一张表的主键值！!

```sql

-- # 创建表时设置外键

CREATE TABLE t_section
(
    sid   INT PRIMARY KEY AUTO_INCREMENT,
    sname VARCHAR(30),
    u_id  INT,
    CONSTRAINT fk_t_user FOREIGN KEY (u_id) REFERENCES t_user (uid)
);

-- # 修改表时设置外键

ALTER TABLE t_session
    ADD CONSTRAINT fk_t_user
        FOREIGN KEY (u_id)
            REFERENCES t_user (uid);

-- # 修改表时删除外键

ALTER TABLE t_section
DROP
FOREIGN KEY fk_t_user;
```

### 5 完整性约束

#### 5.1 Union


`UNION` 是用于合并两个或多个SELECT语句的结果集的SQL操作符。它返回所有符合条件的唯一行，去除了重复的行。以下是关于`UNION`操作的用法，示例数据和操作后的数据变化：

**语法**

`UNION`操作符的基本语法如下：

```sql
SELECT column1, column2, ...
FROM table1
WHERE condition1
UNION
SELECT column1, column2, ...
FROM table2
WHERE condition2;
```

* `SELECT` 子句指定要检索的列。
* `FROM` 子句指定要检索数据的表。
* `WHERE` 子句是可选的，用于指定条件以筛选数据。
* `UNION` 操作符用于合并两个或多个SELECT语句的结果。

**示例数据**

假设有两个表，`employees` 和 `contractors`，它们包含员工和合同工的信息。以下是示例数据：

**employees 表**

| employee_id | first_name | last_name | department |
| --- | --- | --- | --- |
| 1 | John | Doe | HR |
| 2 | Jane | Smith | IT |
| 3 | Alice | Johnson | Sales |

**contractors 表**

| contractor_id | first_name | last_name | project |
| --- | --- | --- | --- |
| 101 | Mike | Brown | ProjectA |
| 102 | Sarah | White | ProjectB |
| 103 | Mark | Green | ProjectA |

**操作及数据变化**

假设我们想合并两个表的员工和合同工信息，可以使用`UNION`来执行此操作。以下是示例SQL查询和操作后的数据变化：

```sql
SELECT employee_id, first_name, last_name, department
FROM employees
UNION
SELECT contractor_id, first_name, last_name, project
FROM contractors;
```

执行上述查询后，将返回以下结果：

| employee_id | first_name | last_name | department |
| --- | --- | --- | --- |
| 1 | John | Doe | HR |
| 2 | Jane | Smith | IT |
| 3 | Alice | Johnson | Sales |
| 101 | Mike | Brown | ProjectA |
| 102 | Sarah | White | ProjectB |
| 103 | Mark | Green | ProjectA |

注意，`UNION`操作会合并两个表的结果，并确保不会包含重复的行。在这个例子中，John Doe出现在`employees`表中，Mike Brown和Mark Green分别出现在`contractors`表中，但合并后的结果集中不会有重复的行。

#### 5.2 连接查询


`JOIN` 是用于将多个表的数据连接起来以生成一个包含来自这些表的相关信息的结果集的SQL操作。`JOIN` 操作通常用于关联表中的数据，以便可以检索和显示相关数据。下面是关于 `JOIN` 操作的用法，示例数据和操作后的数据变化：

**语法**

`JOIN` 操作的基本语法如下：

```sql
SELECT columns
FROM table1
JOIN table2 ON table1.column_name = table2.column_name;
```

* `SELECT` 子句指定要检索的列。
* `FROM` 子句指定要检索数据的表，包括要连接的表。
* `JOIN` 关键字用于指定连接操作。
* `table1` 和 `table2` 是要连接的表的名称。
* `ON` 子句用于指定连接条件，即连接两个表的列。

**示例数据**

假设有两个表，`orders` 和 `customers`，它们包含了订单信息和客户信息。以下是示例数据：

**orders 表**

| order_id | customer_id | order_date | total_amount |
| --- | --- | --- | --- |
| 1 | 101 | 2023-01-15 | 100.00 |
| 2 | 102 | 2023-01-16 | 75.50 |
| 3 | 103 | 2023-01-17 | 150.25 |

**customers 表**

| customer_id | first_name | last_name | email |
| --- | --- | --- | --- |
| 101 | John | Doe | [john@example.com](mailto:john@example.com) |
| 102 | Jane | Smith | [jane@example.com](mailto:jane@example.com) |
| 103 | Alice | Johnson | [alice@example.com](mailto:alice@example.com) |

**操作及数据变化**

假设我们想要将 `orders` 表和 `customers` 表连接，以获取订单信息及与之相关的客户信息。我们可以使用 `JOIN` 操作来实现这一目标。以下是示例SQL查询和操作后的数据变化：

```sql
SELECT orders.order_id, orders.order_date, customers.first_name, customers.last_name
FROM orders
JOIN customers ON orders.customer_id = customers.customer_id;
```

执行上述查询后，将返回以下结果：

| order_id | order_date | first_name | last_name |
| --- | --- | --- | --- |
| 1 | 2023-01-15 | John | Doe |
| 2 | 2023-01-16 | Jane | Smith |
| 3 | 2023-01-17 | Alice | Johnson |

这个查询将 `orders` 表和 `customers` 表连接在一起，基于 `customer_id` 列的匹配，将订单信息与相关客户信息一起显示。这是 `JOIN` 操作的一个基本示例，用于将多个表的数据连接在一起以创建更有用的结果集。

##### 5.2.1. 内连接：
（内连接）: INNER JOIN 只返回两个表中在连接条件中匹配的行。如果没有匹配的行，这些行将被忽略。


```sql
SELECT Employees.Name, Departments.DepartmentName
FROM Employees
INNER JOIN Departments ON Employees.DepartmentID = Departments.DepartmentID;

```

##### 5.2.2. 左连接：

（左连接）: LEFT JOIN 返回左表中的所有行，以及右表中与左表匹配的行。如果没有匹配的行，右表中的列将包含 NULL 值。

```sql
SELECT Employees.Name, Departments.DepartmentName
FROM Employees
LEFT JOIN Departments ON Employees.DepartmentID = Departments.DepartmentID;

```

##### 5.2.3. 右连接:

(右连接）: RIGHT JOIN 返回右表中的所有行，以及左表中与右表匹配的行。如果没有匹配的行，左表中的列将包含 NULL 值。

```sql
SELECT Employees.Name, Departments.DepartmentName
FROM Employees
RIGHT JOIN Departments ON Employees.DepartmentID = Departments.DepartmentID;

```

##### 5.2.4. 全连接
（全连接）: FULL JOIN 返回左表和右表中的所有行，如果没有匹配的行，将在相应的一侧使用 NULL 值。
示例 SQL 查询：

```sql
SELECT Employees.Name, Departments.DepartmentName
FROM Employees
FULL JOIN Departments ON Employees.DepartmentID = Departments.DepartmentID;
```

这些不同类型的 JOIN 允许你根据需求从不同的角度合并表，以满足特定查询的需求。根据你的数据和查询目的，你可以选择适当的 JOIN 类型。


##### 5.2.5. 子查询

嵌套查询，即SELECT中包含SELECT，如果一条语句中存在两个，或两个以上SELECT，那么就是子查询语句了。

子查询出现的位置：

* where后，作为条件的一部分
* from后，作为被查询的一条表

子查询（Subquery）是SQL中的一种查询，它嵌套在另一个查询的内部，用于检索或计算中间结果，以供外部查询使用。子查询通常用于过滤、排序、聚合或执行其他操作，以生成更具体或更复杂的查询结果。以下是关于子查询的用法、示例数据和操作后的数据变化：

**语法**

子查询的基本语法如下：

```sql
SELECT columns
FROM table
WHERE column operator (SELECT columns FROM another_table WHERE condition);
```

* `SELECT` 子句指定要检索的列。
* `FROM` 子句指定要检索数据的主表。
* `WHERE` 子句用于指定过滤条件。
* 子查询 `(SELECT columns FROM another_table WHERE condition)` 嵌套在 `WHERE` 子句中，它可以返回一个值、一列值或一个结果集，用于与外部查询的条件进行比较。

**示例数据**

假设有两个表，`students` 和 `scores`，它们包含了学生信息和他们的考试成绩。以下是示例数据：

**students 表**

| student_id | first_name | last_name | age |
| --- | --- | --- | --- |
| 1 | John | Doe | 18 |
| 2 | Jane | Smith | 19 |
| 3 | Alice | Johnson | 20 |

**scores 表**

| student_id | subject | score |
| --- | --- | --- |
| 1 | Math | 90 |
| 2 | Math | 85 |
| 3 | Math | 92 |
| 1 | English | 88 |
| 2 | English | 92 |
| 3 | English | 90 |

**操作及数据变化**

假设我们想找到每个学生的数学成绩的平均值，可以使用子查询来实现这一目标。以下是示例SQL查询和操作后的数据变化：

```sql
SELECT first_name, last_name, (SELECT AVG(score) FROM scores WHERE subject = 'Math') AS math_avg
FROM students;
```

执行上述查询后，将返回以下结果：

| first_name | last_name | math_avg |
| --- | --- | --- |
| John | Doe | 89.00 |
| Jane | Smith | 89.00 |
| Alice | Johnson | 89.00 |

在这个示例中，子查询 `(SELECT AVG(score) FROM scores WHERE subject = 'Math')` 被嵌套在外部查询中，对每个学生计算了数学成绩的平均值。这个平均值随着每个学生的记录一起显示。

子查询可以用于更复杂的操作，例如子查询可以用于WHERE子句中的条件、作为列的一部分、用于连接操作等。子查询是SQL中强大的工具，用于解决各种复杂的查询需求。


### 6 常用函数

#### 6.1 窗口函数

窗口函数（Window Functions）是SQL中的高级分析工具，它允许您在查询结果集内执行聚合和分析操作，同时保留原始行的关联信息。窗口函数通常用于生成有关窗口（也称为窗口帧或窗口规范）内数据的统计信息，例如累积总和、排名、行号和移动平均等。下面更详细地解释窗口函数的用法、示例数据和操作后的数据变化。

 **语法**

窗口函数的基本语法如下：

```sql
window_function(expression) OVER (
    [PARTITION BY partition_expression, ...]
    [ORDER BY sort_expression [ASC|DESC], ...]
    [frame_specification]
);
```

* `window_function` 是要使用的窗口函数，如 `SUM`、`RANK`、`AVG`、`ROW_NUMBER` 等。
* `expression` 是窗口函数的参数，它指定要进行聚合或分析的列。
* `PARTITION BY` 子句用于将数据集划分为不同的分区，以便窗口函数在每个分区内执行。
* `ORDER BY` 子句用于定义窗口内数据的排序顺序。它可以用于控制排名和累积操作。
* `frame_specification` 是可选的，用于定义窗口的范围，以确定窗口函数如何处理数据。它包括以下部分：
    * `ROWS BETWEEN` 或 `RANGE BETWEEN`：指定窗口范围的起始和结束点。
    * `UNBOUNDED PRECEDING`：表示窗口范围的起始点是无限远的。
    * `CURRENT ROW`：表示窗口范围的结束点是当前行。

**示例数据**

假设有一个表，`sales`，它包含了不同日期和销售额的数据。以下是示例数据：

**sales 表**

| sale_id | sale_date | amount |
| --- | --- | --- |
| 1 | 2023-01-01 | 100.00 |
| 2 | 2023-01-02 | 150.00 |
| 3 | 2023-01-03 | 200.00 |
| 4 | 2023-01-04 | 125.00 |
| 5 | 2023-01-05 | 180.00 |

**操作及数据变化**

**累积总和（Cumulative Sum）**

假设我们想要计算每日销售额的累积总和，可以使用窗口函数来实现。以下是示例SQL查询和操作后的数据变化：

```sql
SELECT sale_date, amount, SUM(amount) OVER (ORDER BY sale_date) AS cumulative_total
FROM sales;
```

执行上述查询后，将返回以下结果：

| sale_date | amount | cumulative_total |
| --- | --- | --- |
| 2023-01-01 | 100.00 | 100.00 |
| 2023-01-02 | 150.00 | 250.00 |
| 2023-01-03 | 200.00 | 450.00 |
| 2023-01-04 | 125.00 | 575.00 |
| 2023-01-05 | 180.00 | 755.00 |

这个查询使用窗口函数 `SUM(amount) OVER (ORDER BY sale_date)` 计算了每日销售额的累积总和，并将结果添加到每行中。`ORDER BY sale_date` 确保数据按照销售日期的顺序进行排序，以便正确计算累积总和。

 **排名（Ranking）**

假设我们想要为销售额最高的日期分配排名，可以使用窗口函数来实现。以下是示例SQL查询和操作后的数据变化：

```sql
SELECT sale_date, amount, RANK() OVER (ORDER BY amount DESC) AS sales_rank
FROM sales;
```

执行上述查询后，将返回以下结果：

| sale_date | amount | sales_rank |
| --- | --- | --- |
| 2023-01-05 | 180.00 | 1 |
| 2023-01-03 | 200.00 | 2 |
| 2023-01-04 | 125.00 | 3 |
| 2023-01-02 | 150.00 | 4 |
| 2023-01-01 | 100.00 | 5 |

这个查询使用窗口函数 `RANK() OVER (ORDER BY amount DESC)` 为销售额最高的日期分配了排名。`ORDER BY amount DESC` 指定按销售额降序排序，以便排名最高的日期排在最前面。

这只是窗口函数的两个示例。窗口函数还可以用于计算行号、计算分组内的平均值、查找前N行、查找行与前一行的差值等各种分析操作。它们是强大的工具，用于处理复杂的分析需求。

#### 6.2 INSERT INTO SELECT

`INSERT INTO SELECT` 是SQL语句，用于将查询的结果集插入到目标表中。这是一种非常有用的方法，可以将一个表中的数据复制到另一个表中，或者从查询中选择的数据插入到新的表中。以下是关于 `INSERT INTO SELECT` 的用法，示例数据和操作后的数据变化：

**语法**

`INSERT INTO SELECT` 语句的基本语法如下：

```sql
INSERT INTO target_table (column1, column2, ...)
SELECT column1, column2, ...
FROM source_table
WHERE condition;
```

* `INSERT INTO` 子句指定目标表和要插入的列。
* `target_table` 是要将数据插入的目标表的名称。
* `column1, column2, ...` 是目标表中要插入数据的列。
* `SELECT` 子句用于指定要从源表选择的列和条件。
* `source_table` 是要选择数据的源表的名称。
* `WHERE` 子句是可选的，用于指定要选择的数据行的条件。

**示例数据**

假设有两个表，`source_table` 和 `target_table`，它们包含了员工信息。以下是示例数据：

**source_table 表**

| employee_id | first_name | last_name | department |
| --- | --- | --- | --- |
| 1 | John | Doe | HR |
| 2 | Jane | Smith | IT |
| 3 | Alice | Johnson | Sales |

**target_table 表**（空表）

| employee_id | first_name | last_name | department |
| --- | --- | --- | --- |

**操作及数据变化**

假设我们想将 `source_table` 中的员工信息插入到 `target_table` 中。可以使用 `INSERT INTO SELECT` 语句来完成这个任务。以下是示例SQL查询和操作后的数据变化：

```sql
INSERT INTO target_table (employee_id, first_name, last_name, department)
SELECT employee_id, first_name, last_name, department
FROM source_table;
```

执行上述查询后，将把 `source_table` 中的员工信息插入到 `target_table` 中，操作后的 `target_table` 数据如下：

**target_table 表**

| employee_id | first_name | last_name | department |
| --- | --- | --- | --- |
| 1 | John | Doe | HR |
| 2 | Jane | Smith | IT |
| 3 | Alice | Johnson | Sales |

这样，我们成功将 `source_table` 中的数据复制到 `target_table` 中。

请注意，您可以根据实际需求选择要插入的列，以及可以在 `SELECT` 子句中使用条件来过滤要插入的数据。`INSERT INTO SELECT` 是在SQL中非常有用的操作，可以将数据从一个表复制到另一个表，或者从一个查询中选择并插入数据。


#### 6.3 regexp_replace

cast(
regexp_replace(
tpep_pickup_datetime,
'(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2}):(\d{2}) (.*)',
'$3-$1-$2 $4:$5:$6'
) as TIMESTAMP
)
根据提供的示例，这是一个在SQL中使用CAST和正则表达式进行数据类型转换的示例。具体来说，它将日期时间字符串转换为TIMESTAMP类型。
下面是对提供的表达式进行分解的解释：
1.regexp_replace
函数使用正则表达式将日期时间字符串进行匹配和替换。具体来说，它将"tpep_pickup_datetime"列中的日期时间字符串进行替换。
正则表达式：'(\d{2})/(\d{2})/(\d{4}) (\d{2}):(\d{2}):(\d{2}) (.*)'
- '(\d{2})'：匹配两位数字作为月份。
- '/'：匹配斜杠分隔符。
- '(\d{2})'：匹配两位数字作为日期。
- '/'：匹配斜杠分隔符。
- '(\d{4})'：匹配四位数字作为年份。
- ' '：匹配空格分隔符。
- '(\d{2})'：匹配两位数字作为小时。
- ':'：匹配冒号分隔符。
- '(\d{2})'：匹配两位数字作为分钟。
- ':'：匹配冒号分隔符。
- '(\d{2})'：匹配两位数字作为秒钟。
- ' '：匹配空格分隔符。
- '(.*)'：匹配剩余的任意字符。
替换字符串：'$3-$1-$2 $4:$5:$6'
- '$3'：替换为正则表达式中匹配的第三个子表达式，即年份。
- '-'：插入短横线作为日期分隔符。
- '$1'：替换为正则表达式中匹配的第一个子表达式，即月份。
- '-'：插入短横线作为日期分隔符。
- '$2'：替换为正则表达式中匹配的第二个子表达式，即日期。
- ' '：插入空格作为日期和时间之间的分隔符。
- '$4'：替换为正则表达式中匹配的第四个子表达式，即小时。
- ':'：插入冒号作为时间分隔符。
- '$5'：替换为正则表达式中匹配的第五个子表达式，即分钟。
- ':'：插入冒号作为时间分隔符。
- '$6'：替换为正则表达式中匹配的第六个子表达式，即秒钟。
2.CAST
函数将替换后的日期时间字符串转换为TIMESTAMP数据类型。

#### 6.4 Partition by 和 group by的区别

PARTITION BY和GROUP BY是SQL中用于对数据进行分组的两种不同方式。
1.GROUP BY:
GROUP BY子句是用于将数据按照指定的列或表达式进行分组，并对每个分组进行聚合计算。GROUP BY通常与聚合函数（如SUM、AVG、MAX、MIN）一起使用，以计算每个分组的聚合结果。它将相同值的行归为一组，并对每个组应用聚合函数。
例如：

```sql
SELECT department, SUM(sales) AS total_sales
FROM sales_table
GROUP BY department;
```

在上述示例中，数据按照部门进行分组，并计算每个部门的销售总额。
2.PARTITION BY:
PARTITION BY子句是用于定义窗口函数（Window Function）的分区方式。它在窗口函数中指定了如何将数据划分为多个窗口，以便对每个窗口进行独立的计算。PARTITION BY与窗口函数一起使用，允许在不使用GROUP BY的情况下对数据进行分组和聚合。
例如：

```sql
SELECT department, sales, SUM(sales) OVER (PARTITION BY department) AS total_sales
FROM sales_table;
```

在上述示例中，数据按照部门进行分区，并使用窗口函数SUM计算每个部门的销售总额，而不需要使用GROUP BY子句。
区别总结：
- GROUP BY用于对数据进行分组，并对每个组应用聚合函数，返回每个分组的聚合结果。
- PARTITION BY用于定义窗口函数的分区方式，将数据划分为多个窗口进行独立的计算，返回每个窗口的计算结果。
需要注意的是，GROUP BY在SELECT语句中会生成一个结果集，每个分组对应一个结果行，而PARTITION BY与窗口函数一起在每个输入行上计算，并返回结果作为新的列。

#### 6.5 SUM和COUNT是两个不同的聚合函数

用于计算数据集的汇总信息。
1.SUM函数用于计算指定列的总和。它接受一个列作为参数，并返回该列中所有值的总和。
示例：

```sql
SELECT SUM(sales_amount) AS total_sales
FROM sales_table;
```

上述示例中，SUM函数被用于计算sales_table中sales_amount列的总和。
1.COUNT函数用于计算指定列的行数。它接受一个列作为参数，并返回该列中非NULL值的行数。
示例：
```sql
SELECT COUNT(customer_id) AS total_customers
FROM customers_table;
```
上述示例中，COUNT函数被用于计算customers_table中customer_id列的行数。
所以，SUM函数用于计算数值列的总和，而COUNT函数用于计算行数。它们在功能和用途上有明显的区别。


#### 6.6 TRUNCATE TABLE

"TRUNCATE TABLE Prices" 是一条 SQL 语句，用于从名为 "Prices" 的表中删除所有数据，同时保持表的结构不变。与 "DELETE FROM Prices" 语句相比，它是一种更快的方式来清空表中的所有行。
当执行 "TRUNCATE TABLE Prices" 语句时，"Prices" 表中的所有行将被永久删除，表将变为空。表的列、索引和约束将保持不变。
需要注意的是，"TRUNCATE TABLE" 语句无法回滚。一旦数据被截断，就无法恢复，因此在使用时应谨慎。此外，根据使用的数据库系统，执行 "TRUNCATE TABLE" 语句可能需要适当的权限或许可。


#### 6.7 SUBSTRING
函数是SQL中用于提取子字符串的函数，它允许你从一个字符串中选择指定位置和长度的子字符串。
在大多数数据库系统中，SUBSTRING函数的语法如下：
scssCopy codeSUBSTRING(string, start, length)

参数说明：
- string：要提取子字符串的原始字符串。
- start：子字符串的起始位置，从1开始计数。
- length：要提取的子字符串的长度。
示例用法：
```sql
SELECT SUBSTRING('Hello, World!', 1, 5); -- 输出 'Hello'
SELECT SUBSTRING('Hello, World!', 8); -- 输出 'World!'
```

#### 6.8 CONCAT

CONCAT函数是SQL中用于连接字符串的函数，它可以将多个字符串拼接在一起。
在大多数数据库系统中，CONCAT函数的语法如下：
scssCopy codeCONCAT(string1, string2, ...)

参数说明：
- string1, string2, ...：要连接的字符串参数，可以是一个或多个。
示例用法：
```sql
SELECT CONCAT('Hello', ', ', 'World!'); -- 输出 'Hello, World!'
SELECT CONCAT('The', ' ', 'quick', ' ', 'brown', ' ', 'fox'); -- 输

```

#### 6.9 coalesce 用法

函数 COALESCE 在 SQL 中用于返回参数列表中的第一个非 NULL 值。它接受多个参数，并按照参数顺序进行判断，返回第一个非 NULL 值。
语法如下：
```sql
COALESCE(value1, value2, ...)
```
使用 COALESCE 函数时，它将按照参数的顺序依次判断每个值，返回第一个非 NULL 值。如果所有参数都为 NULL，则 COALESCE 函数返回 NULL。
下面是一些 COALESCE 函数的示例用法：
示例 1：
```sql
SELECT COALESCE(null, 'Value 1', 'Value 2');
```
结果为：'Value 1'在上述示例中，COALESCE 函数从左到右依次判断参数，遇到第一个非 NULL 值 'Value 1'，并将其作为结果返回。
示例 2：
```sql
SELECT COALESCE(null, null, null, 'Final Value');
```
结果为：'Final Value'在这个例子中，所有参数都为 NULL，但是 COALESCE 函数返回了最后一个非 NULL 值 'Final Value'。
COALESCE 函数对于处理可能为 NULL 的列或表达式非常有用。它允许你在 SQL 查询中处理 NULL 值并提供替代值，以确保查询结果的准确性和一致性。


#### 6.10 rank

在SQL中，有几个与排名（ranking）和窗口函数相关的函数，可以帮助您对查询结果进行排名和分析。以下是一些常见的与排名相关的函数：

1. **RANK()**：`RANK()` 函数用于为查询结果中的行分配排名，通常与窗口函数一起使用。它分配相同排名给相等的值，并在下一个排名跳过相同排名的行。例如：
    
    ```sql
    SELECT employee_id, first_name, last_name, RANK() OVER (ORDER BY salary DESC) AS salary_rank
    FROM employees;
    ```
    
2. **DENSE_RANK()**：`DENSE_RANK()` 函数类似于 `RANK()`，但不跳过相同排名的行，而是为它们分配相同的排名。例如：
    
    ```sql
    SELECT employee_id, first_name, last_name, DENSE_RANK() OVER (ORDER BY salary DESC) AS salary_dense_rank
    FROM employees;
    ```
    
3. **NTILE()**：`NTILE(n)` 函数将结果集分成大约相等的 n 个部分，并为每个部分的行分配一个数字，从1到 n。这对于分析数据的分位数很有用。例如：
    
    ```sql
    SELECT employee_id, first_name, last_name, NTILE(4) OVER (ORDER BY salary DESC) AS salary_quartile
    FROM employees;
    ```
    
4. **ROW_NUMBER()**：`ROW_NUMBER()` 函数为结果集中的每一行分配一个唯一的整数值，不考虑相同的值。它不会跳过相同排名的行。例如：
    
    ```sql
    SELECT employee_id, first_name, last_name, ROW_NUMBER() OVER (ORDER BY hire_date) AS row_num
    FROM employees;
    ```
    