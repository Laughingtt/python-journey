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
      * [5.1 合并结果集](#51-合并结果集)
      * [5.2 连接查询](#52-连接查询)
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

#### 5.1 合并结果集

* UNION：去除重复记录 SELECT * FROM t1 UNION SELECT * FROM t2；
* UNION ALL:不去除重复记录SELECT * FROM t1 UNION ALL SELECT * FROM t2;

#### 5.2 连接查询

1. 内连接：

```sql
-- # 方言版
SELECT e.ename, e.sal, e.comm, d.dname
FROM emp AS e,
     dept AS d
WHERE e.deptno = d.deptno;
-- # 标准版
SELECT *
FROM emp e
         INNER JOIN dept d
                    ON e.deptno = d.deptno;
```

2. 左连接：

```sql
SELECT *
FROM emp e
         LEFT OUTER JOIN dept d
                         ON e.deptno = d.deptno;
```

3. 右连接:

```sql
SELECT *
FROM emp e
         RIGHT OUTER JOIN dept d
                          ON e.deptno = d.deptno;
```

4. 自然连接:

```sql
SELECT *
FROM emp
         NATURAL JOIN dept;
SELECT *
FROM emp
         NATURAL LEFT JOIN dept;
SELECT *
FROM emp
         NATURAL RIGHT JOIN dept;
```

5. 子查询

嵌套查询，即SELECT中包含SELECT，如果一条语句中存在两个，或两个以上SELECT，那么就是子查询语句了。

子查询出现的位置：

* where后，作为条件的一部分
* from后，作为被查询的一条表