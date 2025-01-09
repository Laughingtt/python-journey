## 官网地址部署：

https://seatunnel.apache.org/docs/2.3.8/start-v2/docker/

## SOURCE

https://seatunnel.apache.org/docs/2.3.8/connector-v2/source/Jdbc

## SINK

https://seatunnel.apache.org/docs/2.3.8/connector-v2/sink/Jdbc

### Docker部署

创建一个network

```docker network create seatunnel-network```

启动节点
启动master节点

## start master and export 5801 port

```shell
sudo docker run -d --name seatunnel_master \
    --network seatunnel-network \
    --rm \
    -p 5801:5801 \
    apache/seatunnel:2.3.8  \
    ./bin/seatunnel-cluster.sh -r master
```

获取容器的ip

```shell
docker inspect seatunnel_master

# 172.21.0.2
```

运行此命令获取master容器的ip

启动worker节点

# 将ST_DOCKER_MEMBER_LIST设置为master容器的ip

```shell
docker run -d --name seatunnel_worker_1 \
    --network seatunnel-network \
    --rm \
    -e ST_DOCKER_MEMBER_LIST=172.21.0.2:5801 \
    apache/seatunnel \
    ./bin/seatunnel-cluster.sh -r worker
```

## 启动第二个worker节点

# 将ST_DOCKER_MEMBER_LIST设置为master容器的ip

```shell
docker run -d --name seatunnel_worker_2 \
    --network seatunnel-network \
    --rm \
     -e ST_DOCKER_MEMBER_LIST=172.21.0.2:5801 \
    apache/seatunnel \
    ./bin/seatunnel-cluster.sh -r worker    
```

集群扩容

# 将ST_DOCKER_MEMBER_LIST设置为已经启动的master容器的ip

```shell
docker run -d --name seatunnel_master \
    --network seatunnel-network \
    --rm \
    -e ST_DOCKER_MEMBER_LIST=172.21.0.2:5801 \
    apache/seatunnel \
    ./bin/seatunnel-cluster.sh -r master
```

运行这个命令创建一个worker节点

# 将ST_DOCKER_MEMBER_LIST设置为master容器的ip

```shell
docker run -d --name seatunnel_worker_1 \
    --network seatunnel-network \
    --rm \
    -e ST_DOCKER_MEMBER_LIST=172.21.0.2:5801 \
    apache/seatunnel \
    ./bin/seatunnel-cluster.sh -r worker
```

# 提交作业到集群

使用docker container作为客户端
提交任务

# 将ST_DOCKER_MEMBER_LIST设置为master容器的ip

```shell
docker run --name seatunnel_client \
    --network seatunnel-network \
    -e ST_DOCKER_MEMBER_LIST=172.16.0.2:5801 \
    --rm \
    apache/seatunnel \
    ./bin/seatunnel.sh  -c config/v2.batch.config.template
```

查看作业列表

# 将ST_DOCKER_MEMBER_LIST设置为master容器的ip

```shell
docker run --name seatunnel_client \
    --network seatunnel-network \
    -e ST_DOCKER_MEMBER_LIST=172.16.0.2:5801 \
    --rm \
    apache/seatunnel \
    ./bin/seatunnel.sh  -l
```

## 容器内提交作业,配置network与集群网络一致

```shell
docker run --name seatunnel_client \
    --network seatunnel_seatunnel_network \
    -e ST_DOCKER_MEMBER_LIST=172.16.0.2:5801 \
    --rm \
    -it \
    -v ./jobs:/opt/seatunnel/jobs \
    apache/seatunnel:2.3.8 \
    bash


./bin/seatunnel.sh  -c config/v2.batch.config.template

```

## binlog权限
以下是权限授予的 SQL 语句：

sql
复制代码
```sql
GRANT REPLICATION CLIENT, REPLICATION SLAVE ON *.* TO 'your_user'@'your_host';
FLUSH PRIVILEGES;
```
权限解释：
REPLICATION CLIENT：允许用户执行 SHOW MASTER STATUS，以查看当前的 binlog 文件名和位置。
REPLICATION SLAVE：允许用户读取 binlog 内容，用于 Flink 等工具进行数据同步。
如果希望更小化权限，可以根据实际需求细化权限，确保用户只访问必要的数据库和资源。例如，如果只需要在特定数据库中读取 binlog，可以替换 *.* 为特定数据库名。

完成权限授予后，确保 MySQL binlog 设置正确（即启用了 binlog），并且使用的 MySQL 账户配置正确。

## 测试样本SQL验证CDC模式

```sql
CREATE TABLE IF NOT EXISTS pk_table
(
    id   INT AUTO_INCREMENT,    -- 自动递增的整数ID
    name VARCHAR(255) NOT NULL, -- 名称字段，最大长度为255个字符，不允许为空
    PRIMARY KEY (id)            -- 设置id为主键
);

CREATE TABLE IF NOT EXISTS pk_table2
(
    pid  INT AUTO_INCREMENT,    -- 自动递增的整数ID
    name VARCHAR(255) NOT NULL, -- 名称字段，最大长度为255个字符，不允许为空
    PRIMARY KEY (pid)           -- 设置id为主键
);

INSERT INTO pk_table (name)
VALUES ('Alice'),
       ('Bob');


INSERT INTO pk_table (name)
VALUES ('Charlie'),
       ('Diana');

INSERT INTO pk_table2 (name)
VALUES ('Alice'),
       ('Bob');


INSERT INTO pk_table2 (name)
VALUES ('Charlie'),
       ('Diana');


-- 向表中添加年龄字段
ALTER TABLE pk_table
    ADD COLUMN age INT;

ALTER TABLE pk_table2
    ADD COLUMN age INT;

-- 插入带有年龄的数据
INSERT INTO pk_table (name, age)
VALUES ('Alice', 30),
       ('Bob', 22),
       ('Charlie', 45),
       ('Diana', 28);

INSERT INTO pk_table2 (name, age)
VALUES ('Alice', 30),
       ('Bob', 22),
       ('Charlie', 45),
       ('Diana', 28);

```

## API REST

curl -i http:127.0.0.1:5801/hazelcast/rest/maps/system-monitoring-information