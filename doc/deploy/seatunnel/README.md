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

## 测试样本SQL验证CDC模式

```sql
CREATE TABLE IF NOT EXISTS pk_table
(
    id   INT AUTO_INCREMENT,    -- 自动递增的整数ID
    name VARCHAR(255) NOT NULL, -- 名称字段，最大长度为255个字符，不允许为空
    PRIMARY KEY (id)            -- 设置id为主键
);

INSERT INTO pk_table (name)
VALUES ('Alice');
INSERT INTO pk_table (name)
VALUES ('Bob');
INSERT INTO pk_table (name)
VALUES ('Charlie');
INSERT INTO pk_table (name)
VALUES ('Diana');

-- 向表中添加年龄字段
ALTER TABLE pk_table
    ADD COLUMN age INT;


-- 插入带有年龄的数据
INSERT INTO pk_table (name, age)
VALUES ('Alice', 30);
INSERT INTO pk_table (name, age)
VALUES ('Bob', 22);
INSERT INTO pk_table (name, age)
VALUES ('Charlie', 45);
INSERT INTO pk_table (name, age)
VALUES ('Diana', 28);


```

## API REST

curl -i http:127.0.0.1:5801/hazelcast/rest/maps/system-monitoring-information