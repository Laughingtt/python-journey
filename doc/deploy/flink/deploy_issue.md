

## 问题一
```shell
ClassNotFoundException: org.apache.flink.shaded.guava30.com.google.common.collect.Lists
```

需要下载flink-shaded-guava 到flink/lib目录下
https://mvnrepository.com/artifact/org.apache.flink/flink-shaded-guava/30.1.1-jre-16.2

注意: 重新增加jar包后,需要重启
```shell
./bin/stop-cluster.sh
./bin/start-cluster.sh
```

## 问题二

默认flink下载版本不支持 kafka,mysql,jdbc等connector需要逐一下载对应版本的jar文件


flink 1.20 版本对应connector

mysql-connector https://repo.maven.apache.org/maven2/mysql/mysql-connector-java/8.0.9-rc/
JDBC https://repo.maven.apache.org/maven2/org/apache/flink/flink-connector-jdbc/3.0.0-1.16/flink-connector-jdbc-3.0.0-1.16.jar
kafka https://repo.maven.apache.org/maven2/org/apache/flink/flink-sql-connector-kafka/1.20.0/flink-sql-connector-kafka-1.20.0.jar