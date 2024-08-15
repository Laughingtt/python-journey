
## 一、简介


Databend 是开源的列式数据库系统，它们以高效的数据存储和查询性能而闻名。

1. 架构与设计理念

Databend:

云原生设计：Databend 是一个完全云原生的数据库，设计之初就考虑到了在云端的弹性扩展能力和资源利用效率。它基于对象存储和分布式计算的架构，适合在云环境下动态扩展。
Serverless：Databend 采用无服务器架构，用户不需要管理底层的服务器，系统可以根据负载自动扩展计算资源和存储资源。
存储计算分离：Databend 实现了计算与存储的完全分离，这意味着计算节点可以无缝地添加或删除，而不会影响存储节点的状态
2. 性能与优化

Databend:

云端优化：Databend 在云环境下表现出色，特别是在处理大规模数据分析任务时，它能够利用云资源的弹性来优化查询性能。
向量化执行引擎：Databend 的查询引擎经过向量化优化，能够高效处理批量数据，并提供更快的查询响应时间。
自动优化：Databend 采用了自动调优技术，可以自动调整查询计划和数据分布，减少用户的管理负担。
3. 应用场景

Databend:

云端大数据分析：Databend 非常适合在云环境下的大规模数据分析任务，尤其是在需要弹性计算资源和存储资源的场景下。
企业级云原生应用：Databend 的无服务器架构和自动扩展能力，使其成为构建企业级云原生数据应用的理想选择。
4. 优势总结

Databend:

云原生和无服务器架构，支持存储计算分离，弹性扩展能力强。
自动优化和调优，用户管理负担小。
特别适合云端大规模数据分析和企业级应用。



## minio

```shell
mkdir -p ${HOME}/projects/minio/data

docker run -d \
   --name minio \
   --user $(id -u):$(id -g) \
   --net=host \
   -e "MINIO_ROOT_USER=ROOTUSER" \
   -e "MINIO_ROOT_PASSWORD=CHANGEME123" \
   -v ${HOME}/projects/minio/data:/data \
   minio/minio server /data --console-address ":9001"
```


## docker 部署 Databend
https://docs.databend.cn/guides/deploy/deploy/non-production/deploying-local

```shell
docker run -d \
    --name databend \
    --net=host \
    -v meta_storage_dir:/var/lib/databend/meta \
    -v log_dir:/var/log/databend \
    -e QUERY_DEFAULT_USER=databend \
    -e QUERY_DEFAULT_PASSWORD=databend \
    -e QUERY_STORAGE_TYPE=s3 \
    -e AWS_S3_ENDPOINT=http://172.31.15.63:9000 \
    -e AWS_S3_BUCKET=databend \
    -e AWS_ACCESS_KEY_ID=ROOTUSER \
    -e AWS_SECRET_ACCESS_KEY=CHANGEME123 \
    datafuselabs/databend
```


### docker registry
```shell
{
  "registry-mirrors": [
        "https://docker.mirrors.ustc.edu.cn",
        "https://registry.docker-cn.com",
        "https://docker.m.daocloud.io",
        "https://dockerproxy.com",
        "https://docker.nju.edu.cn"
  ]
}

sudo systemctl daemon-reload
sudo systemctl restart docker
```



## 手动部署databend 

https://docs.databend.cn/guides/deploy/deploy/non-production/deploying-databend
```shell
# meta
sudo ./databend-meta -c ../configs/databend-meta.toml > meta.log 2>&1 &
curl -I  http://127.0.0.1:28101/v1/health



# query
sudo ./databend-query -c ../configs/databend-query.toml > query.log 2>&1 &
curl -I  http://127.0.0.1:8080/v1/health

```

## BendSQL

安装bendsql
https://docs.databend.cn/guides/sql-clients/bendsql/#installing-bendsql

使用bendsql
https://docs.databend.cn/tutorials/