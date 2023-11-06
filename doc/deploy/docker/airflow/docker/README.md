
<!-- TOC -->
  * [部署](#部署)
    * [1.映射目录权限](#1映射目录权限)
    * [2. 启动容器](#2-启动容器)
    * [3. 访问页面](#3-访问页面)
  * [镜像打包](#镜像打包)
    * [1. 打包镜像](#1-打包镜像)
    * [2. 导出镜像](#2-导出镜像)
    * [3. 导入镜像](#3-导入镜像)
<!-- TOC -->

## 部署

### 1.映射目录权限

映射的目录宿主机权限设置为最高，否则容器内写入权限不够

```shell
chmod 777 logs
chmod 777 dags
```

### 2. 启动容器
```shell
docker-compose up -d 
```

### 3. 访问页面

https://airflow.sdgft.com/home

![img.png](attach%2Fimg.png)


## 镜像打包

### 1. 打包镜像

```shell
sudo docker build -f Dockerfile -t sdataft/airflow:v1 --no-cache .
```

### 2. 导出镜像

```shell
docker save -o airflow.tar sdataft/airflow:v1
```

### 3. 导入镜像

```shell
docker load -i airflow.tar
```