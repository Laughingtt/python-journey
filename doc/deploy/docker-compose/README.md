# Docker Compose

Docker Compose 是一个用于定义和运行多个 Docker 容器的工具，通过一个单一的配置文件来管理它们的关系和配置。以下是一些常用的 Docker Compose 命令以及一个示例的 `docker-compose.yml` 配置文件。

### 常用 Docker Compose 命令：

1. **启动服务：**
    
    ```
    docker-compose up
    ```
    
    这会根据配置文件启动所有服务。
    
2. **启动服务并在后台运行：**
    
    ```
    docker-compose up -d
    ```
    
    这会在后台启动服务。
    
3. **停止服务：**
    
    ```
    docker-compose down
    ```
    
    这会停止并删除所有相关的容器。
    
4. **查看服务状态：**
    
    ```
    docker-compose ps
    ```
    
    这会显示正在运行的服务的状态。
    
5. **查看服务日志：**
    
    ```
    docker-compose logs
    ```
    
    这会显示所有服务的日志输出。
    
6. **构建服务：**
    
    ```
    docker-compose build
    ```
    
    这会重新构建服务的镜像。
    
7. **执行命令在服务中：**
    
    ```bash
    docker-compose exec <service-name> <command>
    ```
    
    用于在指定的服务容器中执行命令。
    

### 示例 `docker-compose.yml` 配置文件：


```yaml
version: '3'  # 指定 Compose 文件版本，一般使用 '3' 或 '3.7'

services:  # 定义各个容器服务
  webapp:  # 服务名称
    image: nginx:latest  # 使用的镜像及版本
    container_name: my-web-app  # 指定容器名称
    ports:  # 端口映射
      - "80:80"  # 将主机的端口映射到容器的端口
    volumes:  # 挂载卷
      - ./web-content:/usr/share/nginx/html  # 挂载本地目录到容器内
    environment:  # 环境变量
      - NGINX_PORT=80
    networks:  # 网络连接
      - my-network
    depends_on:  # 依赖关系
      - database

  database:
    image: mysql:5.7
    container_name: my-db
    environment:
      MYSQL_ROOT_PASSWORD: examplepassword
      MYSQL_DATABASE: mydb
    volumes:
      - db-data:/var/lib/mysql
    networks:
      - my-network

volumes:  # 定义卷
  db-data:

networks:  # 定义网络
  my-network:
    driver: bridge  # 指定网络驱动，通常使用 bridge

```

以下是配置项的解释：

* `version`: Compose 文件的版本，不同版本支持不同的功能和语法。
    
* `services`: 定义要运行的容器服务的部分。
    
* `webapp` 和 `database`: 服务名称，您可以自定义。
    
* `image`: 使用的 Docker 镜像及其版本。
    
* `container_name`: 定义容器的名称。
    
* `ports`: 将主机端口映射到容器端口，"主机端口:容器端口"。
    
* `volumes`: 挂载卷，将主机目录或数据卷挂载到容器内。
    
* `environment`: 设置容器的环境变量。
    
* `networks`: 指定容器连接到的网络。
    
* `depends_on`: 定义容器之间的依赖关系。在示例中，`webapp` 依赖于 `database`，即 `database` 会在 `webapp` 启动之前启动。
    
* `volumes`: 定义数据卷，可以被一个或多个容器使用。
    
* `networks`: 定义网络，用于容器间通信。在示例中，使用了默认的 bridge 网络。
    

这只是一个示例，Docker Compose 具有更多配置选项，可以根据需要添加更多容器服务，并自定义各种配置项，以满足您的应用程序的需求。