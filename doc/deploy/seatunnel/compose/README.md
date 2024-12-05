

## 启动master和 work服务
[docker-compose.yaml](docker-compose.yaml)


### 新建容器子网络

networks:
  seatunnel_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.16.0.0/24


## 额外增加work节点服务
[docker-compose-add-work.yaml](docker-compose-add-work.yaml)


### 加入外部网络
networks:
  sealtune_seatunnel_network:
    external: true
