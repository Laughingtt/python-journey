<!-- TOC -->
  * [挂载磁盘](#挂载磁盘)
    * [磁盘确认挂载](#磁盘确认挂载)
    * [格式化分区vdb1](#格式化分区vdb1-)
    * [创建一个挂载点](#创建一个挂载点)
    * [挂载](#挂载)
  * [启动挂载](#启动挂载)
<!-- TOC -->

这篇文章主要介绍了linux如何永久挂载新硬盘和/etc/fstab配置文件的说明,磁盘格式化，磁盘管理、挂载新硬盘（linux运维基础）。

首先列出文件系统的整体磁盘空间使用情况。可以用来查看磁盘已被使用多少空间和还剩余多少空间。
## 挂载磁盘

### 磁盘确认挂载
```shell
df -h
```
![img.png](img%2Fimg.png)


```shell
lsblk
```
列出所有可用块设备的信息，显示他们之间的依赖关系

![img_1.png](img%2Fimg_1.png)

有一块 3T 的磁盘 vdb，我们现在将它进行磁盘分区


```shell
fdisk /dev/vdb
```

![img_2.png](img%2Fimg_2.png)

输入 m 获取帮助，p 查看分区表

![img_3.png](img%2Fimg_3.png)

当前分区里面没有任何信息，继续操作，输入 n 创建一个新的分区

![img_4.png](img%2Fimg_4.png)

选择默认 p 选择主分区  e 扩展分区 直接默认回车就是选择 p

![img_5.png](img%2Fimg_5.png)

输入分区号，默认从1开始，默认回车

![img_6.png](img%2Fimg_6.png)

sector 起始扇区 (2048-4294967295, 默认 2048)：默认回车

![img_7.png](img%2Fimg_7.png)

+ 多少扇区 或多大空间，不会计算的话 可以 写 +1G 或者 选择默认回车

![img_8.png](img%2Fimg_8.png)

最后输入w 保存

![img_9.png](img%2Fimg_9.png)

查看，新建的区分已显示出来

![img_10.png](img%2Fimg_10.png)


### 格式化分区vdb1 

```mkfs.ext4 /dev/vdb1```

![img_11.png](img%2Fimg_11.png)

### 创建一个挂载点

```shell
mkdir /vdb1

```

### 挂载

```shell
mount /dev/vdb1 /vdb1

```

![img_12.png](img%2Fimg_12.png)

## 启动挂载

首先查看UUID

```shell
blkid
```
![img_13.png](img%2Fimg_13.png)

熟练的话可以直接将文件目录写到挂载的配置文件中

将 /dev/vdb1 的 UUID 复制出来，然后写入到/etc/fstab中去

```shell
echo "UUID=e943fbb7-020a-4c64-a48a-2597eb2496df /vdb1 ext4 defaults 0 0" >> /etc/fstab
```
或者 编辑 /etc/fstab 配置文件 挂载（推荐）

vim /etc/fstab

![img_14.png](img%2Fimg_14.png)