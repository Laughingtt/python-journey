# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:test02.py
@time:2021/12/28

"""

# -*- coding: utf-8 -*-
# python=3.6
import lmdb


def lmdb_create():
    # 如果train文件夹下没有data.mbd或lock.mdb文件，则会生成一个空的，如果有，不会覆盖
    # map_size定义最大储存容量，单位是kb，以下定义1TB容量
    env = lmdb.open("./train", map_size=1000)
    env.close()


def lmdb_using():
    env = lmdb.open("./train", map_size=int(1e9))

    # 参数write设置为True才可以写入
    txn = env.begin(write=True)

    # 添加数据和键值
    txn.put(key='1'.encode(), value='aaa'.encode())
    txn.put(key='2'.encode(), value='bbb'.encode())
    txn.put(key='3'.encode(), value='ccc'.encode())

    # 通过键值删除数据
    txn.delete(key='1'.encode())

    # 修改数据
    txn.put(key='3'.encode(), value='ddd'.encode())

    # 通过commit()函数提交更改
    txn.commit()
    env.close()


def lmdb_read():
    env = lmdb.Environment('./train')
    # env = lmdb.open("./train")   # or

    txn = env.begin()  # write=False

    # get函数通过键值查询数据
    print(txn.get('2'.encode()))

    # 通过cursor()遍历所有数据和键值
    for key, value in txn.cursor():
        print(key, value)

    print(txn.stat())
    print(txn.stat()['entries'])  # 读取LMDB文件的样本数量

    # close
    env.close()


def main():
    # lmdb_create()
    lmdb_using()
    lmdb_read()


# errors:
# 1. lmdb.MapFullError: mdb_put: MDB_MAP_FULL: Environment mapsize limit reached
# 解决方法： lmdb.open("./train", map_size=int(1e9)

# 2. TypeError: Won't implicitly convert Unicode to bytes; use .encode()
# 解决方法： TypeError:不会隐式地将Unicode转换为字节,对字符串部分，进行.encode()

if __name__ == '__main__':
    main()
