#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:test.py
@time:2021/06/03

"""

import lmdb

env = lmdb.open('lmdb', map_size=10 * 1024 ** 2,lock=False)


def change():
    # 创建一个事务Transaction对象
    txn = env.begin(write=True)

    # insert/modify
    # txn.put(key, value)
    txn.put(str(1).encode(), "Alice".encode())  # .encode()编码为字节bytes格式
    txn.put(str(2).encode(), "Bob".encode())
    txn.put(str(3).encode(), "Jack".encode())

    # delete
    # txn.delete(key)
    # txn.delete(str(1).encode())

    # 提交待处理的事务
    txn.commit()
    # txn.put(str(4).encode(), "Jack".encode())
    # txn.commit()


def query():
    # 数据库查询
    txn = env.begin()  # 每个commit()之后都需要使用begin()方法更新txn得到最新数据库

    print(txn.get(str(2).encode()))

    for key, value in txn.cursor():
        print(str(key, encoding="utf-8"), str(value, encoding="utf-8"))

    env.close()


change()
query()
