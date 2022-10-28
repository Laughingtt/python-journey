#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:test.py
@time:2021/06/03

"""
import numpy as np
import time
import lmdb
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


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
    txn.delete(str(1).encode())

    # 提交待处理的事务
    txn.commit()


def query():
    # 数据库查询
    txn = env.begin()  # 每个commit()之后都需要使用begin()方法更新txn得到最新数据库

    print(txn.get(str(2).encode()))

    for key, value in txn.cursor():
        print(str(key, encoding="utf-8"), str(value, encoding="utf-8"))

    env.close()


def insert_data(lis, txn_i):
    env = lmdb.open('data/partition/{}'.format(txn_i), create=True, max_dbs=1, max_readers=1024, lock=True, sync=True,
                    map_size=10_737_418_240)
    txn = env.begin(write=True)

    for i in lis:
        txn.put(pickle.dumps(i), pickle.dumps(i))

    txn.commit()
    return lis[-1]


def submit_pool_insert(n=10000, core=16):
    # pool = ThreadPoolExecutor(4) # 1000w 4core 多线程写数据并不快，速度反而慢 61s
    pool = ProcessPoolExecutor(core)  # 多进程还可以 1000w 4core 8s ; 1e 8core 150s
    data_list = np.split(np.arange(n), core)
    objs = []
    print("start")
    for i in range(core):
        obj = pool.submit(insert_data, data_list[i].tolist(), i)  # 异步调用
        objs.append(obj)

    pool.shutdown(wait=True)
    for obj in objs:
        print(obj.result())


def test_single_insert_speed(n=10000):
    # 1000w 36s
    env = lmdb.open('data/single01', create=True, max_dbs=1, max_readers=1024, lock=True, sync=True,
                    map_size=10_737_418_240)
    txn = env.begin(write=True)
    for i in range(n):
        txn.put(pickle.dumps(i), pickle.dumps(i))
    txn.commit()


if __name__ == '__main__':
    t0 = time.time()
    submit_pool_insert(10000000)
    # test_single_insert_speed(10000000)
    print("test_single_insert_speed time :", time.time() - t0)
