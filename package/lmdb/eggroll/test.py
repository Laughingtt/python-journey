# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:test.py
@time:2022/02/24

"""
import time
from broker import FifoBroker
from pair import BatchBroker
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


def single_test():
    broker = FifoBroker()
    bb = BatchBroker(broker, 2)

    bb.put(123)
    bb.put(456)
    bb.put(132)
    bb.signal_write_finish()
    bb.get()


def partition_test(total_partitions=8):
    partitioned_brokers = [FifoBroker() for i in range(total_partitions)]
    partitioned_bb = [BatchBroker(v) for v in partitioned_brokers]
    import random

    def add_data():
        for i in range(1000):
            k = random.choice(range(7))
            partitioned_bb[k].put((i, "p_" + str(k)))
            # print("add :", i)

    print("====")

    def get_data(batch_ibj: BatchBroker):
        for k, v in batch_ibj:
            print(k, v, "\n")
        return 1

    thread_pool = ThreadPoolExecutor(max_workers=8)
    thread_pool.submit(add_data)
    feature = [thread_pool.submit(get_data, i) for i in partitioned_bb]
    res = [i.result() for i in feature]


t0 = time.time()
partition_test()
print("time is :", time.time() - t0)
