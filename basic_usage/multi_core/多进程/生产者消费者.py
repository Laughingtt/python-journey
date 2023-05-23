# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:生产者消费者.py
@time:2022/02/24

"""
import time
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


class FitiObj:
    def __init__(self):
        self._queue = Queue(maxsize=100000)

    def put(self, item):
        self._queue.put(item)

    def get(self):
        return self._queue.get()

    def get_queue(self):
        return self._queue


def create_data(fit: FitiObj):
    for i in range(20000):
        print("加入队列的数据", i)
        fit.put([i])
        time.sleep(0.5)


def get_data(fit: FitiObj):
    while fit.get_queue().empty() is False:
        print("拿到数据:", fit.get())
        time.sleep(1)


fiti = FitiObj()
_executor_pool = ThreadPoolExecutor(max_workers=8)
r1 = _executor_pool.submit(create_data, fiti)
time.sleep(1)
r2 = _executor_pool.submit(get_data, fiti)

# r1.result()
# r2.result()
# print("====")
