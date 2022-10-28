# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:Queue_ex.py
@time:2022/06/01

"""

from multiprocessing import Manager, Pool
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def write(lis, q):
    # 将列表中的元素写入队列中
    for i in lis:
        print('开始写入值%s' % i)
        q.put(i)
        time.sleep(1)


# 读取
def read(q):
    print('开始读取')
    while not q.empty():
        print('读取到:', q.get())
        time.sleep(1)


if __name__ == '__main__':
    # 创建队列
    q = Manager().Queue()
    pool = ProcessPoolExecutor(max_workers=3)
    objs = []
    for i in [["a", "b", "c"], ["a", "e", "c"], ["a", "g", "s"]]:
        sub = pool.submit(write, i, q)
        objs.append(sub)

    for i in objs:
        i.result()

    pool2 = ProcessPoolExecutor(max_workers=2)
    res2 = pool2.map(read, [q for i in range(2)])
    print(list(res2))

    # # 创建写入进程
    # pw = Process(target=write, args=(q,))
    # pr = Process(target=read, args=(q,))
    # # 启动进程
    # pw.start()
    # pw.join()
    # pr.start()
    # pr.join()
