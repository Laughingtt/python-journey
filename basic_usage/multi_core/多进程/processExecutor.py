#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:processExecutor.py
@time:2020/12/21

"""

from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import time, os
from functools import partial


def piao(n, name):
    print('%s is piaoing %s' % (name, os.getpid()))
    time.sleep(1)
    return n ** 2


def muiti_fun():
    p = ThreadPoolExecutor(5)
    objs = []
    start = time.time()
    for i in range(5):
        obj = p.submit(piao, i, 'safly %s' % i)  # 异步调用
        objs.append(obj)

    p.shutdown(wait=True)
    print('主', os.getpid())
    for obj in objs:
        print(obj.result())

    stop = time.time()
    print(stop - start)


def muiti_func_map():
    p = ProcessPoolExecutor(5)
    start = time.time()
    f = partial(piao, name="tian")
    res = p.map(f, [1, 2, 3, 4, 5])
    print(list(res))

    p.shutdown(wait=True)

    stop = time.time()
    print(stop - start)


if __name__ == '__main__':
    muiti_fun()
    # muiti_func_map()
