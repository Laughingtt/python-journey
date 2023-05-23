#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:processExecutor.py
@time:2020/12/21

"""

# from multiprocessing import Process,Pool
from concurrent.futures import ProcessPoolExecutor
import time, random, os
from functools import partial


def piao(n, name):
    print('%s is piaoing %s' % (name, os.getpid()))
    time.sleep(1)
    return n ** 2


if __name__ == '__main__':
    p = ProcessPoolExecutor(5)
    objs = []
    start = time.time()
    f = partial(piao, name="tian")
    res = p.map(f, [1, 2, 3, 4, 5])
    print(list(res))

    p.shutdown(wait=True)

    stop = time.time()
    print(stop - start)
