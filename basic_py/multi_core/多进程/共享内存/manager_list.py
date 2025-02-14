# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:manager_dict.py
@time:2022/03/04

"""
import numpy as np
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor

import multiprocessing
import time
from bitarray import bitarray


def worker(lis, n):
    for i in lis:
        # print(i)
        pass
    return None


if __name__ == '__main__':
    mgr = multiprocessing.Manager()
    lis = mgr.list()
    bit_ = bitarray("1111111111")
    for i in range(10000):
        lis.append(bit_)
    # arr = np.split(np.arange(100000), 8)
    pool = ProcessPoolExecutor(8)

    result = []
    t0 = time.time()
    for i in range(8):
        res = pool.submit(worker, lis, [])
        result.append(res)

    pool.shutdown(wait=True)
    for obj in result:
        print(obj.result())
    print(time.time()-t0)

    print("=====")
