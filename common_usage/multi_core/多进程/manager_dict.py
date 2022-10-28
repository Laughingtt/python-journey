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


def worker(dic, lis):
    for i in lis:
        dic[i] = i
    return 1


if __name__ == '__main__':
    mgr = multiprocessing.Manager()
    dic = mgr.dict()
    arr = np.split(np.arange(100000), 8)
    pool = ProcessPoolExecutor(8)

    result = []
    for i in range(8):
        res = pool.submit(worker, dic, arr[i].tolist())
        result.append(res)

    pool.shutdown(wait=True)
    for obj in result:
        print(obj.result())

    print("=====")