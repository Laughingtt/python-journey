# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:joblib_test.py
@time:2021/11/10

"""

from joblib import Parallel, delayed
import time
from math import sqrt


def single(a):
    """ 定义一个简单的函数  """
    time.sleep(1)  # 休眠1s
    print(a)


start = time.time()  # 记录开始的时间
res = Parallel(n_jobs=6,backend="multiprocessing")(delayed(single)(i) for i in range(10))  # 并行化处理
Time = time.time() - start  # 计算执行的时间
print(res)
print(Time)
# res = Parallel(n_jobs=1)(delayed(sqrt)(i ** 2) for i in range(10))
# print(res)
