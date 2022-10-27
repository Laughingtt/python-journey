# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:first_run.py
@time:2022/10/18

"""
import ray
import time

ray.init()


# Define the square task.
@ray.remote
def square(x):
    c = 0
    for i in range(100000000):
        c = c + i
    return c


# Launch four parallel square tasks.
t0 = time.time()
futures = [square.remote(i) for i in range(4)]

# Retrieve results.
print(ray.get(futures))
print(time.time() - t0)
