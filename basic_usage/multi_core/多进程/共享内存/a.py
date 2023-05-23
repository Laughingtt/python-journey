# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:a.py
@time:2022/07/15
https://docs.python.org/zh-cn/3/library/multiprocessing.shared_memory.html
"""
from multiprocessing import shared_memory
from bitarray import bitarray

a = bitarray("1010101")
shm_a = shared_memory.ShareableList(['张三', 2, 'abc'], name='123')

shm_b = shared_memory.ShareableList(name='123')


print(shm_b[0])  # ‘张三’
print(shm_b[1])  # 2
print(shm_b[2])  # ‘abc

