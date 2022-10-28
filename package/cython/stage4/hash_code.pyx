# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:hash_key.py
@time:2022/03/01

"""

import pickle as c_pickle
import time

cdef int M = 2 ** 31
cdef int seed = 31

def pickle_kv(k, v):
    return (c_pickle.dumps(k), c_pickle.dumps(v))

def hash_code(s):
    h = len(s)
    for c in s:
        # to singed int
        if c > 127:
            c = -256 + c
        h = h * seed
        if h > 2147483647 or h < -2147483648:
            h = (h & (M - 1)) - (h & M)
        h = h + c
        if h > 2147483647 or h < -2147483648:
            h = (h & (M - 1)) - (h & M)
    if h == 0 or h == -2147483648:
        h = 1
    return h if h >= 0 else abs(h)

def _hash_key_to_partition(key, partitions):
    return hash_code(key) % partitions

def main(n=1000000):
    t0 = time.time()

    for i in range(n):
        k_bytes, v = pickle_kv(i, i)
        p = _hash_key_to_partition(k_bytes, 16)
        # print(p)

    print(time.time() - t0)
