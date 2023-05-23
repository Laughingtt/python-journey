#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:processExecutor.py
@time:2020/12/21

"""
import pickle
import sys
import numpy as np
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
import time, os
from functools import partial
from bitarray import bitarray


def piao(n, name, shape, dtype):
    existing_shm = shared_memory.SharedMemory(name=name)
    c = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    print('%s is piaoing %s,point %s' % (name, os.getpid(), (n, c[n][n])))
    existing_shm.close()
    time.sleep(1)
    return n ** 2


def muiti_fun():
    a = [bitarray("1010101010010101010101") for i in range(5)]
    print(a)
    size = sys.getsizeof(a) * 2
    print("size is {}".format(size))
    a_bytes = pickle.dumps(a)
    shm = shared_memory.SharedMemory(create=True, size=size)
    buff = shm.buf
    buff[:len(a_bytes)] = a_bytes
    p = ProcessPoolExecutor(5)
    objs = []
    start = time.time()
    for i in range(5):
        obj = p.submit(piao, i, shm.name)  # 异步调用
        objs.append(obj)

    p.shutdown(wait=True)
    print('主', os.getpid())
    for obj in objs:
        print(obj.result())

    stop = time.time()
    print(stop - start)
    time.sleep(1)
    shm.close()
    shm.unlink()


if __name__ == '__main__':
    muiti_fun()
    # muiti_func_map()
