#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:processExecutor.py
@time:2020/12/21

"""
import numpy as np
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor
import time, os
from functools import partial


def piao(n, name, shape, dtype):
    existing_shm = shared_memory.SharedMemory(name=name)
    c = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    print('%s is piaoing %s,point %s' % (name, os.getpid(), (n, c[n][n])))
    existing_shm.close()
    time.sleep(1)
    return n ** 2


def muiti_fun():
    a = np.random.random((500000, 10))
    print(a)
    print("size is {}".format(a.nbytes/1024/1024))
    shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
    shm_np_array = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
    np.copyto(shm_np_array, a)
    p = ProcessPoolExecutor(5)
    objs = []
    start = time.time()
    for i in range(5):
        obj = p.submit(piao, i, shm.name, shm_np_array.shape, shm_np_array.dtype)  # 异步调用
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
    print(123)


if __name__ == '__main__':
    muiti_fun()
    # muiti_func_map()
