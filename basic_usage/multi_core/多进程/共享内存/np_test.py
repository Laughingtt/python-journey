# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:np_test.py
@time:2022/07/18

"""

import numpy as np
from multiprocessing import shared_memory

a = np.array([1, 1, 2, 3, 5, 8])
shm = shared_memory.SharedMemory(create=True, size=a.nbytes)

b = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
b[:] = a[:]
print(b)

# ==============new process
import numpy as np
from multiprocessing import shared_memory

existing_shm = shared_memory.SharedMemory(name=shm.name)
c = np.ndarray((6,), dtype=np.int64, buffer=existing_shm.buf)
print(c)
c[2] = 888
print(b)
