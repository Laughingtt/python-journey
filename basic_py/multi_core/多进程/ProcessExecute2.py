#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:ProcessExecute2.py
@time:2020/12/21

"""
from Crypto.Cipher import AES
from concurrent.futures import ProcessPoolExecutor
import time, random, os
import numpy as np
from bitarray import bitarray

def bitarr_to_byte(bit_str):
    b = bitarray(bit_str)
    a = b.tobytes()
    return a

PRF_key = np.random.randint(1, size=128, dtype=np.uint8)
PRF_key_bytes = bitarr_to_byte(PRF_key.tolist())
cipher = AES.new(PRF_key_bytes, AES.MODE_CTR)

def piao(name, n):
    print('%s is piaoing %s' % (name, os.getpid()))
    time.sleep(1)
    print(cipher.encrypt("123".encode("utf-8")))
    return n ** 2


if __name__ == '__main__':
    p = ProcessPoolExecutor(5)
    start = time.time()
    for i in range(5):
        res=p.submit(piao,'safly %s' %i,i).result() #同步调用
        print(res)

    p.shutdown(wait=True)
    print('主', os.getpid())

    stop = time.time()
    print(stop - start)