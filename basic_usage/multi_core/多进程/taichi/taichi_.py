# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:taichi_.py
@time:2022/08/02

"""
import time
import numpy as np

import taichi as ti

ti.init()

a = ti.Matrix.ndarray(n=2, m=3, dtype=ti.uint8, shape=(2, 2))
b = np.ones(3, dtype=np.uint8)


@ti.func
def is_prime(n: int):
    result = True
    for k in range(2, int(n ** 0.5) + 1):
        if n % k == 0:
            result = False
            break
    return result


@ti.kernel
def count_primes(n: ti.types.ndarray()) -> int:
    count = 0
    for k in n:
        if is_prime(k):
            count += 1

    return count


# def is_prime(n: int):
#     result = True
#     for k in range(2, int(n ** 0.5) + 1):
#         if n % k == 0:
#             result = False
#             break
#     return result
#
#
# def count_primes(n: int) -> int:
#     count = 0
#     for k in range(2, n):
#         if is_prime(k):
#             count += 1
#
#     return count

class T:
    def test(self):
        t0 = time.time()
        lis = np.array([i for i in range(1000000)])
        print(count_primes(lis))
        print(time.time() - t0)

    def test2(self):
        D_matrix = np.zeros((9, 10))
        D_matrix2 = np.random.random(size=(1000, 30))
        print(D_matrix)
        print(D_matrix2)
        print_np(D_matrix, D_matrix2)
        print(D_matrix)


@ti.kernel
def print_np(n: ti.types.ndarray(), m: ti.types.ndarray()):
    print(n.shape)
    print(m.shape)
    print(n[0, 0])
    print(m[0, 0])
    # for i in range(9):
    #     for j in range(1):
    #         n[i, j] = m[0,0]
    # m[i, j] = 1


if __name__ == '__main__':
    t = T()

    t.test2()
