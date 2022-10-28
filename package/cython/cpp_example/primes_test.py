# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:primes_test.py
@time:2021/11/10

"""

import time


def primes(nb_primes):
    p = []
    n = 2
    while len(p) < nb_primes:
        # Is n prime?
        for i in p:
            if n % i == 0:
                break

        # If no break occurred in the loop
        else:
            p.append(n)
        n += 1
    return p


if __name__ == '__main__':
    t0 = time.time()
    print(len(primes(10000)))
    print(time.time() - t0)
