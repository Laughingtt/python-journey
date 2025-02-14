# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:test_m.py
@time:2021/11/19

"""
import os
import gc
import psutil
import time


# gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_LEAK)


def print_mem():
    proc = psutil.Process(os.getpid())
    mem0 = proc.memory_info().rss
    print("current is {}".format(mem0 / 1024 / 1024))


def main():
    print_mem()
    lis = [i for i in range(1000000)]
    lis.append(lis)

    print_mem()
    del lis
    print_mem()

    # print(gc.collect())
    print_mem()

    time.sleep(1)
    print_mem()


def test01():
    a = []
    b = []
    a.append(b)
    b.append(a)
    del a
    print(gc.collect())
    del b
    print(gc.collect())


def test02():
    print(gc.garbage)
    print_mem()
    lis = [i for i in range(10)]
    lis.append(lis)
    print(gc.garbage)
    del lis
    main()
    print(gc.collect())
    print_mem()
    print(gc.garbage)


if __name__ == '__main__':
    test02()
