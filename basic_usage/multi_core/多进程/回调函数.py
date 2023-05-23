# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:回调函数.py
@time:2022/06/02

"""
import multiprocessing


# 设置回调函数
def setcallback(x):
    print("write",x)
    with open('result.txt', 'a+') as f:
        line = str(x) + "\n"
        f.write(line)


def multiplication(num):
    print(num)
    return num * (num + 1)


if __name__ == '__main__':
    pool = multiprocessing.Pool(6)
    for i in range(1000):
        pool.apply_async(func=multiplication, args=(i,), callback=setcallback)
    pool.close()
    pool.join()
