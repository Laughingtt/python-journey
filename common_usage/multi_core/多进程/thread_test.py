# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:thread_test.py
@time:2021/11/19

"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import time


# 定义一个准备作为线程任务的函数
def action(max, age):
    print("age is {}".format(age))
    my_sum = 0
    for i in range(max):
        # print(threading.current_thread().name + '  ' + str(i))
        my_sum += i
    return my_sum


# 创建一个包含2条线程的线程池
pool = ThreadPoolExecutor(max_workers=2)
# 向线程池提交一个task, 50会作为action()函数的参数
future1 = pool.submit(action, 50, 20)
# 向线程池再提交一个task, 100会作为action()函数的参数
future2 = pool.submit(action, 100, 30)
# 判断future1代表的任务是否结束
print(future1.done())
# time.sleep(3)
# 判断future2代表的任务是否结束
print(future2.done())
# 查看future1代表的任务返回的结果
print(future1.result())
# 查看future2代表的任务返回的结果
print(future2.result())
# 关闭线程池
pool.shutdown()
