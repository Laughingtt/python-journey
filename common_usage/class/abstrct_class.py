# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:abstrct_class.py
@time:2021/12/31

"""

import abc


class A(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def greet(self):
        pass

    @abc.abstractmethod
    def _greet2(self):
        pass


class B(A):
    def greet(self):
        pass

    def _greet2(self):
        pass


class C(B):
    pass


if __name__ == "__main__":
    b = B()  # 正常实例化
    c = C()  # 解释器抛错

# 输出：
# C类中没有定义greet()方法导致的报错
# Traceback (most recent call last):
#   File "xxx", line xxx, in <module>
#     c = C()
# TypeError: Can't instantiate abstract class C with abstract methods greet
