#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:多继承.py
@time:2020/11/30

"""


class A(object):

    def __init__(self):
        print('A')


class B(object):

    def __init__(self):
        print('B')


class C(A):

    def __init__(self):
        super(C, self).__init__()
        print('C')


class D(B):

    def __init__(self):
        super(D, self).__init__()
        print('D')

    def test(self):
        print(self.a)


class E(C, D):
    def __init__(self):
        self.a = 1
        super(E, self).__init__()


e = E()
e.test()
