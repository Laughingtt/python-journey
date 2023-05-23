# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:_psi_preprocess.py
@time:2021/12/31

"""
import abc
from abc import ABCMeta


class ABC:
    def __init__(self):
        self.names = "123"

    @property
    def name(self):
        return self.names

    @name.setter
    def name(self, names):
        self.names = names

    @name.deleter
    def name(self):
        del self.names


a = ABC()
print(a.name)
a.name = "tian"
del a.name
# print(a.name)
