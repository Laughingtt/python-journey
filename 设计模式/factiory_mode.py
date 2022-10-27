#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:factiory_mode.py
@time:2021/02/26

"""

"""
工厂模式用于生成不同类型的有着相似功能的类，factory的作用在于生成类

"""


class People(object):
    def __init__(self):
        pass

    def get_name(self):
        return self.name


class Male(People):
    def __init__(self, name):
        self.name = name
        print("male", name)


class Female(People):
    def __init__(self, name):
        self.name = name
        print("female", name)


class Factory(object):
    def make_factory(self, sex, name):
        if sex == "male":
            return Male(name)
        else:
            return Female(name)


f = Factory()
n = f.make_factory("male", "tian")
print(n.get_name())
