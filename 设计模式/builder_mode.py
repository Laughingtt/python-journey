#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:factiory_mode.py
@time:2021/02/26

"""

"""
建造模式：建造者模式更为高层一点，将所有细节都交由子类实现

1. 有一个接口类，定义创建对象的方法。一个指挥员类，接受创造者对象为参数。两个创造者类，创建对象方法相同，内部创建可自定义

2.一个指挥员，两个创造者(瘦子 胖子)，指挥员可以指定由哪个创造者来创造

区别：
工厂模式：由用户来选择生成那个类
建造者模式:已经生成了现成的类，也就是不同的类，dirctor负责指挥，共同的方法执行

工厂模式注重的是整体对象的创建方法，而建造者模式注重的是对象的创建过程，创建对象的过程方法可以在创建时自由调用。


"""



class builder(object):
    def __init__(self):
        pass

    def get_name(self):
        return self.name


class Male(builder):
    def __init__(self, name):
        super().__init__()
        self.name = name
        print("male", name)

    def get_name(self):
        print("male name", self.name)


class Female(builder):
    def __init__(self, name):
        super().__init__()
        self.name = name
        print("female", name)

    def get_name(self):
        print("female name", self.name)


class Director(object):
    def __init__(self, obj):
        self.obj = obj

    def command(self):
        self.obj.get_name()


m = Male("tian")
f = Male("li")
d = Director(m)
d.command()

f = Director(m)
f.command()
