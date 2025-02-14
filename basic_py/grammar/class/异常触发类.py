#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:异常触发类.py
@time:2020/12/07

"""


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# Define a class to raise Line errors
class LineError(Exception):  # 继承自基类Exception
    def __init__(self, ErrorInfo):
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo


class Line:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

        if point1.x == point2.x and point1.y == point2.y:
            raise LineError("Cannot create line")


line = Line(Point(1, 2), Point(1, 2))