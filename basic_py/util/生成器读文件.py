#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:生成器读文件.py
@time:2021/04/19

"""

from mmap import mmap


def get_lines(f_p):
    with open(f_p, "r+") as f:
        m = mmap(f.fileno(), 0)
        tmp = 0
        for i, char in enumerate(m):
            if char == b"\n":
                yield m[tmp:i + 1].decode()
                tmp = i + 1


if __name__ == "__main__":
    for i in get_lines("生成器.py"):
        print(i)
