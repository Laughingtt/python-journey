# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:setup.py
@time:2021/10/15

"""

from setuptools import setup
from Cython.Build import cythonize

setup(
    name="hello",
    version="1.0.0",
    ext_modules=cythonize("hello.pyx"),
    zip_safe=False
)

