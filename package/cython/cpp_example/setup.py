# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:setup.py
@time:2021/11/09

"""

from setuptools import setup

from Cython.Build import cythonize

setup(ext_modules=cythonize("hello.pyx"))
