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
    ext_modules=cythonize("restaurant.pyx")
)

