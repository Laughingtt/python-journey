# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:setup.py
@time:2021/11/09

"""

from distutils.core import Extension, setup
from Cython.Build import cythonize

# ext = [Extension("hello.sub", ["sub.pyx"]),
#        Extension("hello.pri", ["pri.pyx"])]

ext = [Extension("hello.sub", ["sub.pyx","pri.pyx"])]

setup(ext_modules=cythonize(ext, language_level=3))
