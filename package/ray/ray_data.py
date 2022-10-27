# !/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:ray_data.py
@time:2022/10/26

"""

import ray

ds = ray.data.range(10000)

ray.data.read_csv()
ray.data.from_numpy()
