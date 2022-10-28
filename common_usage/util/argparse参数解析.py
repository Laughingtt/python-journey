#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:argparse参数解析.py
@time:2021/03/01

"""

import argparse

parser = argparse.ArgumentParser(description="down model and data")

parser.add_argument("-j", "-job_id", required=False, type=str, help="runtime job id")
args = parser.parse_args()
print(args)
