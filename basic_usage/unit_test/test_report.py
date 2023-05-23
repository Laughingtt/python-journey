#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
@company:UDAI
@author:tianjian
@file:test_report.py
@time:2021/03/12

"""
import unittest  # 单元测试模块
from BeautifulReport import BeautifulReport as bf  # 导入BeautifulReport模块，这个模块也是生成报告的模块，但是比HTMLTestRunner模板好看


class TestCalc(unittest.TestCase):
    def setUp(self):  # 每个用例运行之前运行的
        print('setup是啥时候运行的')

    def tearDown(self):  # 每个用例运行之后运行的
        print('teardown是啥时候运行的')

    @classmethod
    def setUpClass(cls):  # 在所有用例执行之前运行的
        print('我是setUpclass，我位于所有用例的开始')

    @classmethod
    def tearDownClass(cls):  # 在所有用例都执行完之后运行的
        print('我是tearDownClass，我位于多有用例运行的结束')

    def testcc(self):  # 函数名要以test开头，否则不会被执行
        '''这是第一个测试用例'''  # 用例描述，在函数下，用三个单引号里面写用例描述
        self.assertEqual(1, 1)
        print('第一个用例')

    def testaa(self):
        '''这个是第二个测试用例'''
        self.assertEqual(1, 1)
        print('第二个用例')

    def testdd(self):
        '''用例描述3'''
        print('第三个用例')

    def testbb(self):
        '''用例描述4'''
        print('第四个用例')


suite = unittest.TestSuite()  # 定义一个测试集合
suite.addTest(unittest.makeSuite(TestCalc))  # 把写的用例加进来（将TestCalc类）加进来
run = bf(suite)  # 实例化BeautifulReport模块
run.report(filename='test', description='这个描述参数是必填的')
