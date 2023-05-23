import unittest
import paramunittest
import time


@paramunittest.parametrized(
    {"user": "admin", "psw": "123", "result": "true"},
    {"user": "admin1", "psw": "1234", "result": "true"},
)
class TestDemo(unittest.TestCase):
    def setParameters(self, user, psw, result):
        '''这里注意了，user, psw, result三个参数和前面定义的字典一一对应'''
        self.user = user
        self.user = psw
        self.result = result

    def testcase(self):
        print("开始执行用例：--------------")
        time.sleep(0.5)
        print("输入用户名：%s" % self.user)
        print("输入密码：%s" % self.user)
        print("期望结果：%s " % self.result)
        time.sleep(0.5)
        self.assertTrue(self.result == "true")


if __name__ == "__main__":
    unittest.main(verbosity=2)
