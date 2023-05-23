import os
import unittest

from prettytable import PrettyTable


ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
TASK_LOG_PATH = os.getcwd()


class Sub1(unittest.TestCase):
    table = PrettyTable(["type", "algo_name", "status"])

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        """
        初始化测试类
        """

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        print("============= result =================\n\n ", cls.table)

    def test_0(self):
        self.table.add_row(["classification", 0, "success"])
        self.assertEqual(0, 0)

    def test_1(self):
        self.table.add_row(["classification", 1, "success"])
        self.assertEqual(1, 1)



if __name__ == '__main__':
    unittest.main()
