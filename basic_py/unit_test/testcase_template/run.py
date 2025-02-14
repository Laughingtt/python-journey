import os
import os.path
import unittest

ROOT_PATH = os.path.abspath(os.path.dirname(__file__))
TASK_LOG_PATH = os.getcwd()

ml_path = os.path.join(ROOT_PATH, 'sub1')
dl_path = os.path.join(ROOT_PATH, 'sub2')

test_path = [ml_path, dl_path]


def get_all_case():
    suite = unittest.TestSuite()
    for _path in test_path:
        discover = unittest.defaultTestLoader.discover(_path, pattern="*test.py", top_level_dir=ROOT_PATH)
        suite.addTest(discover)
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(get_all_case())
