a = None
if a is not None:
    print(1)
else:
    print(2)

import os

# 当前文件绝对路径
print(os.path.dirname(os.path.abspath(__file__)))
