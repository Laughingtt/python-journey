from __future__ import print_function

# 在导入c静态代码时需要用到
cimport cython

# cimport 导入必须包含pxd文件才行，pxd文件中只需要函数或类的声明
# 只有cdef 类型的函数需要声明
from pri cimport prime
from pri import prime2


cdef class Shrubbery:
    cdef int width
    cdef int height

    def __init__(self, w, h):
        self.width = w
        self.height = h

    def describe(self):
        print("This shrubbery is", self.width,
              "by", self.height, "cubits.")

def main(n):
    print(prime(n))
    print(prime2(n))