# distutils: language = c++
### 声明对应的语言

from Hello cimport Hello

cdef class PyHello:
    ### __cinit__和__dealloc__方法中，它们保证在创建和删除 Python 实例时只调用一次。
    cdef Hello*c_rect  # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self):
        self.c_rect = new Hello()

    def __dealloc__(self):
        del self.c_rect

    def get_n(self, n):
        return self.c_rect.primesc(n)
