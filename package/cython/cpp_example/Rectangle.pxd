# 是必须要声明的
from libcpp.vector cimport vector
cdef extern from "src/Rectangle.cpp":
    pass


# Declare the class with cdef
cdef extern from "src/Rectangle.h" namespace "shapes":
    cdef cppclass Rectangle:
        Rectangle() except +
        Rectangle(int, int, int, int) except +
        int x0, y0, x1, y1
        int getArea()
        void getSize(int*width, int*height)
        void move(int, int)


# cdef extern from "src/hello_world.cpp":
#     pass
#
# cdef extern from "src/hello_world.h" namespace "shapes":
#     cdef cppclass Hello:
#         Hello() except +
#         vector[int] primesc(unsigned int nb_primes);