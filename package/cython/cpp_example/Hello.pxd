# 是必须要声明的
from libcpp.vector cimport vector

cdef extern from "src/hello_world.cpp":
    pass

cdef extern from "src/hello_world.h" namespace "test":
    cdef cppclass Hello:
        Hello() except +
        vector[int] primesc(unsigned int nb_primes);