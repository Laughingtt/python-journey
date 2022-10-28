# distutils: language = c++
### 声明对应的语言

from Rectangle cimport Rectangle
# from Hello cimport Hello
from libcpp.vector cimport vector

def primesc(unsigned int nb_primes):
    cdef int n, i
    cdef vector[int] p
    p.reserve(nb_primes)  # allocate memory for 'nb_primes' elements.

    n = 2
    while p.size() < nb_primes:  # size() for vectors is similar to len()
        for i in p:
            if n % i == 0:
                break
        else:
            p.push_back(n)  # push_back is similar to append()
        n += 1

    # If possible, C values and C++ objects are automatically
    # converted to Python objects at need.
    return p  # so here, the vector will be copied into a Python list.

def main():
    rec_ptr = new Rectangle(1, 2, 3, 4)  # Instantiate a Rectangle object on the heap
    try:
        rec_area = rec_ptr.getArea()
    finally:
        del rec_ptr  # delete heap allocated object

    cdef Rectangle rec_stack  # Instantiate a Rectangle object on the stack
    return rec_area

def primes(nb_primes):
    p = []
    n = 2
    while len(p) < nb_primes:
        # Is n prime?
        for i in p:
            if n % i == 0:
                break

        # If no break occurred in the loop
        else:
            p.append(n)
        n += 1
    return p

cdef list _prime(int nb_primes):
    cdef list p = []
    cdef int n = 2
    while len(p) < nb_primes:
        # Is n prime?
        for i in p:
            if n % i == 0:
                break

        # If no break occurred in the loop
        else:
            p.append(n)
        n += 1
    return p

def primes2(int nb_primes):
    return _prime(nb_primes)



cdef class PyRectangle2:
    ### __cinit__和__dealloc__方法中，它们保证在创建和删除 Python 实例时只调用一次。
    cdef Rectangle*c_rect  # hold a pointer to the C++ instance which we're wrapping

    def __cinit__(self, int x0, int y0, int x1, int y1):
        self.c_rect = new Rectangle(x0, y0, x1, y1)

    def __dealloc__(self):
        del self.c_rect

    def get_area(self):
        return self.c_rect.getArea()

cdef class PyRectangle:
    cdef Rectangle c_rect

    def __cinit__(self, int x0, int y0, int x1, int y1):
        self.c_rect = Rectangle(x0, y0, x1, y1)

    def get_area(self):
        return self.c_rect.getArea()

    def get_size(self):
        cdef int width, height
        self.c_rect.getSize(&width, &height)
        return width, height

    def move(self, dx, dy):
        self.c_rect.move(dx, dy)

    # Attribute access
    @property
    def x0(self):
        return self.c_rect.x0
    @x0.setter
    def x0(self, x0):
        self.c_rect.x0 = x0

    # Attribute access
    @property
    def x1(self):
        return self.c_rect.x1
    @x1.setter
    def x1(self, x1):
        self.c_rect.x1 = x1

    # Attribute access
    @property
    def y0(self):
        return self.c_rect.y0
    @y0.setter
    def y0(self, y0):
        self.c_rect.y0 = y0

    # Attribute access
    @property
    def y1(self):
        return self.c_rect.y1
    @y1.setter
    def y1(self, y1):
        self.c_rect.y1 = y1
