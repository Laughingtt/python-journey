
cdef class Rectangle:
    cdef int x0, y0
    cdef int x1, y1

    def __init__(self, int x0, int y0, int x1, int y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    def area(self):
        area = (self.x1 - self.x0) * (self.y1 - self.y0)
        if area < 0:
            area = -area
        return area

def rectArea(x0, y0, x1, y1):
    rect = Rectangle(x0, y0, x1, y1)
    return rect.area()


cdef class Rectangle2:
    cdef int x0, y0
    cdef int x1, y1

    def __init__(self, int x0, int y0, int x1, int y1):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    cdef int _area(self):
        cdef int area = (self.x1 - self.x0) * (self.y1 - self.y0)
        if area < 0:
            area = -area
        return area

    def area(self):
        return self._area()

def rectArea2(x0, y0, x1, y1):
    cdef Rectangle2 rect = Rectangle2(x0, y0, x1, y1)
    return rect._area()


cdef class Count(object):

    def sum_count(self,int c_count):
        return self.sum_count_(c_count)

    cdef int sum_count_(self,int c_count):

        cdef int s
        for i in range(c_count):
            if i % 3 == 1:
                s += i / (i + 1)
            else:
                s += 0

        return s


print("hello world")