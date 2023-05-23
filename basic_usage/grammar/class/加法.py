class A:
    def __add__(self, other):
        print("A add")
        return A()

    def __radd__(self, other):
        print("A addr")


class B:
    pass


a = A()
b = B()

"""
a+b 调用add
b+a 调用aadr
"""
print(a + b)
