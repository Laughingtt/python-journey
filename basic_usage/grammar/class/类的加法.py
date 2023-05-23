class A:
    def __init__(self, age):
        self.age = age

    def __add__(self, other):
        return A(self.age + other.age)

    def __mul__(self, other):
        return A(self.age * other.age)

    def __repr__(self):
        return "repr function {}".format(self.age)

    def __str__(self):
        return "str function {}".format(self.age)


a = A(10)
b = A(12)
c = a * b
print(c)
