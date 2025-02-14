import functools


def add(x, y):
    print(x, y)
    return x + y


f = functools.partial(add, 10)
##设定除了x之外的参数，相当于对现有的函数进行一个包装相当于新函数为f(x) =等价于 add(x,y=10)

print(f(15))
