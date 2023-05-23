class SGD:
    pass


def func(name: 'str') -> int:
    return type(name)


func(123)


def demo(name: str, age: 'int > 0' = 20) -> str:  # ->str 表示该函数的返回值是str类型的
    print(name, type(name))
    print(age, type(age))
    return "hello world"


demo(1)  # 这里的参数1会显示黄色, 但是可以运行不会报错
