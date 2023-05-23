def func(*args):
    print(args)


def func2(*args, **kwargs):
    print(args)
    print(kwargs)


def fun3(a, b, c, d, r, t, y):
    print(a, b, c, d)
    print(r, t, y)


def fun4(r, t, y):
    print(r, t, y)


def fun5(*args, **kwargs):
    print(args)
    print(kwargs)


if __name__ == '__main__':
    func(1, 2, 3, 4)
    func2(1, 2, 3, a=1, b=3)
    a = [1, 2, 3, 4, 5, 6, 7]
    fun3(*a)

    """
    *X 将数组打散分配
    **X 将字典打散分配
    函数内可以接收任意参数值,*X转化为元祖，**X转化为字典
    """

    b = {"r": 3, "t": 4, "y": 5}
    fun4(**b)

    fun5(1, 2, 3, 4, a=1, b=2)
