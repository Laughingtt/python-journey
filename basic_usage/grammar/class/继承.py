class A():
    def __init__(self):
        super().__init__()

    def hello(self):
        pass


class B(A):
    name = "jian"

    def __init__(self, name):
        super().__init__()
        self.name = name

    def hello(self):
        print("我是B的方法")


class C(B):
    def __init__(self, name):
        print(name)
        super().__init__(name)
        print(name)


a = C("tian")
a.hello()
