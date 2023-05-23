class Test(object):
    name = "tina"

    def __init__(self):
        self.name = "tina"

    @staticmethod
    def hello():
        print("hello")

    @staticmethod
    def hi():
        Test.hello()

    @classmethod
    def nihao(cls):
        print(cls.name)


Test.nihao()
