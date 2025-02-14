class Test():
    name = "tianjian"

    def __enter__(self):
        print("enter")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit")


with open() as p:
    pass

with Test() as t:
    print(t.name)
