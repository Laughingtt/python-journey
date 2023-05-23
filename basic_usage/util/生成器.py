print("yield:")


def test_yield():
    for i in range(5):
        yield i * i


n = test_yield()

#占用内存小

# 生成器可以用循环调用
for i in n:
    print(i)


# 也可以用next来调用
# print(next(n))
# print(next(n))

# print([next(n) for i in range(5)])
