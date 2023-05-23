import pickle

dataDic = {0: [1, 2, 3, 4],
           1: ('a', 'b'),
           2: {'c': 'yes', 'd': 'no'}}


class Test():
    age = 18
    sex = "男"

    def __init__(self, name):
        self.name = name


# 可以将obj对象进行序列化存储，也可以将字典，list等等序列化存储,dumps,loads是转为字符串处理
# t = Test("tian")
#
# fw = open('./data_pickle.txt',"wb")
#
# pickle.dump(t,fw)
#
# fw.close()


fr = open("./data_pickle.txt", "rb")
data = pickle.load(fr)
print(data, type(data))
print(data.age, data.sex)
fr.close()
