import multiprocessing as mp
from multiprocessing.dummy import Pool
import time
import pickle
import tempfile


class Sample:
    def __init__(self, obj):
        self.obj = obj

lis = [Sample([j for j in range(3000000)]) for i in range(8)]


def fun(x):
    res = lis
    return res


# tmp = tempfile.NamedTemporaryFile()
# with open(tmp.name, 'wb') as f:
#     f.write(lis)
#
# tmp = tempfile.NamedTemporaryFile()
# with open(tmp.name, 'rb') as f:
#     res = f.read()


### write
# pick_file = open('./temp.txt', 'wb')
# pickle.dump(lis,pick_file)
# pick_file.close()


pool = mp.Pool()


t0 = time.time()
lis = pool.map(fun, [1,2,3,4,5,6,7,8])
print("time is :{}".format(time.time() - t0))

# print(lis)