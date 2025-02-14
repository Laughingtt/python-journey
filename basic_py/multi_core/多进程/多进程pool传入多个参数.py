import multiprocessing as mp
import itertools
from functools import partial

class Myargs:
    def __init__(self,index,item):
        self.index = index
        self.item = item

    def get(self):
        return self.item

    def get_index(self):
        return self.index

    def set(self,item):
        self.item = item

    def set_index(self,index):
        self.index = index


def job(index):
    return index


def multicore(z):
    x_y = [Myargs(i,[i,i*2]) for i in range(4)]
    pool = mp.Pool()  # 无参数时，使用所有cpu核
    # pool = mp.Pool(processes=3) # 有参数时，使用CPU核数量为3
    res = pool.map(job, x_y)
    res = [(i.get_index(),i.get()) for i in res]
    res = sorted(res,key=lambda res:res[0],reverse=True)
    return res


if __name__ == '__main__':
    res = multicore(1)
    print(res)
