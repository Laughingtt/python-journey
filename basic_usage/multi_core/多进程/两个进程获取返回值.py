
import random
import time
import multiprocessing


def worker(name, q):
    time.sleep(3)
    q.put(name+"jian")
    print(name)

def main_fun():
    time.sleep(2)
    print("123")

q = multiprocessing.Queue()
p = multiprocessing.Process(target=worker, args=("tian", q))
p.start()


main_fun()
p.join()
print(q.get())

# p.join()
print("456")

