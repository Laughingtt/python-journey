from multiprocessing import Pool
# from multiprocessing.dummy import Pool
import time
from functools import wraps
import inspect

def log_elapsed(func):
    func_name = func.__name__

    @wraps(func)
    def _fn(*args, **kwargs):
        t = time.time()
        name = f"{func_name}#{kwargs['func_tag']}" if 'func_tag' in kwargs else func_name
        rtn = func(*args, **kwargs)
        frame = inspect.getouterframes(inspect.currentframe(), 2)
        print(f"{frame[1].filename.split('/')[-1]}:{frame[1].lineno} call {name}, takes {time.time() - t}s")
        return rtn
    return _fn

@log_elapsed
def func(i): #返回值只有进程池才有,父子进程没有返回值
    time.sleep(0.5)
    return i*i

@log_elapsed
def func2(x):
    return x+2



if __name__ == '__main__':
    p = Pool(5)
    ret = p.map(func,[1,2,3,4,5])

    print(ret)

    ret2 = p.map(func2,[1,2,3,4,5])
    print(ret2)

