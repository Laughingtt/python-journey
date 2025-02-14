class Foo(object):
    def __init__(self, val=2):
        self.val = val

    def __getstate__(self):
        print("I'm being pickled :{}".format(self.__dict__))
        self.val *= 2
        return self.__dict__

    def __setstate__(self, d):
        print("I'm being unpickled with these values:{}".format(d))
        self.__dict__ = d
        self.val *= 3


import pickle

f = Foo()
f_string = pickle.dumps(f)
f_new = pickle.loads(f_string)
