from multiprocess.dummy import Process

def print_hello(n):
    print(n)

print(1)
lis=[]
for i in range(10):
    p = Process(target=print_hello,args=(i,))
    p.start()
    lis.append(p)

for t in lis:
    t.join()

print("finished")