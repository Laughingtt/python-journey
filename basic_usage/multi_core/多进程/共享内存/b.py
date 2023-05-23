from multiprocessing import shared_memory

shm_b = shared_memory.ShareableList(name='123')
print(shm_b[0])  # ‘张三’
print(shm_b[1])  # 2
print(shm_b[2])  # ‘abc
