import numpy as np
import timeit
def func(v):
    print(f'{v}\nshape:{np.shape(v)}')

start_time = timeit.default_timer()

a = np.arange(10000).reshape(10,1000).astype(np.float32)
b = np.zeros((1000,20)).astype(np.int8)
for i in range(10000):
    a@b

end_time = timeit.default_timer()
print(f'-------\nusing time: {(end_time-start_time)*1000}ms')

