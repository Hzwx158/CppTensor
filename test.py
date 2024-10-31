import numpy as np
import timeit
def func(v):
    print(f'v:{v}\nshape:{np.shape(v)}')

start_time = timeit.default_timer()
a = np.arange(12).reshape((2,2,3))+1

func( a>np.array([1]).reshape(1,1,1,1))

end_time = timeit.default_timer()
print(f'-------\nusing time: {(end_time-start_time)*1000}ms')

