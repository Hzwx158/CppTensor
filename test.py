import numpy as np
import timeit

a = np.arange(12).reshape((2,2,3))+1
start_time = timeit.default_timer()

a[1, :, 0:-1] = [100,99]
# print(a)

end_time = timeit.default_timer()
print(f'-------\nusing time: {(end_time-start_time)*1000}ms')

# def func(v):
#     print(f'v:{v}\nshape:{np.shape(v)}')

# func(a[[[1],[0]],:,:])