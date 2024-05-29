import numpy as np
import torch
import timeit

timer=timeit.default_timer()
np.ndarray
a = np.array([1,2,3,4,5,6,7,8,9,10,11,12]).reshape((2,2,3))
print(a)
b = a[
    [[[1,1]],[[0,0]]],
    [[1,1],[1,1]], 
    [[1,2],[1,2]]
]
print(b)
b = a[
    [[1],[0]],
    1,
    [1,2]
]
print(b)
'''
    [[[1,1]],[[0,0]]],
    [[1,1],[1,1]],
    [[1,2],[1,2]]
'''

print(str((timeit.default_timer()-timer)*1000)+'ms')