import numpy as np
import torch
import timeit

timer=timeit.default_timer()

a = np.array([1,2,3,4,5,6,7,8]).reshape((2,2,2))
# print(a)
for i in range(2):
    a[[1,i],[i,1]]+=[i+1,i-2]
# print(a)
'''

'''

print(str((timeit.default_timer()-timer)*1000)+'ms')