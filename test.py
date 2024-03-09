import numpy as np
import timeit

f=open('./pytime.txt','w')
#for _ in range(0,10000):
timer=timeit.default_timer()
a=np.array([1,2,3,4,5,6,7,8]).reshape(1,2,2,1,2)
toprint= a+1
print(str((timeit.default_timer()-timer)*1000)+',')
