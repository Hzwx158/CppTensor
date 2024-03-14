import numpy as np
import timeit

f=open('./pytime.txt','w')
timer=timeit.default_timer()
#for _ in range(0,10000):
a=np.arange(start=0,stop=8).reshape(2,2,2)


toprint=a
print(toprint)

print(str((timeit.default_timer()-timer)*1000)+',')
