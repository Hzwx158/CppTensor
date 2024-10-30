import numpy as np
import torch
import timeit

timer=timeit.default_timer()
for i in range(0,10000):
    pass

print(str((timeit.default_timer()-timer)*1000)+'ms')