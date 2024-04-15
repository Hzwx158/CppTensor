import numpy as np
import torch
import timeit

timer=timeit.default_timer()

x=torch.tensor([1.], requires_grad=True)
y=torch.tensor([3.], requires_grad=True)
z=2*x*x+y
z.backward()
print(x.grad)



print(str((timeit.default_timer()-timer)*1000)+',')
