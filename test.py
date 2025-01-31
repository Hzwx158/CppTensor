import numpy as np
import timeit
import torch
from pprint import pprint

def func(v):
    print(f'{v}\nshape:{np.shape(v)}')

def test_time(f):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        f(*args, **kwargs)
        end_time = timeit.default_timer()
        print(f'-------\nusing time: {(end_time-start_time)*1000}ms')
    return wrapper

a = np.arange(10000).reshape(10,1000) #.astype(np.int32)
b = np.ones((1000,20)) #.astype(np.int32)

@test_time
def main():
    a = torch.tensor([1., 2., 3.], requires_grad=True)
    b = torch.tensor([4., 5., 6.], requires_grad=True)
    c = a + b
    pprint(type(c.grad_fn))

if __name__=='__main__':
    main()

