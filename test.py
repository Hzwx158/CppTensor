import numpy as np
a=np.array([1,2,3,4,5,6,7,8,9,10,11,12]).reshape(2,2,3)
a[0,1]=[100,99,98]
print(
    a
)