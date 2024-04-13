# Cpp Tensor
Write this to learn numpy
>## Files
- /tensor/base.h : to implement some cpp grammer and some easy basic functions
- /tensor/pointer.hpp: to realize my own pointer type.
- /tensor/simplevector.hpp(.cpp): to change std::vector<size_t> and implement my own vector type. Using memset/memcpy to accelerate the process of vector copying.
- /tensor/tensor.hpp: to declare class Tensor
- /tensor/tensor_tensor.cpp: definition of class Tensor.
- /tensor/tensor_operators.cpp: definition of some operator functions of class Tensor.
- /tensor/tensor_reftensor.cpp: definition of class RefTensor.
>## Classes
### UniquePointer
This class is to help guarantee a pointer is deleted when not using it. Similar to std::unique_pointer
### Shape
This class is to manage high-dim operations of tensor class. It has two members (vector<size_t>): shape and product.
- shape: record each dimension's size. e.g. {2,2,3}
- product: record the production of array "shape". For example, shape {3,6,4}'s corresponding "product" is {$3*6*4*1, 6*4*1, 4*1, 1$}. This member is to help calculate the "step size" of some dim.
### Tensor
See this as a shaped array. It has 2 members: mArray(UniquePointer) and shape(Shape). 
- mArray: to record the memory allocated. 
- shape: to record the tensor's shape.

Now I've realized the functions "broadcast","at","print", "reshape" and so on. But function like "sum" haven't been realized yet.  