# Cpp Tensor: ShapedArray
Write this to learn numpy.
>## Symbols
To represent the relationship between shape and tensors, I use some symbols:
- $\text{\$}\vec{s}$ means a tensor with a shape of $\vec{s}$. For instance, $\text{\$}(3,4)$ means a matrix with shape being $(3,4)$. 
- shape($\vec{t}$) means the shape of tensor $\vec{t}$. 

You can see such representations in docs of functions.
>## Files
- ./utils
  + base.h: to implement some cpp grammer and some easy basic functions.
  + errors.h(.cpp): implement an error module.
  + pointer.hpp: to realize my own `UniquePointer` type.
  + list.hpp(.cpp): implement a `FixedArray` class. Compared to `std::vector`, it has no `push_back` or `pop_back`, but it can be assigned to anothor `FixedArray` instance with different length. Using memset/memcpy to accelerate the process of copying.
  + thread_pool.hpp(.cpp): implement a simple thread pool.
- ./shaped
  + shape.hpp(.cpp): implement class `Shape`, which constributes to the shape control. 
  + array.hpp: to declare class `ShapedArray`.
  + _array_idx.hpp: implement `ShapedArray::at(...)`. 
  + _array_arr.hpp: implement constructor, destroyer and some member functions of `ShapedArray` .
  + _array_op.hpp: implement some operations on `ShapedArray`.
- ./autograd
  + tensor.hpp(.cpp): implement some a computation map data structure `Var`.
  + tensor_grad.cpp: implement the functions of computing gradiants according to the map.
>## Classes
### UniquePointer
This class is to help guarantee a pointer will be deleted automatically when not using it. Similar to std::unique_pointer.
```cpp
numcpp::UniquePointer<int> ptr(new int(10));
auto p = std::move(ptr);
```
### Shape
This class is to manage high-dim operations of `ShapedArray` class. It has two members (`FixedArray<size_t>`): `shape` and `product`.
- shape: record each dimension's size. e.g. {2,2,3}
- product: record the production of array "shape". For example, shape `{3,6,4}`'s corresponding `product` is `{3*6*4*1, 6*4*1, 4*1, 1}`. This member is to help calculate the "step size" of some dim.

You can use `std::cout` to output an instance of `Shape`.
```c++
numcpp::Shape shape{2,2,3};
std::cout << shape << std::endl;
```
### ShapedArray
See this as a shaped array. It has 2 members: mArray(UniquePointer) and shape(Shape). 
- mArray: to record the memory allocated. 
- shape: to record the shaped array's shape.

You can use `std::cout` to output an instance of `ShapedArray`.

Now I've realized the functions "broadcast","at","print", "reshape" and so on. But function like "sum" haven't been realized yet.  