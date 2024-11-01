# Cpp Tensor: ShapedArray
Write this to learn numpy.
>## Files
- ./utils
  + base.h: to implement some cpp grammer and some easy basic functions.
  + pointer.hpp: to realize my own `UniquePointer` type.
  + list.hpp(.cpp): implement a `FixedArray` class. Compared to `std::vector`, it has no `push_back` or `pop_back`, but it can be assigned to anothor `FixedArray` instance with different length. Using memset/memcpy to accelerate the process of copying.
  + errors.h(.cpp): implement an error module.
- ./shaped
  + shape.hpp(.cpp): implement class `Shape`, which constributes to the shape control. 
  + array.hpp: to declare class `ShapedArray`.
  + _array_idx.hpp: implement `ShapedArray::at(...)`. 
  + _array_arr.hpp: implement constructor, destroyer and some member functions of `ShapedArray` .
>## Classes
### UniquePointer
This class is to help guarantee a pointer will be deleted automatically when not using it. Similar to std::unique_pointer.
### Shape
This class is to manage high-dim operations of `ShapedArray` class. It has two members (`FixedArray<size_t>`): `shape` and `product`.
- shape: record each dimension's size. e.g. {2,2,3}
- product: record the production of array "shape". For example, shape `{3,6,4}`'s corresponding `product` is `{3*6*4*1, 6*4*1, 4*1, 1}`. This member is to help calculate the "step size" of some dim.

You can use `std::cout` to output an instance of `Shape`.
### ShapedArray
See this as a shaped array. It has 2 members: mArray(UniquePointer) and shape(Shape). 
- mArray: to record the memory allocated. 
- shape: to record the shaped array's shape.

You can use `std::cout` to output an instance of `Shape`.

Now I've realized the functions "broadcast","at","print", "reshape" and so on. But function like "sum" haven't been realized yet.  