#ifndef NUMCPP_SHAPED_PRIVATE_ARRAY_OP
#define NUMCPP_SHAPED_PRIVATE_ARRAY_OP
#include "./array.hpp"
#include <cmath>
#include "../utils/matmul.hpp"
namespace numcpp{

#define OP_DEF_CODE(opStr, opName)\
template<class DType>\
template<class T>\
ShapedArray<DType> &ShapedArray<DType>::operator opStr##= (const ShapedArray<T> &obj){\
    Shape const &obj_shape = obj.getShape();\
    if(Shape::broadcast(shape, obj_shape)!=shape)\
        throw Error::wrong(__FILE__,__func__,"Wrong shape!");\
    size_t l = shape.bufSize();\
    for(size_t i=0; i<l; ++i)\
        mArray[i] opStr##= *obj.data(Shape::offsetBeforeBroadcast(i, shape, obj_shape));\
    return *this;\
}\
template<class DType>\
template<class T>\
ShapedArray<DType> &ShapedArray<DType>::operator opStr##= (const T &num){\
    size_t l = shape.bufSize();\
    for(size_t i=0; i<l; ++i)\
        mArray[i] opStr##= num;\
    return *this;\
}\
template<class DType>\
template<class T>\
ShapedArray<op_ret_t<EOperation::opName, DType, T>> \
ShapedArray<DType>::operator opStr(const ShapedArray<T> &obj) const\
{\
    Shape const &obj_shape = obj.getShape();\
    Shape resShape = Shape::broadcast(shape, obj_shape);\
    size_t l = resShape.bufSize();\
    using Ret = decltype(*mArray opStr *obj.mArray);\
    auto ptr = new Ret[l];\
    if constexpr(DEBUG)\
        std::cout<<"Pointer Alloc @"<<static_cast<void*>(ptr)<<'['<<l<<']'<<std::endl;\
    auto obj_m = obj.mArray;\
    for(size_t i=0; i<l; ++i){\
        ptr[i] = mArray[Shape::offsetBeforeBroadcast(i, resShape, shape)]\
        opStr obj_m[Shape::offsetBeforeBroadcast(i, resShape, obj_shape)];\
    }\
    return ShapedArray<Ret>(std::move(ptr), std::move(resShape));\
}\
template<class DType>\
template<class T, class useless>\
ShapedArray<op_ret_t<EOperation::opName, DType, T>> \
ShapedArray<DType>::operator opStr(const T &num) const\
{\
    size_t l = shape.bufSize();\
    using Ret = decltype(*mArray opStr num);\
    auto ptr = new Ret[l];\
    if constexpr(DEBUG)\
        std::cout<<"Pointer Alloc @"<<static_cast<void*>(ptr)<<'['<<l<<']'<<std::endl;\
    for(size_t i=0; i<l; ++i){\
        ptr[i] = mArray[i] opStr num;\
    }\
    return ShapedArray<Ret>(std::move(ptr), shape);\
}\
template<class DType, class T, class useless = std::enable_if_t<!is_ShapedArray_v<T>>>\
ShapedArray<op_ret_t<EOperation::opName, T, DType>> \
operator opStr(const T &num, const ShapedArray<DType> &obj)\
{\
    Shape const &obj_shape = obj.getShape();\
    size_t l = obj_shape.bufSize();\
    using Ret = decltype(num opStr (*obj.data()));\
    auto ptr = new Ret[l];\
    auto obj_m = obj.data();\
    if constexpr(DEBUG)\
        std::cout<<"Pointer Alloc @"<<static_cast<void*>(ptr)<<'['<<l<<']'<<std::endl;\
    for(size_t i=0; i<l; ++i){\
        ptr[i] = num opStr obj_m[i];\
    }\
    return ShapedArray<Ret>(std::move(ptr), obj_shape);\
}
OP_DEF_CODE(+, ADD)
OP_DEF_CODE(-, SUB)
OP_DEF_CODE(*, MUL)
OP_DEF_CODE(%, MOL)
#undef OP_DEF_CODE

//----------------------------------------除法要特供-------------------------------------
inline double _div_(double a, double b){
    if(!b){
        if(!a) return NAN;
        return a>0 ? constant::float64_inf: constant::float64_ninf;
    }
    return a/b;
}
template<class T>
inline T &_div_assigned_(T &a, double b){
    if(!b){
        if(!a) return NAN;
        if constexpr(std::is_unsigned_v<T>)
            return inf_v<T>;
        else return a = (a>0 ? inf_v<T>: ninf_v<T>);
    }
    return a/=b;
}

template<class DType>
template<class T>
ShapedArray<DType> &ShapedArray<DType>::operator/=(const ShapedArray<T> &obj){
    Shape const &obj_shape = obj.getShape();
    if(Shape::broadcast(shape, obj_shape)!=shape)
        throw Error::wrong(__FILE__,__func__,"Wrong shape!");
    size_t l = shape.bufSize();
    for(size_t i=0; i<l; ++i)
        _div_assigned_(mArray[i],*obj.data(Shape::offsetBeforeBroadcast(i, shape, obj_shape)));
    return *this;
}
template<class DType>
template<class T>
ShapedArray<DType> &ShapedArray<DType>::operator/= (const T &num){
    size_t l = shape.bufSize();
    for(size_t i=0; i<l; ++i)
        _div_assigned_(mArray[i], num);
    return *this;
}
template<class DType>
template<class T>
ShapedArray<op_ret_t<EOperation::DIV, DType, T>> 
ShapedArray<DType>::operator/(const ShapedArray<T> &obj) const
{
    Shape const &obj_shape = obj.getShape();
    Shape resShape = Shape::broadcast(shape, obj_shape);
    size_t l = resShape.bufSize();
    auto ptr = new double[l];
    if constexpr(DEBUG)
        std::cout<<"Pointer Alloc @"<<static_cast<void*>(ptr)<<'['<<l<<']'<<std::endl;
    auto obj_m = obj.mArray;
    for(size_t i=0; i<l; ++i){
        ptr[i] = _div_(
            mArray[Shape::offsetBeforeBroadcast(i, resShape, shape)],
            obj_m[Shape::offsetBeforeBroadcast(i, resShape, obj_shape)]
        );
    }
    return ShapedArray<double>(std::move(ptr), std::move(resShape));
}
template<class DType>
template<class T, class useless>
ShapedArray<op_ret_t<EOperation::DIV, DType, T>>
ShapedArray<DType>::operator/(const T &num) const
{
    size_t l = shape.bufSize();
    auto ptr = new double[l];
    if constexpr(DEBUG)
        std::cout<<"Pointer Alloc @"<<static_cast<void*>(ptr)<<'['<<l<<']'<<std::endl;
    for(size_t i=0; i<l; ++i){
        ptr[i] = _div_(mArray[i], num);
    }
    return ShapedArray<double>(std::move(ptr), shape);
}
template<class DType, class T, class useless = std::enable_if_t<!is_ShapedArray_v<T>>>
ShapedArray<op_ret_t<EOperation::DIV, T, DType>>
operator/(const T &num, const ShapedArray<DType> &obj)
{
    Shape const &obj_shape = obj.getShape();
    size_t l = obj_shape.bufSize();
    auto ptr = new double[l];
    auto obj_m = obj.data();
    if constexpr(DEBUG)
        std::cout<<"Pointer Alloc @"<<static_cast<void*>(ptr)<<'['<<l<<']'<<std::endl;
    for(size_t i=0; i<l; ++i){
        ptr[i] = _div_(num, obj_m[i]);
    }
    return ShapedArray<double>(std::move(ptr), obj_shape);
}
//--------------------------------------逻辑运算符---------------------------------------

#define LOGICAL_OP_DEF_CODE(opStr)\
template<class DType>\
template<class T>\
ShapedArray<bool> ShapedArray<DType>::operator opStr(const ShapedArray<T> &obj) const{\
    Shape obj_shape = obj.getShape();\
    Shape resShape = Shape::broadcast(shape, obj_shape);\
    size_t l = resShape.bufSize();\
    bool *ptr = new bool[l];\
    auto obj_m = obj.data();\
    for(size_t i=0;i<l;++i){\
        ptr[i] = mArray[Shape::offsetBeforeBroadcast(i, resShape, shape)] \
            opStr obj_m[Shape::offsetBeforeBroadcast(i, resShape, obj_shape)];\
    }\
    return ShapedArray<bool>(std::move(ptr), std::move(resShape));\
}\
template<class DType>\
template<class T>\
ShapedArray<bool> ShapedArray<DType>::operator opStr(const T &num) const{\
    size_t l = shape.bufSize();\
    bool *ptr = new bool[l];\
    for(size_t i=0;i<l;++i){\
        ptr[i] = mArray[i] opStr num;\
    }\
    return ShapedArray<bool>(std::move(ptr), shape);\
}

LOGICAL_OP_DEF_CODE(>)
LOGICAL_OP_DEF_CODE(>=)
LOGICAL_OP_DEF_CODE(<)
LOGICAL_OP_DEF_CODE(<=)
LOGICAL_OP_DEF_CODE(==)
LOGICAL_OP_DEF_CODE(!=)
#undef LOGICAL_OP_DEF_CODE
template<class T1, class T2>
int _compare_(T1 const &a, T2 const &b){
    if(a>b) return 1;
    else if(a<b) return -1;
    return 0;
}
template<class DType>
template<class T>
ShapedArray<int> ShapedArray<DType>::compare(const ShapedArray<T> &obj) const{
    Shape obj_shape = obj.getShape();
    Shape resShape = Shape::broadcast(shape, obj_shape);
    size_t l = resShape.bufSize();
    int *ptr = new int[l];
    auto obj_m = obj.data();
    for(size_t i=0;i<l;++i){
        ptr[i] = numcpp::_compare_(
            mArray[Shape::offsetBeforeBroadcast(i, resShape, shape)],
            obj_m[Shape::offsetBeforeBroadcast(i, resShape, obj_shape)]
        );
    }
    return ShapedArray<int>(std::move(ptr), std::move(resShape));
}
template<class DType>
template<class T>
ShapedArray<int> ShapedArray<DType>::compare(const T &num) const{
    size_t l = shape.bufSize();
    int *ptr = new int[l];
    for(size_t i=0;i<l;++i){
        ptr[i] = numcpp::_compare_(mArray[i], num);
    }
    return ShapedArray<int>(std::move(ptr), shape);
}

//----------------------------------------三角函数---------------------------------------
#define TRI_OP_DEF_CODE(func, ...)\
template<class DType>\
ShapedArray<double> ShapedArray<DType>::func() const{\
    size_t l = shape.bufSize();\
    double *ptr = new double[l];\
    for(size_t i=0;i<l;++i){\
        ptr[i] = __VA_ARGS__ (mArray[i]);\
    }\
    return ShapedArray<double>(std::move(ptr), shape);\
}\
template<class DType>\
inline ShapedArray<double> func(const ShapedArray<DType> &obj){return obj.func();}

TRI_OP_DEF_CODE(exp, std::exp)
TRI_OP_DEF_CODE(log, std::log)
TRI_OP_DEF_CODE(sin, std::sin)
TRI_OP_DEF_CODE(cos, std::cos)
TRI_OP_DEF_CODE(tan, std::tan)
TRI_OP_DEF_CODE(asin, std::asin)
TRI_OP_DEF_CODE(acos, std::acos)
TRI_OP_DEF_CODE(atan, std::atan)
TRI_OP_DEF_CODE(sec, 1/std::cos)
TRI_OP_DEF_CODE(csc, 1/std::sin)
TRI_OP_DEF_CODE(cot, 1/std::tan)
#undef TRI_OP_DEF_CODE

template<class DType>
template<class T>
ShapedArray<op_ret_t<EOperation::MUL, DType, T>> ShapedArray<DType>::matmul(const ShapedArray<T> &obj) const
{
    Shape const &obj_shape = obj.getShape();
    if(shape.dimNumber()!=2||obj_shape.dimNumber()!=2||shape[1]!=obj_shape[0])
        throw Error::wrong(__FILE__, __func__,"Wrong Shape!");
    auto res = numcpp::fill<op_ret_t<EOperation::MUL, DType, T>>(0, {shape[0], obj_shape[1]});
    linalg::_matmul(mArray+0, obj.data(), res.data(), shape[0], obj_shape[0], obj_shape[1]);
    return res;
}
}
#endif