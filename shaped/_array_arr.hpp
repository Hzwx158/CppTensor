#ifndef NUMCPP_SHAPED_PRIVATE_ARRAY_ARR
#define NUMCPP_SHAPED_PRIVATE_ARRAY_ARR
#include "./array.hpp"
namespace numcpp{
//--------------------------------内存---------------------------
template<class DType>
ShapedArray<DType>::ShapedArray(std::initializer_list<DType> init_list, const Shape &shape_)
:mArray(init_list.size()? new DType[init_list.size()]:nullptr)
,shape(shape_.empty()?Shape({init_list.size()}):shape_)
{
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    if(init_list.size()!=shape.bufSize())
        throw Error::wrong(__FILE__, __func__, "Shape unsame!");
    if(init_list.size())
        memcpy(mArray, init_list.begin(), init_list.size()*sizeof(DType));
}

template<class DType>
ShapedArray<DType>::ShapedArray(const std::vector<DType> &init_vec, const Shape &shape_)
:mArray(init_vec.size()? new DType[init_vec.size()]:nullptr)
,shape(shape_.empty()?Shape({init_vec.size()}):shape_)
{
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    if(init_vec.size()!=shape.bufSize())
        throw Error::wrong(__FILE__, __func__, "Shape unsame!");   
    if(init_vec.size())
        memcpy(mArray, init_vec.data(), init_vec.size()*sizeof(DType));
}

template<class DType>
ShapedArray<DType>::ShapedArray(const DType &num)
:mArray(new DType(num))
,shape({}){
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
}

template<class DType>
ShapedArray<DType>::ShapedArray(const std::vector<ShapedArray> &tensors, const Shape &shape_)
{
    if((!shape_.empty())&&tensors.size()!=shape_.bufSize())
        throw Error::wrong(__FILE__, __func__, "Wrong shape");
    const size_t tCnt=tensors.size();
    for(size_t i=0;i<tCnt-1;++i)
        if(tensors[i].shape!=tensors[i+1].shape)
            throw Error::wrong(__FILE__, __func__, "Tensors should have same shape!");
    shape = (shape_.empty()?Shape{tCnt}:shape_) + tensors[0].shape;
    mArray = new DType[shape.bufSize()];
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    const size_t stepSize = tensors[0].shape.bufSize();
    for(size_t i=0;i<tCnt;++i)
    {
        // tensors[i].print();
        memcpy(mArray+i*stepSize, tensors[i].mArray, sizeof(DType)*stepSize);
    }
}

template<class DType>
ShapedArray<DType>::ShapedArray(const ShapedArray &obj)
    :mArray(new DType[obj.shape.bufSize()])
    ,shape(obj.shape)
{
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    if(shape.empty()) 
        throw Error::author(__FILE__, __func__, "Shape can't be empty");
    auto size=shape.bufSize();
    memcpy(mArray, obj.mArray, sizeof(DType)*size);
}

template<class DType>
ShapedArray<DType>::ShapedArray(ShapedArray &&obj)
    :mArray(std::move(obj.mArray))
    ,shape(std::move(obj.shape))
{
    //obj.clear();
}

template<class DType>
void ShapedArray<DType>::clear(){
    mArray.clear();
    shape.clear();
}

template<class DType>
ShapedArray<DType> &ShapedArray<DType>::operator=(
    std::conditional_t<
        std::is_pointer_v<DType>, 
        ShapedArray<std::remove_pointer_t<DType>>,
        ShapedArray<DType>
    > const &obj
){
    if constexpr(std::is_pointer_v<DType>){
        // pointer
        Shape const &obj_shape = obj.getShape();
        if(Shape::broadcast(shape, obj_shape)!=shape)
            throw Error::wrong(__FILE__,__func__,"Different Shape");
        size_t l = shape.bufSize();
        DType src = obj.data();
        for(size_t i=0;i<l;++i)
            *mArray[i] = src[Shape::offsetBeforeBroadcast(i, shape, obj_shape)];
        return *this;
    }else{
        // not pointer
        if(this==&obj) 
            return *this;
        if(obj.shape.empty()){
            clear();
            return *this;
        }
        shape = obj.shape;
        mArray = new DType[shape.bufSize()];
#if DEBUG
        std::cout<<"Pointer Alloc @"<<static_cast<void*>(mArray)<<std::endl;
#endif
        memcpy(mArray, obj.mArray, shape.bufSize()*sizeof(DType));
        return *this;
    }
}

template<class DType>
ShapedArray<DType> &ShapedArray<DType>::operator=(
    std::conditional_t<
        std::is_pointer_v<DType>, 
        ShapedArray<std::remove_pointer_t<DType>>,
        ShapedArray<DType>
    > &&obj
){
    if constexpr(std::is_pointer_v<DType>){
        Shape const &obj_shape = obj.getShape();
        if(Shape::broadcast(shape, obj_shape)!=shape)
            throw Error::wrong(__FILE__,__func__,"Different Shape");
        size_t l = shape.bufSize();
        DType src = obj.data();
        for(size_t i=0;i<l;++i)
            *mArray[i] = std::move(src[Shape::offsetBeforeBroadcast(i, shape, obj_shape)]);
        return *this;
    }else{
        if(this==&obj)
            return *this;
        clear();
        if(obj.shape.empty())
            return *this;
        shape=std::move(obj.shape);
        mArray = std::move(obj.mArray);
        //obj.clear();
        return *this;
    }
}
//-------------------------其他初始化-----------------------

/**
 * @brief 填充
 * @param num 填充值
 * @param shape 填充的shape
 */
template<class T>
ShapedArray<T> fill(const T &num, const Shape &shape){
    size_t l=shape.bufSize();
    T *ptr=new T[l];
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(ptr)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    if((!num)||(num==(T)(-1)))
        memset(ptr, (size_t)num, sizeof(T)*l);
    else for(size_t i=0;i<l;++i)
        ptr[i] = num;
    return ShapedArray<T>(std::move(ptr), shape);
}

template<class T>
ShapedArray<T> arange(const T &be, const T &en, const T &step = 1, const Shape &shape=Shape()){
    if(step==0)
        throw Error::wrong(__FILE__, __func__, "<step> shouldn't be 0!");
    if((en-be)*step<0)
        throw Error::wrong(
            __FILE__, __func__, 
            "arange("+Error::llToStr(be)+","+Error::llToStr(en)
            +","+Error::llToStr(step)+") is illegal"
        );
    size_t len = numcpp::ceil(en-be, step);
    if((!shape.empty())&&shape.bufSize()!=len)
        throw Error::wrong(__FILE__,__func__, "Wrong shape size");
    T *ptr = new T[len];
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(ptr)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    for(size_t i=0;i<len;++i)
        ptr[i] = be+i*step;
    if(shape.empty())
        return ShapedArray<T>(std::move(ptr), Shape({len}));
    else return ShapedArray<T>(std::move(ptr), shape);
}
//-------------------------功能-----------------------------

template<class DType>
void ShapedArray<DType>::to(const Shape &shape_){
    if(shape.bufSize()!=shape_.bufSize())
        throw Error::wrong(__FILE__, __func__, "Wrong Size");
    shape=shape_;
}

template<class DType>
template<class T>
ShapedArray<T> ShapedArray<DType>::to() const{
    T *ptr = new T[shape.bufSize()];
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(ptr)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    size_t l = shape.bufSize();
    for(size_t i=0;i<l;++i){
        ptr[i] = T(mArray[i]); 
    }
    return ShapedArray<T>(std::move(ptr), shape);
}

template<class DType>
ShapedArray<DType> ShapedArray<DType>::reshape(const Shape &shape_){
    ShapedArray<DType> res = *this;
    res.to(shape_);
    return res;
}


// ShapedRefArray ShapedArray::where(std::function<bool(const DType &)> cond) const{
//     const auto size=shape.bufSize();
//     DType *head=mArray;
//     PtrVector<DType> res(size);
//     size_t k=0;
//     for(size_t i=0;i<size;++i)
//         if(cond(head[i]))
//             res[k++]=(head+i);
//     res.shrinkTo(k);
//     return res;
// }

// METHOD(DType**)::atByNumbers(const Shape::SizeTArray &index, DType **resBegin) const
// {
//     const auto argCount = index.size();
//     if(index.empty()||argCount>shape.dimNumber())
//         throw std::runtime_error("From ShapedArray::at(const vector &):\n\tWrong size of index");
//     auto [offset, resCnt] = shape.offsetOf(index);
//     DType *be=mArray+offset;
//     for(size_t i=0;i<resCnt;++i)
//         resBegin[i]=be+i;
//     return resBegin+resCnt;
// }

//----------------------静态/友元--------------------------

// ShapedArray zeros(const Shape &shape) {return ShapedArray(0,shape);}
// ShapedArray ones(const Shape &shape) {return ShapedArray(1,shape);}
// ShapedArray arange(ShapedArray::DType be, ShapedArray::DType en, ShapedArray::DType step, const Shape &shape){
//     if(en==be)
//         return ShapedArray();
//     if((en-be)*step<=0)
//         throw std::runtime_error("From ShapedArray::arange:\n\tWrong step");
//     size_t cnt = static_cast<size_t>(abs((en-be)/step));
//     if((!shape.empty())&&(shape.bufSize()!=cnt))
//         throw std::runtime_error("From ShapedArray::arange:\n\tWrong shape");
//     if(!cnt)
//         return ShapedArray();
//     ShapedArray::NumberPtr p=new ShapedArray::DType[cnt];
// #if DEBUG
//     std::cout<<"Pointer Alloc @"<<static_cast<void*>(p)<<'['<<cnt<<']'<<std::endl;
// #endif
//     ShapedArray::DType *h=p;
//     for(;be<en;be+=step,++h)
//         *h=be;
//     if(shape.empty())
//         return ShapedArray(std::move(p), Shape{cnt});
//     else return ShapedArray(std::move(p), shape);//调用构造函数不一样，别省这个if
// }

}

#endif