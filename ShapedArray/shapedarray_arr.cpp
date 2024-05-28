#include "./shapedarray.hpp"
#include <cstring>
namespace tryAI{


//--------------------------------内存---------------------------

ShapedArray::ShapedArray(std::initializer_list<Number> init_list, const Shape &shape_)
:mArray(init_list.size()? new Number[init_list.size()]:nullptr)
,shape(shape_.isEmpty()?Shape({init_list.size()}):shape_)
{
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    if(init_list.size()!=shape.bufSize())
        throw std::runtime_error("From ShapedArray::ShapedArray({},const Shape &):\n\tShape unsame!");    
    if(init_list.size())
        memcpy(mArray, init_list.begin(), init_list.size()*sizeof(Number));
}
ShapedArray::ShapedArray(Number num, const Shape &shape_)
:mArray(shape_.isEmpty()? nullptr : new Number[shape_.bufSize()])
,shape(shape_){
    if(shape.isEmpty())
        throw std::runtime_error("From ShapedArray::ShapedArray(Number, const Shape &):\n\tEmpty shape!");
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    const auto size=shape.bufSize();
    for(size_t i=0;i<size;++i)
        mArray[i]=num;
}
ShapedArray::ShapedArray(const std::vector<ShapedArray> &tensors, const Shape &shape_)
{
    if((!shape_.isEmpty())&&tensors.size()!=shape_.bufSize())
        throw std::runtime_error("From ShapedArray::ShapedArray(vector<ShapedArray>, Shape):\n\tWrong shape");
    const size_t tCnt=tensors.size();
    for(size_t i=0;i<tCnt-1;++i)
        if(tensors[i].shape!=tensors[i+1].shape)
            throw std::runtime_error("From ShapedArray::ShapedArray(vector<ShapedArray>, Shape):\n\tTensors should have same shape");
    shape = shape_ + tensors[0].shape;
    mArray = new Number[shape.bufSize()];
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    const size_t stepSize = tensors[0].shape.bufSize();
    for(size_t i=0;i<tCnt;++i)
        memcpy(mArray+i*stepSize, tensors[i].mArray, sizeof(Number)*stepSize);
}
ShapedArray::ShapedArray(const ShapedArray &obj)
:mArray(new Number[obj.shape.bufSize()])
,shape(obj.shape)
{
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    if(shape.isEmpty()) 
        throw std::runtime_error("@Author from ShapedArray::ShapedArray(const ShapedArray &):\n\tshape can't be empty");
    auto size=shape.bufSize();
    memcpy(mArray, obj.mArray, sizeof(Number)*size);
}
ShapedArray::ShapedArray(ShapedArray &&obj)
:mArray(std::move(obj.mArray)),shape(std::move(obj.shape))
{
    //obj.clear();
}
void ShapedArray::clear(){
    mArray.clear();
    shape.clear();
}
ShapedArray &ShapedArray::operator=(const ShapedArray &obj){
    if(this==&obj) 
        return *this;
    clear();
    if(obj.shape.isEmpty())
        return *this;
    shape=obj.shape;
    mArray.reset(new Number[shape.bufSize()]);
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(mArray)<<std::endl;
#endif
    memcpy(mArray, obj.mArray, shape.bufSize()*sizeof(Number));
    return *this;
}
ShapedArray &ShapedArray::operator=(ShapedArray &&obj){
    if(this==&obj)
        return *this;
    clear();
    if(obj.shape.isEmpty())
        return *this;
    shape=std::move(obj.shape);
    mArray = std::move(obj.mArray);
    //obj.clear();
    return *this;
}

//-------------------------功能-----------------------------

std::ostream &operator<<(std::ostream &osm, const ShapedArray &obj){
    printShaped<ShapedArray::Number>(obj.mArray,obj.shape,osm);
    return osm/*<<"\nshape: "<<obj.shape*/;
}
void ShapedArray::reshape(const Shape &shape_){
    if(shape.bufSize()!=shape_.bufSize())
        throw std::runtime_error("From ShapedArray::reshape(const Shape &):\n\tWrong Size");
    shape=shape_;
}

void ShapedArray::foreach(std::function<void(Number&)> func){
    const auto size=shape.bufSize();
    Number *head=mArray;
    for(size_t i=0;i<size;++i)
        func(head[i]);
}
ShapedRefArray ShapedArray::at(std::function<bool(const Number &)> cond) const{
    const auto size=shape.bufSize();
    Number *head=mArray;
    PtrVector<Number> res(size);
    size_t k=0;
    for(size_t i=0;i<size;++i)
        if(cond(head[i]))
            res[k++]=(head+i);
    res.shrinkTo(k);
    return res;
}

//----------------------静态--------------------------

ShapedArray ShapedArray::arange(Number be, Number en, Number step, const Shape &shape){
    if(en==be)
        return ShapedArray();
    if((en-be)*step<=0)
        throw std::runtime_error("From ShapedArray::arange:\n\tWrong step");
    size_t cnt = static_cast<size_t>(abs((en-be)/step));
    if((!shape.isEmpty())&&(shape.bufSize()!=cnt))
        throw std::runtime_error("From ShapedArray::arange:\n\tWrong shape");
    if(!cnt)
        return ShapedArray();
    NumberPtr p=new Number[cnt];
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(p)<<'['<<cnt<<']'<<std::endl;
#endif
    Number *h=p;
    for(;be<en;be+=step,++h)
        *h=be;
    if(shape.isEmpty())
        return ShapedArray(std::move(p), Shape{cnt});
    else return ShapedArray(std::move(p), shape);//调用构造函数不一样，别省这个if
}

}