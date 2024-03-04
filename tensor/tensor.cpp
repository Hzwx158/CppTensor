#include "./tensor.hpp"
namespace tryAI{
//--------------------------------内存---------------------------

Tensor::Tensor(const std::vector<Number> &init_list, const Shape &shape_)
:mArray((init_list.size()?new Number[init_list.size()]:nullptr))
,shape(shape_.isEmpty()?Shape({init_list.size()}):shape_)
{
#if DEBUG
    std::cout<<"Tensor Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    Number *pItem=mArray;
    for(auto &num:init_list)
        *(pItem++)=num;
}
Tensor::Tensor(Number num)
:mArray(new Number(num))
,shape({}){
#if DEBUG
    std::cout<<"Tensor Alloc @"<<static_cast<void*>(mArray)<<std::endl;
#endif
}
Tensor::Tensor(const Tensor &obj)
:mArray((obj.getSize()?new Number[obj.getSize()]:nullptr))
,shape(obj.shape)
{
#if DEBUG
    std::cout<<"Tensor Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    if(shape.isEmpty()) 
        throw std::runtime_error("@Author from Tensor::Tensor(const Tensor &):\n\tshape can't be empty");
    Number *p=mArray;
    Number *oP=obj.mArray;
    const auto size=shape.bufSize();
    for(size_t i=0;i<size;++i)
        p[i]=oP[i];
}
Tensor::Tensor(Tensor &&obj)
:mArray(std::move(obj.mArray)),shape(std::move(obj.shape))
{
    obj.clear();
}
void Tensor::clear(){
    mArray.clear();
    shape.clear();
}
Tensor &Tensor::operator=(const Tensor &obj){
    if(this==&obj) 
        return *this;
    clear();
    if(obj.shape.isEmpty())
        return *this;
    shape=obj.shape;
    mArray.reset(new Number[shape.bufSize()]);
#if DEBUG
    std::cout<<"Tensor Alloc @"<<static_cast<void*>(mArray)<<std::endl;
#endif
    Number *p=mArray;
    Number *oP=obj.mArray;
    const auto size=shape.bufSize();
    for(size_t i=0;i<size;++i)
        p[i]=oP[i];
    return *this;
}
Tensor &Tensor::operator=(Tensor &&obj){
    if(this==&obj)
        return *this;
    clear();
    if(obj.shape.isEmpty())
        return *this;
    shape=std::move(obj.shape);
    mArray = std::move(obj.mArray);
    obj.clear();
    return *this;
}

std::ostream &operator<<(std::ostream &osm, const Tensor &obj){
    printShaped<Tensor::Number>(obj.mArray,0,obj.shape,osm);
    return osm<<"\nshape: "<<obj.shape;
}

//-------------------------功能-----------------------------

void Tensor::reshape(const Shape &shape_){
    if(shape.bufSize()!=shape_.bufSize())
        throw std::runtime_error("From Tensor::reshape(const Shape &):\n\tWrong Size");
    shape=shape_;
}
/*
RefTensor Tensor::at(const std::vector<size_t> &index) const {
    if(index.empty()||index.size()>shape.dimNumber())
        throw std::runtime_error("From Tensor::at(const vector &):\n\tWrong size of index");
    Number *head=mArray;
    Number *be=head,*en;
    const auto argCount=index.size();
    size_t idx;
    size_t dimSize;
    for(size_t i=0;i<argCount;++i){
        dimSize=shape.dimSizeOf(i);
        if(!toBoundedIdx(index[i],dimSize,&idx))
            throw std::out_of_range("From Tensor::at(const vector &):\n\tOut of range");
        be+=idx*shape.stepSizeOf(i);
    }
    en=be+shape.stepSizeOf(argCount-1);
    std::vector<Number*> res;
    for(;be!=en;++be)
        res.push_back(be);
    RefTensor rt(res);
    rt.shape=shape.sliced(argCount);
    return rt;
}
RefTensor Tensor::operator[](const std::vector<size_t> &index) const{
    return at(index);
}*/

void Tensor::foreach(std::function<void(Number&)> func){
    const auto size=shape.bufSize();
    Number *head=mArray;
    for(size_t i=0;i<size;++i)
        func(head[i]);
}
RefTensor Tensor::filter(std::function<bool(const Number &)> cond) const{
    const auto size=shape.bufSize();
    Number *head=mArray;
    std::vector<Number*> res;
    for(size_t i=0;i<size;++i)
        if(cond(head[i]))
            res.push_back(head+i);
    return res;
}

//-----------------------RefTensor---------------------

RefTensor::RefTensor(const std::vector<Number*> &init_list)
:mArray(nullptr), shape({init_list.size()})
{
    auto size=init_list.size();
    if(!size) {
        shape.clear();
        return;
    }
    mArray=new NumberPtr[size];
#if DEBUG
    std::cout<<"RefTensor Alloc @"<<static_cast<void*>(mArray)<<'['<<size<<']'<<std::endl;
#endif
    auto p=mArray;
    for(auto &pNum:init_list)
        *(p++)=pNum;
}
RefTensor::RefTensor(RefTensor &&obj)
:mArray(obj.mArray), shape(std::move(obj.shape))
{
    obj.mArray=nullptr;
}
RefTensor::RefTensor(const std::vector<RefTensor> &refTensors)
:mArray(nullptr),shape()
{
    if(refTensors.empty()) return;
    auto &be=refTensors[0];
    auto argSize=refTensors.size();
    if(argSize>1){
        //浪费时间，但不会反复new delete
        for(auto iter=refTensors.begin()+1;iter!=refTensors.end();++iter)
            if(be.shape!=iter->shape)
                throw std::runtime_error("From RefTensor(const vector<Ref> &):\n\tWrong shape");
    }
    auto shapeDimNumber=be.shape.dimNumber();//shape原本维度数
    auto shapeBufSize=be.shape.bufSize();//每个RefTensor的占内存数
    Shape::Vector vec(shapeDimNumber+1,0);//this->shape的初始化vec
    auto &s=be.shape.asVector();//每个RefTensor的shape
    vec[0]=refTensors.size();//新添的一维度
    for(size_t i=0;i<shapeDimNumber;++i)
        vec[i+1]=s[i];//把原shape拷贝进去
    shape=Shape(std::move(vec));
    mArray=new NumberPtr[shapeBufSize*argSize];
#if DEBUG
    std::cout<<"RefTensor Alloc @"<<static_cast<void*>(mArray)<<'['<<shapeBufSize*argSize<<']'<<std::endl;
#endif
    for(size_t i=0;i<argSize;++i){
        auto &m=refTensors[i].mArray;
        for(size_t j=0;j<shapeBufSize;++j)
            mArray[i*shapeBufSize+j]=m[j];
    }
}
RefTensor::~RefTensor()
{
    if(mArray){
#if DEBUG
    std::cout<<"RefTensor Free @"<<static_cast<void*>(mArray)<<std::endl;
#endif
        delete[] mArray;
    }
    shape.clear();
}
void RefTensor::operator=(RefTensor &&obj){
    this->~RefTensor();
    mArray=obj.mArray;
    shape=std::move(obj.shape);
    obj.mArray=nullptr;
}
void RefTensor::operator=(const Tensor &obj){
    if(shape!=obj.shape)
        throw std::runtime_error("From RefTensor::operator=(const Tensor &):\n\tWrong Shape!");
    Number *oP=obj.mArray;
    auto size=shape.bufSize();
    for(size_t i=0;i<size;++i)
        *(mArray[i])=oP[i];
}
Tensor RefTensor::asTensor() const {
    std::vector<Number> vec;
    auto en=mArray+shape.bufSize();
    for(auto p=mArray;p!=en;++p)
        vec.push_back(**p);
    Tensor res(vec);
    res.reshape(shape);
    return res;
}
RefTensor::operator Tensor(){
    return asTensor();
}
std::ostream &operator<<(std::ostream &osm, const RefTensor &obj){
    printShaped<RefTensor::NumberPtr>(
        obj.mArray,0,obj.shape,osm,
        [](std::ostream &osm, auto &ele){
            osm<<(*ele);
        }
    );
    return osm<<"\nshape: "<<obj.shape;
}
}