#include "./tensor.hpp"
#include <cstring>
#include <cmath>
namespace tryAI{

//-----------------------RefTensor---------------------

RefTensor::RefTensor(const PtrVector<Number> &init_list)
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
    //std::cout<<"shape:"<<shape<<std::endl;
#endif
    memcpy(mArray, init_list.data(), size*init_list.sizeOfT);
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
    //检查shape一致性  
    if(argSize>1){
        //浪费时间，但不会反复new delete
        for(auto iter=refTensors.begin()+1;iter!=refTensors.end();++iter)
            if(be.shape!=iter->shape)
                throw std::runtime_error("From RefTensor(const vector<Ref> &):\n\tWrong shape");
    }
    //计算新shape
    const auto shapeDimNumber=be.shape.dimNumber();//shape原本维度数
    const auto shapeBufSize=be.shape.bufSize();//每个RefTensor的占内存数
    shape=Shape(
        Shape::Vector(shapeDimNumber+1),
        Shape::Vector(shapeDimNumber+2)
    );//先开出来这些存储空间
    auto sbegin=shape.getShapeData();
    auto pbegin=shape.getProductData();
    memcpy(sbegin+1, be.shape.getShapeData(),(shapeDimNumber)*Shape::Vector::sizeOfT);
    *sbegin=argSize;//把旧shape拷贝到+1处，再添一个
    memcpy(pbegin+1, be.shape.getProductData(), (shapeDimNumber+1)*Shape::Vector::sizeOfT);
    *pbegin=argSize*pbegin[1];
    //分配内存并拷贝
    mArray=new NumberPtr[shapeBufSize*argSize];
#if DEBUG
    std::cout<<"RefTensor Alloc @"<<static_cast<void*>(mArray)<<'['<<shapeBufSize*argSize<<']'<<std::endl;
#endif
    for(size_t i=0;i<argSize;++i)
        memcpy(
            mArray+i*shapeBufSize, 
            refTensors[i].mArray, 
            shapeBufSize*sizeof(NumberPtr)
        );
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
    //Tensor obj1=Tensor::broadcast(obj, shape);
    if(obj.shape.bufSize()!=shape.bufSize())
        throw std::runtime_error("@Author: When not broadcast, it's wrong shape from RefTensor::operator=");
    Number *oP=obj.mArray;
    auto size=shape.bufSize();
    for(size_t i=0;i<size;++i)
        *(mArray[i])=oP[i];
}
Tensor RefTensor::asTensor() const {
    const auto bufSize=shape.bufSize();
    UniquePointer<Number> ptr=new Number[bufSize];
#if DEBUG
    std::cout<<"Tensor Alloc @"<<static_cast<void*>(ptr)<<'['<<bufSize<<']'<<std::endl;
#endif
    for(size_t i=0;i<bufSize;++i)
        ptr[i]=**(mArray+i);
    return Tensor(std::move(ptr), shape);
}
RefTensor::operator Tensor(){
    return asTensor();
}
std::ostream &operator<<(std::ostream &osm, const RefTensor &obj){
    printShaped<RefTensor::NumberPtr>(
        obj.mArray,obj.shape,osm,
        [](std::ostream &osm, auto &ele){
            osm<<(*ele);
        }
    );
    return osm<<"\nshape: "<<obj.shape;
}

//--------------------------------内存---------------------------


Tensor::Tensor(std::initializer_list<Number> init_list, const Shape &shape_)
:mArray(init_list.size()? new Number[init_list.size()]:nullptr)
,shape(shape_.isEmpty()?Shape({init_list.size()}):shape_)
{
#if DEBUG
    std::cout<<"Tensor Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    if(init_list.size()!=shape.bufSize())
        throw std::runtime_error("From Tensor::Tensor({},const Shape &):\n\tShape unsame!");    
    if(init_list.size())
        memcpy(mArray, init_list.begin(), init_list.size()*sizeof(Number));
}
Tensor::Tensor(Number num, const Shape &shape_)
:mArray(shape_.isEmpty()? nullptr : new Number[shape_.bufSize()])
,shape(shape_){
    if(shape.isEmpty())
        throw std::runtime_error("From Tensor::Tensor(Number, const Shape &):\n\tEmpty shape!");
#if DEBUG
    std::cout<<"Tensor Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    const auto size=shape.bufSize();
    for(size_t i=0;i<size;++i)
        mArray[i]=num;
}
Tensor::Tensor(const Tensor &obj)
:mArray(new Number[obj.shape.bufSize()])
,shape(obj.shape)
{
#if DEBUG
    std::cout<<"Tensor Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
    if(shape.isEmpty()) 
        throw std::runtime_error("@Author from Tensor::Tensor(const Tensor &):\n\tshape can't be empty");
    auto size=shape.bufSize();
    memcpy(mArray, obj.mArray, sizeof(Number)*size);
}
Tensor::Tensor(Tensor &&obj)
:mArray(std::move(obj.mArray)),shape(std::move(obj.shape))
{
    //obj.clear();
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
    memcpy(mArray, obj.mArray, shape.bufSize()*sizeof(Number));
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
    //obj.clear();
    return *this;
}

//-------------------------功能-----------------------------

std::ostream &operator<<(std::ostream &osm, const Tensor &obj){
    printShaped<Tensor::Number>(obj.mArray,obj.shape,osm);
    return osm<<"\nshape: "<<obj.shape;
}
void Tensor::reshape(const Shape &shape_){
    if(shape.bufSize()!=shape_.bufSize())
        throw std::runtime_error("From Tensor::reshape(const Shape &):\n\tWrong Size");
    shape=shape_;
}

void Tensor::foreach(std::function<void(Number&)> func){
    const auto size=shape.bufSize();
    Number *head=mArray;
    for(size_t i=0;i<size;++i)
        func(head[i]);
}
RefTensor Tensor::at(std::function<bool(const Number &)> cond) const{
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

//--------------------------运算符----------------------------------
#define ASMD_OPERATOR_DEFINE(operatorName)\
Tensor &Tensor::operator operatorName##= (const Tensor &obj){\
    *this=broadcast(*this, obj.shape);\
    Tensor tmp=broadcast(obj, shape);\
    const auto bufSize=shape.bufSize();\
    for(size_t i=0;i<bufSize;++i)\
        mArray[i] operatorName##= tmp.mArray[i];\
    return *this;\
}\
Tensor Tensor::operator operatorName(const Tensor &obj) const{\
    Tensor res=broadcast(*this, obj.shape);\
    Tensor tmp=broadcast(obj, res.shape);\
    auto bufSize=res.shape.bufSize();\
    for(size_t i=0;i<bufSize;++i)\
        res.mArray[i] operatorName##= tmp.mArray[i];\
    return res;\
}
ASMD_OPERATOR_DEFINE(+)
ASMD_OPERATOR_DEFINE(-)
ASMD_OPERATOR_DEFINE(*)
ASMD_OPERATOR_DEFINE(/)
#undef ASMD_OPERATR_DEFINE

//res是obj的拷贝构造，mArray是res的内存条，bufSize是mArray的长度，i是遍历变量
#define TensorForEachFunc \
Tensor res(obj);\
Tensor::Number *mArray=res.mArray;\
size_t bufSize=res.shape.bufSize();\
for(size_t i=0;i<bufSize;++i)

Tensor exp(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=::exp(mArray[i]);
    }
    return res;
}
Tensor log(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=::log(mArray[i]);
    }
    return res;
}
Tensor sigmoid(const Tensor &obj){
    double x;
    TensorForEachFunc{
        x=::exp(mArray[i]);
        mArray[i]=x/(1+x);
    }
    return res;
}
Tensor sin(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=::sin(mArray[i]);
    }
    return res;
}
#undef TensorForEachFunc
//-----------------------广播---------------------

Tensor Tensor::broadcast(const Tensor &src, const Shape &shape){
    //1.形状需要非空
    const auto &srcShape=src.shape;
    if(srcShape.isEmpty()) 
        throw std::runtime_error("@Author: From Tensor::broadcast:\n\t<srcShape> is empty!");
    if(shape.isEmpty())
        throw std::runtime_error("@Author: From Tensor::broadcast:\n\t<shape> is empty!");
    const auto srcBufSize=srcShape.bufSize();
    const auto srcDimNumber=srcShape.dimNumber();
    auto resShape=Shape::broadcast(srcShape, shape);
    const auto resBufSize=resShape.bufSize();
    //不管怎么样，先拷贝到dest里总是必要的
    UniquePointer<Number> dest=new Number[resBufSize];
#if DEBUG
    std::cout<<"Tensor Alloc @"<<static_cast<void*>(dest)<<'['<<resBufSize<<']'<<std::endl;
#endif
    memcpy(dest, src.mArray, srcBufSize*sizeof(Number));
    //2.扩张结果显示和srcShape长度一致，只需直接复制、返回resShape即可
    if(resBufSize==srcBufSize)
        return Tensor(std::move(dest), std::move(resShape));
    //3.扩展结果与srcShape不一致，按规则扩张到resShape形状上
    const auto resDimNumber=resShape.dimNumber();
    const auto resProduct=resShape.getProductData();
    for(size_t rDim=0, srcCnt, copyCnt, srcStepSize, resStepSize, i, j, k;
        rDim<resDimNumber;++rDim
    ){//对每一维度(倒着)
        //计算需要复制几份(copyCnt)
        if(rDim<srcDimNumber)
            copyCnt=resShape.dimSizeOf(resDimNumber-1-rDim)/srcShape.dimSizeOf(srcDimNumber-1-rDim);
        else copyCnt=resShape.dimSizeOf(resDimNumber-1-rDim);
        if(copyCnt==1) //无需复制
            continue;
        //需要复制，说明srcDimSize是1, copyCnt==resShape.dimSizeOf(-rDim-1)
        //在原内存上的起点间隔(每个被复制的有几个元素), 没错就是resShape.stepSize[dim], 因为前面已经复制过了
        srcStepSize=resProduct[resDimNumber-rDim];//stepSizeOf(-rDim-1)
        //copyCnt=resStepSize/srcStepSize
        resStepSize=resProduct[resDimNumber-1-rDim]; //stepSizeOf(-rDim-2)，可能有[0]故不用stepSize函数
        //有几个需要被拷贝
        if(rDim<srcDimNumber)
            srcCnt=srcBufSize/srcShape.stepSizeOf(srcDimNumber-rDim-1);
        else srcCnt=1;
        for(i=0;i<srcCnt;++i){//倒着拷贝第i个
            for(j=(i+1==srcCnt);j<copyCnt;++j)//偏移正反无所谓
                memcpy(
                    dest+(srcCnt-i-1)*resStepSize+j*srcStepSize, 
                    dest+(srcCnt-i-1)*srcStepSize, 
                    srcStepSize*sizeof(Number)
                );
        }
    }
    return Tensor(std::move(dest),std::move(resShape));
}

//----------------------静态--------------------------

Tensor Tensor::arange(Number be, Number en, Number step, const Shape &shape){
    if(en==be)
        return Tensor();
    if((en-be)*step<=0)
        throw std::runtime_error("From Tensor::arange:\n\tWrong step");
    size_t cnt = static_cast<size_t>(abs((en-be)/step));
    if((!shape.isEmpty())&&(shape.bufSize()!=cnt))
        throw std::runtime_error("From Tensor::arange:\n\tWrong shape");
    if(!cnt)
        return Tensor();
    NumberPtr p=new Number[cnt];
#if DEBUG
    std::cout<<"Tensor Alloc @"<<static_cast<void*>(p)<<'['<<cnt<<']'<<std::endl;
#endif
    Number *h=p;
    for(;be<en;be+=step,++h)
        *h=be;
    if(shape.isEmpty())
        return Tensor(std::move(p), Shape{cnt});
    else return Tensor(std::move(p), shape);//调用构造函数不一样，别省这个if
}

}