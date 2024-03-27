#include "./tensor.hpp"
#include <cstring>
namespace tryAI{


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