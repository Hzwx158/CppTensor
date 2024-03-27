#include "./tensor.hpp"
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

}