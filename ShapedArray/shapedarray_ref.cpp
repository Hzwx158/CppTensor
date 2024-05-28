#include "./shapedarray.hpp"
namespace tryAI{
//-----------------------ShapedRefArray---------------------

ShapedRefArray::ShapedRefArray(const PtrVector<Number> &init_list)
:mArray(nullptr), shape({init_list.size()})
{
    auto size=init_list.size();
    if(!size) {
        shape.clear();
        return;
    }
    mArray=new NumberPtr[size];
#if DEBUG
    std::cout<<"ShapedRefArray Alloc @"<<static_cast<void*>(mArray)<<'['<<size<<']'<<std::endl;
    //std::cout<<"shape:"<<shape<<std::endl;
#endif
    memcpy(mArray, init_list.data(), size*init_list.sizeOfT);
}
ShapedRefArray::ShapedRefArray(ShapedRefArray &&obj)
:mArray(obj.mArray), shape(std::move(obj.shape))
{
    obj.mArray=nullptr;
}
ShapedRefArray::ShapedRefArray(const std::vector<ShapedRefArray> &refTensors)
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
                throw std::runtime_error("From ShapedRefArray(const vector<Ref> &):\n\tWrong shape");
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
    std::cout<<"ShapedRefArray Alloc @"<<static_cast<void*>(mArray)<<'['<<shapeBufSize*argSize<<']'<<std::endl;
#endif
    for(size_t i=0;i<argSize;++i)
        memcpy(
            mArray+i*shapeBufSize, 
            refTensors[i].mArray, 
            shapeBufSize*sizeof(NumberPtr)
        );
}
ShapedRefArray::~ShapedRefArray()
{
    if(mArray){
#if DEBUG
    std::cout<<"ShapedRefArray Free @"<<static_cast<void*>(mArray)<<std::endl;
#endif
        delete[] mArray;
    }
    shape.clear();
}
void ShapedRefArray::operator=(ShapedRefArray &&obj){
    this->~ShapedRefArray();
    mArray=obj.mArray;
    shape=std::move(obj.shape);
    obj.mArray=nullptr;
}
void ShapedRefArray::operator=(const ShapedArray &obj){
    Shape resShape = Shape::broadcast(shape, obj.shape);
    if(resShape!=shape)
        throw std::runtime_error("From ShapedRefArray::operator=:\n\t<obj.shape> can't broadcast to <shape>");
    Number *oP=obj.mArray;
    auto size=shape.bufSize();
    for(size_t i=0;i<size;++i)
    {
        *(mArray[i])=oP[Shape::offsetBeforeBroadcast(i, shape, obj.shape)];
    }
}
ShapedArray ShapedRefArray::asTensor() const {
    const auto bufSize=shape.bufSize();
    UniquePointer<Number> ptr=new Number[bufSize];
#if DEBUG
    std::cout<<"Pointer Alloc @"<<static_cast<void*>(ptr)<<'['<<bufSize<<']'<<std::endl;
#endif
    for(size_t i=0;i<bufSize;++i)
        ptr[i]=**(mArray+i);
    return ShapedArray(std::move(ptr), shape);
}
ShapedRefArray::operator ShapedArray(){
    return asTensor();
}
std::ostream &operator<<(std::ostream &osm, const ShapedRefArray &obj){
    printShaped<ShapedRefArray::NumberPtr>(
        obj.mArray,obj.shape,osm,
        [](std::ostream &osm, auto &ele){
            osm<<(*ele);
        }
    );
    return osm<<"\nshape: "<<obj.shape;
}

}