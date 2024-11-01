#include "shape.hpp"
namespace numcpp{

Shape::Shape(std::initializer_list<size_t> shape_)
:shape(shape_.begin(),shape_.size()),product(shape_.size()+1)
{
    const auto size=shape_.size();
    product[size]=1;
    generateProduct();
}
void Shape::generateProduct(){
    const auto shapeSize=shape.size();
    for(auto i=shapeSize; i>0; --i)
        product[i-1]=shape[i-1]*product[i];
}
size_t Shape::stepSizeOf(size_t dim) const {
    if(!toBoundedIndex(dim, shape.size(), &dim)) //这里必须检查是[0, shapeSize)之间的，然后再+1
        throw Error::outOfRange(__FILE__, __func__, dim, 0, shape.size());
    return product[dim+1];
}
Shape Shape::sliced(size_t be, size_t en) const{
    const auto size=shape.size();
    if(!toBoundedIndex(be,size,&be))
        throw Error::outOfRange(__FILE__,__func__, be, 0, size);
    if(!toBoundedIndex(en-1,size,&en));
        throw Error::outOfRange(__FILE__,__func__, en, 1, size+1);
    if(be==en) return Shape({});
    if(be>en)
        throw Error::wrong(__FILE__, __func__, "Wrong begin end");
    Shape res;
    res.shape=SizeTArray(shape.data()+be, en-be);
    res.generateProduct();
    return res;
}
Shape Shape::sliced(size_t be) const{
    const auto size=shape.size();
    if(be==size) return Shape({});
    if(!toBoundedIndex(be,size,&be))
        throw Error::outOfRange(__FILE__,__func__,be, 0, size);
    return Shape(
        SizeTArray(shape.data()+be, size-be), 
        SizeTArray(product.data()+be, size-be+1)
    );
}

std::pair<size_t,size_t> Shape::offsetOf(const SizeTArray &index) const{
    const auto argCount=index.size();
    size_t be=0;
    for(size_t i=0,dimSize,idx; i<argCount; ++i){
        dimSize = dimSizeOf(i);
        if(!toBoundedIndex(index[i],dimSize,&idx))
            throw Error::outOfRange(__FILE__,__func__,index[i], 0, dimSize);
        be+=idx*stepSizeOf(i);
    }
    auto resCnt=stepSizeOf(argCount-1);
    return {be,resCnt};
}

void Shape::squeeze(size_t dim){
    const auto oldDimNumber = dimNumber();
    if(!toBoundedIndex(dim, oldDimNumber, &dim))
        throw Error::outOfRange(__FILE__,__func__, dim, 0, oldDimNumber);
    if(dimSizeOf(dim)!=1)
        throw Error::wrong(__FILE__,__func__,"dimSizeOf(dim) is not 1");
    const auto oldShapeBegin = shape.data();
    SizeTArray newShapeVec(oldDimNumber-1); //仅去掉一个维
    const auto newShapeBegin  = newShapeVec.data();
    if(dim)
        memcpy(newShapeBegin, oldShapeBegin, dim*sizeof(size_t));
    if(oldDimNumber-dim-1)
        memcpy(newShapeBegin+dim, oldShapeBegin+dim+1, (oldDimNumber-dim-1)*sizeof(size_t));
    shape = std::move(newShapeVec);
    //对product类似操作
    const auto oldProductBegin = product.data();
    SizeTArray newShapeProduct(oldDimNumber);
    const auto newProductBegin = newShapeProduct.data();
    memcpy(newProductBegin, oldProductBegin, (dim+1)*sizeof(size_t));
    if(oldDimNumber-dim-1)
        memcpy(newProductBegin+dim+1, oldProductBegin+dim+2, (oldDimNumber-dim-1)*sizeof(size_t));
    product = std::move(newShapeProduct);
}
void Shape::squeeze(){ //TODO: 可以对比一下，是开俩数组memcpy导一遍快，还是循环两遍快
    const auto oldDimNumber = dimNumber();
    if(!oldDimNumber)
        return;
    size_t cnt1=0;
    for(size_t i = 0; i<oldDimNumber; ++i){
        if(shape[i]==1)
            ++cnt1;
    }
    if(!cnt1) //没有1
        return; 
    SizeTArray newShape(oldDimNumber-cnt1);
    SizeTArray newProduct(oldDimNumber-cnt1+1);
    newProduct[0]=product[0];
    size_t idx=0;
    for(size_t i=0; i<oldDimNumber; ++i){
        if(shape[i]==1) continue;
        newShape[idx]=shape[i];
        newProduct[++idx]=product[i+1];
    }
    shape = std::move(newShape);
    product = std::move(newProduct);
}

bool Shape::operator==(const Shape &shape_) const{
    if(&shape_==this) 
        return true;
    if(product.size()!=shape_.product.size())//不用shape.size()的原因：防止空shape误判
        return false;
    auto shapeSize=shape_.shape.size();
    int tmp=memcmp(shape.data(), shape_.shape.data(), shapeSize*shape.sizeOfT);
    return !tmp;
}
Shape Shape::operator+(const Shape &shape_) const {
    const size_t s1 = shape.size();
    const size_t s2 = shape_.shape.size();
    Shape res{SizeTArray(s1+s2), SizeTArray(s1+s2+1)};
    memcpy(res.shape.data(), shape.data(), s1*SizeTArray::sizeOfT);
    memcpy(res.shape.data()+s1, shape_.shape.data(), s2*SizeTArray::sizeOfT);
    memcpy(res.product.data(), product.data(), s1*SizeTArray::sizeOfT);
    memcpy(res.product.data()+s1, shape_.product.data(), (s2+1)*SizeTArray::sizeOfT);
    const size_t s2len=shape_.bufSize();
    for(size_t i=0;i<s1;++i)
        res.product[i]*=s2len;
    return res;
}
std::ostream &operator<<(std::ostream &osm, const Shape &obj){
    if(obj.empty())
        return osm<<"(empty)";
    osm<<'(';
    auto s=obj.shape.size();
    for(size_t i=0;i<s;++i){
        osm<<obj.shape[i];
        if(i+1!=s) osm<<',';
    }
    return osm<<')';
}
Shape Shape::broadcast(const Shape &shape1, const Shape &shape2){
    const Shape &longer=(shape1.dimNumber()>shape2.dimNumber()?shape1:shape2);
    const Shape &shorter=*((&shape1)-(&longer)+(&shape2));
    const auto shorterDimNumber=shorter.dimNumber();
    if(
        shorterDimNumber<=longer.dimNumber() &&
        shorter.bufSize()==1
    ) return longer;
    auto longerVec=longer.shape;
    const auto &shorterVec=shorter.shape;
    bool changed=false;
    //计算每一维度的值
    for(size_t i=0;i<shorterDimNumber;++i){
        auto &l=longerVec[-i-1];
        auto &s=shorterVec[-i-1];
        if(l!=s){
            if(l==1){
                changed=true;//需要更改
                l=s;
            }
            else if(s!=1)//既不相等也没1
                throw Error::wrong(__FILE__,__func__,"These two shape can't be broadcast");
        }
    }
    if(!changed) 
        return longer;
    //计算product并拷贝
    const auto resSize=longer.dimNumber();
    SizeTArray productVec(resSize+1);
    productVec[resSize]=1;
    for(size_t i=resSize;i>0;--i)
        productVec[i-1]=productVec[i]*longerVec[i-1];
    return Shape(std::move(longerVec),std::move(productVec));
}

Shape Shape::broadcast(const std::vector<Shape> &shapes)
{
    if(shapes.empty())
        throw Error::wrong(__FILE__,__func__,"<shapes> is empty");
    Shape res = shapes[0];
    size_t l = shapes.size();
    for(size_t i=1;i<l;++i)
        res = broadcast(res, shapes[i]);
    return res;
}

size_t Shape::offsetBeforeBroadcast(size_t broOffset, const Shape &broShape, const Shape &srcShape)
{
    // offset = (idx) cdot (shape.stepSizes)  
    size_t broDimNumber = broShape.dimNumber();
    size_t srcDimNumber = srcShape.dimNumber();
    if(broDimNumber < srcDimNumber)
        throw Error::wrong(__FILE__,__func__,"len(<broShape>)="+Error::ullToStr(broDimNumber)+"<len(<srcShape>)="+Error::ullToStr(srcDimNumber));
    //与srcShape无关的部分可以直接不要
    size_t dimDelta = broDimNumber - srcDimNumber;
    // for(size_t i=0;i<dimDelta;++i)
    //     broOffset%=broShape.stepSizeOf(i);
    if(dimDelta)
        broOffset %= broShape.stepSizeOf(dimDelta-1);
    //破译出每维坐标, 并直接计算偏移值
    size_t srcOffset=0;
    for(size_t i=0, srcIdx, broIdx;i<srcDimNumber;++i)
    {
        broIdx = broOffset/broShape.stepSizeOf(i+dimDelta);
        if(broIdx >= broShape.dimSizeOf(i+dimDelta))
            throw Error::author(__FILE__,__func__,"<broIdx> out of range");
        srcIdx = (srcShape.dimSizeOf(i)==1?0:broIdx);
        srcOffset += srcIdx * srcShape.stepSizeOf(i);
        broOffset %= broShape.stepSizeOf(i+dimDelta);
    }
    return srcOffset;
}

}