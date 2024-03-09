#include "shape.hpp"
namespace tryAI{
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
    if(!toBoundedIdx(dim, shape.size(), &dim)) //这里必须检查是[0, shapeSize)之间的，然后再+1
        throw std::out_of_range("From Shape::stepSizeOf(size_t):\n\tOut of range");
    return product[dim+1];
}
Shape Shape::sliced(size_t be, size_t en) const{
    bool legal1,legal2;
    const auto size=shape.size();
    legal1=toBoundedIdx(be,size,&be);
    legal2=toBoundedIdx(en-1,size,&en);
    if(!(legal1&&legal2))
        throw std::out_of_range("From Shape::sliced(size_t,size_t):\n\tOut of range");
    if(be==en) return Shape({});
    if(be>en)
        throw std::runtime_error("From Shape::sliced(size_t,size_t):\n\tWrong begin end");
    Shape res;
    res.shape=Vector(shape.data()+be, en-be);
    res.generateProduct();
    return res;
}
Shape Shape::sliced(size_t be) const{
    bool legal;
    const auto size=shape.size();
    if(be==size) return Shape({});;
    legal=toBoundedIdx(be,size,&be);
    if(!legal)
        throw std::out_of_range("From Shape::sliced(size_t):\n\tOut of range");
    return Shape(
        Vector(shape.data()+be, size-be), 
        Vector(product.data()+be, size-be+1)
    );
}
bool Shape::operator==(const Shape &shape_) const{
    if(&shape_==this) 
        return true;
    if(product.size()!=shape_.product.size())//不用shape.size()的原因：防止空shape误判
        return false;
    auto shapeSize=shape_.shape.size();
    for(size_t i=0;i<shapeSize;++i)
        if(shape[i]!=shape_.shape[i])
            return false;
    return true;
}
std::ostream &operator<<(std::ostream &osm, const Shape &obj){
    if(obj.isEmpty())
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
                throw std::runtime_error("From Shape::broadcast:\n\tThese two shape can't be broadcast");
        }
    }
    if(!changed) 
        return longer;
    //计算product并拷贝
    const auto resSize=longer.dimNumber();
    Vector productVec(resSize+1);
    productVec[resSize]=1;
    for(size_t i=resSize;i>0;--i)
        productVec[i-1]=productVec[i]*longerVec[i-1];
    return Shape(std::move(longerVec),std::move(productVec));
}
}