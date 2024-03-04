#include "shape.hpp"
namespace tryAI{
size_t *toBoundedIdx(size_t idx, size_t bufSize, size_t *res){
    if(!res)
        throw std::runtime_error("From Shape.toBoundedIdx:\n\t<res> is nullptr");
    if(0<=idx && idx<bufSize){
        *res=idx;
        return res;
    }
    idx+=bufSize;
    if(0<=idx && idx<bufSize){
        *res=idx;
        return res;
    }
    return nullptr;
}
void Shape::generateProduct(){
    if(shape.empty()){
        product={1};
        return;
    }
    product=Vector(shape.size()+1, 1);
    const auto shapeSize=shape.size();
    for(auto i=shapeSize-1; i>0; --i)
        product[i-1]=shape[i]*product[i];
    product[shapeSize]=shape[0]*product[0];
}
size_t Shape::stepSizeOf(size_t dim) const {
    if(!toBoundedIdx(dim, shape.size(), &dim))
        throw std::out_of_range("From Shape::stepSizeOf(size_t):\n\tOut of range");
    return product[dim];
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
    ++en;
    for(auto p=be;p!=en;++p){
        res.shape.push_back(shape[p]);
        res.product.push_back(product[p]);
    }
    res.product.push_back(be?product[be-1]:product[shape.size()]);
    return res;
}
Shape Shape::sliced(size_t be) const{
    bool legal;
    const auto size=shape.size();
    if(be==size) return Shape({});;
    legal=toBoundedIdx(be,size,&be);
    if(!legal)
        throw std::out_of_range("From Shape::sliced(size_t):\n\tOut of range");
    Shape res;
    for(auto p=be;p!=size;++p){
        res.shape.push_back(shape[p]);
        res.product.push_back(product[p]);
    }
    res.product.push_back(be?product[be-1]:product[shape.size()]);
    return res;
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
        osm<<obj.shape.at(i);
        if(i+1!=s) osm<<',';
    }
    return osm<<')';
}
}