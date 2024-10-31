#include "./list.hpp"
namespace numcpp{

size_t *toBoundedIndex(size_t idx, size_t bufSize, size_t *res){
    if(!res)
        throw Error::wrong(__FILE__, __func__, "<res> is nullptr");
    if(idx<bufSize){
        *res=idx;
        return res;
    }
    idx+=bufSize;
    if(idx<bufSize){
        *res=idx;
        return res;
    }
    return nullptr;
}

FixedArray<size_t> Slice::getIndices(size_t len) const{
    size_t b,e;
    if(!toBoundedIndex(start_, len, &b))
        throw Error::outOfRange(__FILE__, __func__, start_, 0, len);
    if(end_ == constant::int64_inf)
        e = len;
    else{
        if(!toBoundedIndex(end_-1, len, &e))
            throw Error::outOfRange(__FILE__, __func__, end_, 1, len+1);
        else ++e;
    }
    if(e==b) return {};
    assert((e-b)*step_!=0);
    size_t res_len; //ceil((en-be)/step)
    if(b<e && step_<0)
        res_len = numcpp::ceil(b+1, -step_);
    else if(b>e && step_>0)
        res_len = numcpp::ceil(len-b, step_);
    else res_len = numcpp::ceil((long long)(e-b), step_);
    FixedArray<size_t> res (res_len);
    for(size_t i=b, k=0; i!=e && i>=0 && i<len; i+=step_, k++)
        res[k] = i;
    return res;
}
std::ostream &operator<<(std::ostream &osm, const Slice &obj){
    static auto output = [&osm](long long num){
        if(num==constant::int64_inf)
            osm<<"inf";
        else if(num== constant::int64_ninf)
            osm<<"-inf";
        else osm<<num;
    };
    osm << "Slice(";
    output(obj.start_);
    osm <<',';
    output(obj.end_);
    osm <<',';
    output(obj.step_);
    return osm << ')';
}
}
