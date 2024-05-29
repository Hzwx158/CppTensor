#include "./index.hpp"
#include "./shapedarray.hpp"
namespace tryAI{
Slice::Slice(size_t begin_, size_t end_, size_t stepSize_)
: begin(begin_), stepSize(stepSize_), end(end_){}
ShapedArray Slice::toIndices(size_t length) const
{
    size_t B,E;
    if(!toBoundedIdx(begin, length, &B))
        throw std::out_of_range("From Slice::toIndices: <B> Out of range");
    if(!toBoundedIdx(end-1, length, &E)) //为什么end-1: 因为end本身是不含的。
        throw std::out_of_range("From Slice::toIndices: <E> Out of range");
    ++E; //把end恢复到不含的

    size_t resLen;
    if(B>E)
        resLen = (B-E)/(-stepSize);
    else resLen = (E-B)/stepSize;

    ShapedArray::Number *res = new ShapedArray::Number[resLen];
    size_t k=0;
    for(size_t i=B;i!=E;i+=stepSize)
        res[k++]=i;
    return ShapedArray(res, {resLen});
}


#define genIndexMacro(type) ShapedArray genIndex(type num, size_t){\
    return ShapedArray(static_cast<ShapedArray::Number>(num));\
}
genIndexMacro(size_t)
genIndexMacro(long long)
genIndexMacro(int)
genIndexMacro(unsigned)
genIndexMacro(unsigned short)
genIndexMacro(short)
ShapedArray genIndex(const ShapedArray &arr, size_t){return arr;}
ShapedArray genIndex(const Slice &slice, size_t len){return slice.toIndices(len);}
#undef genIndexMacro

std::ostream &operator<<(std::ostream &osm, const Slice &slc)
{
    return osm<<"Slice("
        <<static_cast<long long>(slc.begin)<<','
        <<static_cast<long long>(slc.end)<<','
        <<static_cast<long long>(slc.stepSize)<<')';
}

std::ostream &operator<<(std::ostream &osm, const Index &idx)
{
    if(idx.isNum)
        osm<<"Index("<<static_cast<long long>(idx.data.idx)<<')';
    else
        osm<<"Index("<<idx.data.slice<<')';
    return osm;
}

}