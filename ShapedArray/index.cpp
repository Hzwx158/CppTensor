#include "./index.hpp"
namespace tryAI{
Slice::Slice(size_t begin_, size_t end_, size_t stepSize_)
: begin(begin_), stepSize(stepSize_), end(end_){}
std::vector<size_t> Slice::toIndices(size_t length)
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

    std::vector<size_t> res(resLen);
    size_t k=0;
    for(size_t i=B;i!=E;i+=stepSize)
        res[k++]=i;
    return res;
}
}