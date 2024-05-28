#pragma once
#include "./simplevector.hpp"
namespace tryAI{
class Slice{
private:
    size_t begin;
    size_t stepSize;
    size_t end;
public:
    Slice(size_t begin_=0, size_t end_=-1, size_t stepSize_=1);
    std::vector<size_t> toIndices(size_t length);
};

template<class T>
struct IsVectorSizeT
{
    static const bool flag=false;
    IsVectorSizeT(const T &){}
};
template<>
struct IsVectorSizeT<std::vector<size_t>>
{
    static const bool flag=true;
    IsVectorSizeT(const std::vector<size_t> &){}
};
}