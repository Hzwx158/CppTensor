#ifndef CPPTENSOR_SHAPEDARRAY_INDEX_H
#define CPPTENSOR_SHAPEDARRAY_INDEX_H
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


}
#endif