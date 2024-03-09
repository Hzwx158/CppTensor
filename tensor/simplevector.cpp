#include "./simplevector.hpp"
namespace tryAI
{
size_t *toBoundedIdx(size_t idx, size_t bufSize, size_t *res){
    if(!res)
        throw std::runtime_error("From toBoundedIdx:\n\t<res> is nullptr");
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

} // namespace tryAI
