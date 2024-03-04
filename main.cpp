#include "./tensor/tensor.hpp"
useStdIO
using tryAI::Tensor;
using tryAI::Shape;
using tryAI::list;


int main()
{
#if 1
    auto t=new Tensor({1,2,3,4,5,6,7,8,9,10,11,12},{2,2,3});
    t->at(0,1)={100,99,98};
    cout<<(*t)<<endl;
    delete t;
#else

#endif
    return 0;
}