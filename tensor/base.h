#pragma once
#ifndef CPPTENSOR_TENSOR_BASE
#define CPPTENSOR_TENSOR_BASE
#include <iostream>
#include <stdexcept>
#include <vector>

namespace tryAI{
/**
 * @brief 只运行一次助手 
 * @param retval 多次运行时的返回值，可以不填
*/
#define H_RUN_ONCE(retval) \
static bool __runOnceHelper=false;\
if(__runOnceHelper) return retval;\
__runOnceHelper=true;

//可输出类标签
#define H_OUTPUTABLE(ClassName) \
friend std::ostream & operator<<(std::ostream &osm, const ClassName &obj);

#define useStdIO using std::cin;using std::cout;using std::endl;

#define TENSOR_RES_OP_DECL(ClassName, operatorName) ClassName &operator operatorName##= (const ClassName &obj);\
ClassName operator operatorName(const ClassName &obj) const {\
    ClassName res(*this);\
    return res operatorName##= obj;\
}

template<class T, class ReleaseFunc>
class Releaser{
private:
    const ReleaseFunc &release;
    T &obj;
public:
    Releaser(T &obj_, const ReleaseFunc &releaseFunc):
    obj(obj_), release(releaseFunc){}
    ~Releaser(){release(obj);}
};
template<class T, class R>
class Releaser<T, R(T::*)()>{
public:
    using TMemFunc=R(T::*)();
private:
    TMemFunc release;
    T &obj;
public:
    Releaser(T &obj_, TMemFunc releaseFunc):
    obj(obj_), release(releaseFunc){}
    ~Releaser(){
        if(release)
            (obj.*release)();
    }
};

inline void rprintBits(double d){
    size_t tmp=*(size_t*)(&d);
    const static int s=64;
    for(int i=0;i<s;++i){
        std::cout<<(tmp&1?1:0);
        tmp>>=1;
    }
    std::cout<<std::endl;
}
#endif

}