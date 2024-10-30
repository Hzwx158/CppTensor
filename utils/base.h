#ifndef NUMCPP_UTILS_BASE_H
#define NUMCPP_UTILS_BASE_H
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <cassert>

namespace numcpp{
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
friend std::ostream & operator<<(std::ostream &osm, const ClassName &obj)

#define useStdIO using std::cin;using std::cout;using std::endl;

#define TENSOR_RES_OP_DECL(ClassName, operatorName) ClassName &operator operatorName##= (const ClassName &obj);\
ClassName operator operatorName(const ClassName &obj) const {\
    ClassName res(*this);\
    return res operatorName##= obj;\
}

#define DEF_TYPE_JUDGE(className) \
template<class T>\
struct Is_##className{\
    static constexpr bool flag = false;\
};\
template<>\
struct Is_##className < className > {\
    static constexpr bool flag = true;\
};\
template<class T>\
constexpr bool is_##className##_v = Is_##className<T>::flag;

/**
 * @brief 上下文管理器
 * @tparam T 被管理对象类型
 * @tparam ReleaseFunc 管理函数类型
 */
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
/**
 * @brief 从低位到高位输出二进制内容
 * @param d 浮点数
 */
inline void rprintBits(double d){
    size_t tmp=*(size_t*)(&d);
    const static int s=64;
    for(int i=0;i<s;++i){
        std::cout<<(tmp&1?1:0);
        tmp>>=1;
    }
    std::cout<<std::endl;
}

#define SCD static constexpr double
#define SCL static constexpr long long
struct constant{
    SCD pi = 3.14159265358979323846;
    SCD e = 2.7182818284590452354;
    SCD log2e = 1.4426950408889634074;
    SCD log10e = 0.43429448190325182765;
    SCD ln2 = 0.69314718055994530942;
    SCD ln10 = 2.30258509299404568402;
    SCD pi_2 = 1.57079632679489661923;
    SCD pi_4 = 0.78539816339744830962;
    SCD _1_pi = 0.31830988618379067154;
    SCD _2_pi = 0.63661977236758134308;
    SCD sqrt_pi = 1.12837916709551257390;
    SCD sqrt2 = 1.41421356237309504880;
    SCD sqrt1_2 = 0.70710678118654752440;
    SCL INT_INF = LLONG_MAX;
    SCL INT_NINF = LLONG_MIN;
    SCL unsigned UINT_INF = ULLONG_MAX;
};
#undef SCD
#undef SCL

template<class T>
bool isZeros(const T &obj){
    const size_t size = sizeof(obj);
    printf("%llu\n",size);
    if(size<=1ull)
        return ! ((*(int8_t*)(&obj)) & static_cast<int8_t>(-1));
    else if(size<=2)
        return ! ((*(int16_t*)(&obj)) & static_cast<int16_t>(-1));
    else if(size<=4)
        return ! ((*(int32_t*)(&obj)) & static_cast<int32_t>(-1));
    return ! ((*(int64_t*)(&obj)) & static_cast<int64_t>(-1));
}

template<class T>
struct IsVector
{
    static constexpr bool flag=false;
    IsVector(const T &){}
};
template<class T>
struct IsVector<std::vector<T>>
{
    static constexpr bool flag=true;
    IsVector(const std::vector<T> &){}
};
template<class T>
bool isVector(const T &){return IsVector<T>::flag;}

inline bool any(std::initializer_list<bool> list)
{
    for(auto &val : list)
        if(val)
            return true;
    return false;
}
inline bool all(std::initializer_list<bool> list)
{
    for(auto &val : list)
        if(!val)
            return false;
    return true;
}

}
#endif