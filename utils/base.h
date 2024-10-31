#ifndef NUMCPP_UTILS_BASE_H
#define NUMCPP_UTILS_BASE_H
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cfloat>
#include <cstdint>
#include <cassert>

#define DEBUG 0
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

//使用stdio
#define useStdIO using std::cin;using std::cout;using std::endl;

//判断是否是某类型的代码生成模板
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
// 极大极小值助手
template<class T>
struct _INF{};
template<>
struct _INF<uint64_t>{
    static constexpr uint64_t inf = ULLONG_MAX;
};
template<>
struct _INF<int64_t>{
    static constexpr int64_t inf = LLONG_MAX;
    static constexpr int64_t ninf = LLONG_MIN;
};
template<>
struct _INF<uint32_t>{
    static constexpr uint32_t inf = UINT32_MAX;
};
template<>
struct _INF<int32_t>{
    static constexpr int32_t inf = INT32_MAX;
    static constexpr int32_t ninf = INT32_MIN;
};
template<>
struct _INF<uint16_t>{
    static constexpr uint16_t inf = UINT16_MAX;
};
template<>
struct _INF<int16_t>{
    static constexpr int16_t inf = INT16_MAX;
    static constexpr int16_t ninf = INT16_MIN;
};
template<>
struct _INF<uint8_t>{
    static constexpr uint8_t inf = UINT8_MAX;
};
template<>
struct _INF<int8_t>{
    static constexpr int8_t inf = INT8_MAX;
    static constexpr int8_t ninf = INT8_MIN;
};
template<>
struct _INF<float>{
    static constexpr float inf = FLT_MAX;
    static constexpr float ninf = FLT_MIN;
};
template<>
struct _INF<double>{
    static constexpr double inf = DBL_MAX;
    static constexpr double ninf = DBL_MIN;
};
template<class T>
static constexpr T inf_v = _INF<T>::inf;
template<class T>
static constexpr T ninf_v = _INF<T>::ninf;
#define SCD static constexpr double
#define SCL static constexpr long long
/**
 * @brief 常量集合
 */
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
    SCL int64_inf = inf_v<int64_t>;
    SCL int64_ninf = ninf_v<int64_t>;
    SCL unsigned uint64_inf = inf_v<uint64_t>;
    SCD eps = 1e-13;
    SCD float64_inf = inf_v<double>;
    SCD float64_ninf = ninf_v<double>;
};
#undef SCD
#undef SCL

/**
 * @brief 判断一个变量的二进制内容是否是0
 * @param obj 一个变量
 * @return 是则true
 */
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
enum class EOperation:uint8_t{
    ADD,
    SUB,
    MUL,
    DIV,
    BIT,
    MOL,
    LOGICAL,
};
template<EOperation op, class T1, class T2>
struct OpRetHelper{};
template<EOperation op, class T1, class T2>
using op_ret_t = typename OpRetHelper<op, T1, T2>::type;
template<class T1, class T2>
struct OpRetHelper<EOperation::ADD, T1, T2>{
    using type = decltype(std::declval<T1>()+std::declval<T2>());
};
template<class T1, class T2>
struct OpRetHelper<EOperation::SUB, T1, T2>{
    using type = decltype(std::declval<T1>()-std::declval<T2>());
};
template<class T1, class T2>
struct OpRetHelper<EOperation::MUL, T1, T2>{
    using type = decltype(std::declval<T1>()*std::declval<T2>());
};
template<class T1, class T2>
struct OpRetHelper<EOperation::DIV, T1, T2>{
    // using type = decltype(std::declval<T1>()/std::declval<T2>());
    using type = double;
};
template<class T1, class T2>
struct OpRetHelper<EOperation::MOL, T1, T2>{
    using type = decltype(std::declval<T1>()%std::declval<T2>());
};
template<class T1, class T2>
struct OpRetHelper<EOperation::BIT, T1, T2>{
    using type = decltype(std::declval<T1>()|std::declval<T2>());
};
template<class T1, class T2>
struct OpRetHelper<EOperation::LOGICAL, T1, T2>{
    using type = decltype(std::declval<T1>()||std::declval<T2>());
};


}
#endif