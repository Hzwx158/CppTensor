#ifndef NUMCPP_UTILS_LIST_HPP
#define NUMCPP_UTILS_LIST_HPP
#include "utils/base.h"
#include "utils/errors.h"
#include <functional>
#include <cstring>
#include <sstream>
namespace numcpp{


/**
 * @brief 判断一个数是否是合理下标、并转成[0, bufSize)的下标值
 * @param idx 输入下标值
 * @param bufSize 数组长度
 * @param res 转成合理下标值的存储位置
 * @return 如果不合理，返回nullptr；合理返回res
 */
size_t *toBoundedIndex(size_t idx, size_t bufSize, size_t *res);

/**
 * @brief 运行时定长数组；不可动态修改长度，只能在初始化、重新赋值时更改长度
 * @attention 尽量使用平凡类型；非平凡类型可能会导致析构函数调用次数比构造函数次数多，这是为了速度和内存少做出的牺牲
 */
template<class T>
class FixedArray{
private:
    T *mArray;
    size_t len;
    /**
     * @brief 复制操作。需要在分配好mArray和len之后调用。
     * @attention 仅负责填充内容，不负责清空旧有内容，也不负责安全检查
     * @param mbegin 用以复制的内存起点
     * @param l 用以复制的元素个数
     * @param initializing 为true，用拷贝复制；为false，用赋值
     */
    void _copy(T const *mbegin, size_t l, bool initializing){
        if(!l) return;
        if constexpr(std::is_trivial_v<T>)
            memcpy(mArray, mbegin, l*sizeOfT);
        else {
            if(initializing)
                for(size_t i=0;i<l;++i)
                    new (mArray+i) T(mbegin[i]);
            else for(size_t i=0;i<l;++i)
                mArray[i] = mbegin[i];
        }
    }
public:
    static constexpr size_t sizeOfT = sizeof(T);
    /**
     * @brief 默认构造函数
     */
    constexpr FixedArray():mArray(nullptr), len(0ull){}
    /**
     * @brief 通过初始化列表构造
     * @param inits 初始化列表
     */
    FixedArray(std::initializer_list<T> inits)
        : mArray(inits.size() ? (T*)(::operator new[](inits.size()*sizeOfT)) : nullptr)
        ,len(inits.size())
    {
        if(!len) return;
        _copy(inits.begin(), inits.size(), true);
    }
    /**
     * @brief 通过指针+长度，拷贝初始化
     * @param initVals 初始化数组的指针
     * @param len_ 初始化元素个数
     */
    FixedArray(const T *initVals, size_t len_)
        :mArray(len_?(T*)(::operator new[](len_*sizeOfT)):nullptr)
        ,len(len_)
    {
        if(!initVals){
            if(len) 
                throw Error::wrong(__FILE__, __func__, "Wrong arguments");
            return;
        }
        _copy(initVals, len, true);
    }
    /**
     * @brief 用数组进行初始化
     * @param arr 数组变量
     */
    template<size_t N>
    FixedArray(const T (&arr)[N])
        :mArray(N?(T*)(::operator new[](N*sizeOfT)):nullptr)
        ,len(N)
    {
        if(!N) return;
        _copy(arr, N, true);
    }
    /**
     * @brief 预留长度式初始化
     * @attention 只有在下面三点都满足时才会调用构造函数：
     * (1) construct_when_create 被设置为 true；
     * (2) T 是非平凡类型；
     * (3) T 可以默认构造。
     * 请谨慎使用该构造函数。这里未调用T的构造函数的话，会导致T的构造函数比析构函数调用次数更少。
     * @param len_ 预留长度
     * @param construct_when_create 是否调用构造函数；为保证效率，所有平凡类型保证一定在这时不会构造
     */
    FixedArray(size_t len_, bool construct_when_create = false)
        :mArray(len_?(T*)(::operator new[](len_*sizeOfT)):nullptr)
        ,len(len_)
    {
        if constexpr(
            (!std::is_trivial_v<T>) && 
            std::is_default_constructible_v<T>
        )
            if(construct_when_create)
                for(size_t i=0;i<len;++i)
                    new (mArray+i) T(); 
        // TODO: 我们究竟需不需要在这里初始化？不初始化会不会造成问题？
    }
    /**
     * @brief 填充式构造
     * @param len_ 元素个数
     * @param initVal 元素初始化值
     */
    FixedArray(size_t len_, const T &initVal)
        :mArray(len_?(T*)(::operator new[](len_*sizeOfT)):nullptr)
        ,len(len_)
    {
        for(size_t i=0;i<len;++i)
            new (mArray+i) T(initVal);
    }
    /**
     * @brief 拷贝构造
     * @param array 被拷贝对象
     */
    FixedArray(const FixedArray &array)
        :mArray(array.len?(T*)(::operator new[](array.len*sizeOfT)):nullptr)
        ,len(array.len)
    {
        if(!array.mArray) return;
        _copy(array.mArray, len, true);
    }
    /**
     * @brief 移动构造
     * @param array 被接管对象
     */
    FixedArray(FixedArray &&array) noexcept
        :mArray{array.mArray}
        ,len{array.len}
    {
        if(this!=&array){
            array.len=0;
            array.mArray=nullptr;
        }
    }
    /**
     * @brief 清空
     */
    void clear(){
        if(mArray){
            if constexpr(!std::is_trivial_v<T>){
                for(size_t i=0;i<len;++i)
                    mArray[i].~T();
            }
            ::operator delete[](mArray);
            mArray = nullptr;
        }
        len=0;
    }
    FixedArray<T> shrink_to(size_t l) const {
        FixedArray<T> res(l, false);
        res._copy(mArray, l, true);
        return res;
    }
    /**
     * @brief 析构函数
     */
    ~FixedArray(){clear();}
    /**
     * @brief 拷贝赋值函数
     * @param array 用于赋值的对象
     * @attention 只有长度一致时不会刷新内存。所以建议尽量不要过于频繁地调用该函数
     * @return *this
     */
    FixedArray &operator=(const FixedArray &array){
        if(this == &array) return *this;
        if(array.len==len)
            _copy(array.mArray, len, false);
        else {
            clear();
            if(!array.mArray)
                return *this;
            len = array.len;
            mArray = (T*)::operator new[](sizeOfT*len);
            _copy(array.mArray, len, true);
        }
        return *this;
    }
    /**
     * @brief 移动赋值函数
     * @param array 被接管对象
     * @return *this
     */
    FixedArray &operator=(FixedArray &&array) noexcept{
        if(this == &array) return *this;
        clear();
        len = array.len;
        mArray = array.mArray;
        array.len=0;
        array.mArray=nullptr;
        return *this;
    }
    /**
     * @brief 取下标函数
     * @param idx 下标
     * @return 一个引用
     */
    T &operator[](size_t idx) const {return at(idx);}
    /**
     * @brief 用于获取mArray
     * @return mArray
     */
    T *data() const {return mArray;}
    /**
     * @brief 取下标函数
     * @param idx 下标
     * @return 一个引用
     */
    T &at(size_t idx) const{
        if(!toBoundedIndex(idx, len, &idx))
            throw Error::outOfRange(__FILE__,__func__, idx, 0, len);
        return mArray[idx];
    }
    /**
     * @brief 获取长度
     * @return 数组长度
     */
    size_t size() const {return len;}
    /**
     * @brief 判断数组是否已空
     * @return 空则true
     */
    bool empty() const {return !len;}
    /**
     * @brief 连接两个数组
     * @param ano 另一个数组
     * @return 新数组
     */
    FixedArray<T> operator+(const FixedArray<T> &ano) const{
        FixedArray<T> res(this->len + ano.len);
        if(mArray)
            res._copy(mArray, len, true);
        if(ano.mArray){
            res.mArray += len;
            res._copy(ano.mArray, ano.len, true);
            res.mArray -= len;
        }
        return res;
    }
    /**
     * @brief 判断某元素是否在数组中
     * @param ele 元素
     * @param cmp 比较函数，默认是==
     * @return 在则true
     */
    template<class Comparator = std::not_equal_to<T>>
    bool contains(const T &ele, Comparator &&cmp={}) const{
        for(const auto &obj:*this)
            if(!cmp(ele, obj))
                return true;
        return false;
    }
    using iterator = T*;
    using reverse_iterator = std::reverse_iterator<T*>;
    /**
     * @brief 迭代器起点
     * @return 迭代器起点
     */
    iterator begin() const{return mArray;}
    /**
     * @brief 迭代器终点
     * @return 迭代器终点
     */
    iterator end() const {return mArray+len;}
    /**
     * @brief 反向迭代器起点
     * @return 倒数第一个
     */
    reverse_iterator rbegin() const {return reverse_iterator(end());}
    /**
     * @brief 反向迭代器终点
     * @return 倒数第
     */
    reverse_iterator rend() const {return reverse_iterator(begin());}
    H_OUTPUTABLE(FixedArray){
        std::stringstream oss;
        oss << "F{";
        for(size_t i=0; i<obj.len; ++i){
            oss << obj.mArray[i];
            if(i+1!=obj.len)
                oss << ',';
        }
        oss << '}';
        return osm << oss.str();
    }
};


/**
 * @brief 切片类
 */
struct Slice{
    using ll = long long;
    ll start_;
    ll end_;
    ll step_;
    /**
     * @brief 构造函数
     * @param end 结束(不含), 默认是无穷
     */
    constexpr explicit Slice(ll end=constant::int64_inf):start_{0}, end_{end}, step_{1}{}
    /**
     * @brief 构造函数
     * @param start 起始(含)
     * @param end 结束(不含)
     * @param step 步长，默认1
     */
    constexpr explicit Slice(ll start, ll end, ll step=1):start_{start}, end_{end}, step_{step}{
        if(!step)
            throw Error::wrong(__FILE__, __func__, "<step> should not be 0!");
    }
    /**
     * @brief 从某下标起，直到无穷
     * @param start 起始(含)
     * @param step 步长，默认1
     * @return Slice对象
     */
    static Slice from(ll start, ll step=1){
        return Slice(start, constant::int64_inf, step);
    }
    /**
     * @brief 直到某下标结束，同Slice(ll end, ll step=1)
     * @param end 终止(不含)
     * @param step 步长，默认1
     * @return Slice对象
     */
    static Slice to(ll end, ll step=1){
        return Slice(0, end, step);
    }
    /**
     * @brief 获取正确的坐标序列
     * @param len 数组大小
     * @return 一个坐标序列
     */
    FixedArray<size_t> getIndices(size_t len) const;
    H_OUTPUTABLE(Slice);
};
/**
 * @brief 计算ceil(a/b)
 * @param a 分子
 * @param b 分母
 * @return ceil(a/b)
 */
inline long long ceil(long long a, long long b){
    if(!b) throw Error::divByZero(__FILE__, __func__);
    double f = a/(b+0.0);
    long long d = a/b;
    bool isInt = (f-d)<=constant::eps;
    return isInt? d:d+1;
}
}
#endif
