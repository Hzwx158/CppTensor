#ifndef NUMCPP_UTILS_LIST_HPP
#define NUMCPP_UTILS_LIST_HPP
#include "./base.h"
#include "./errors.h"
#include <functional>
#include <cstring>
namespace numcpp{

template<class Derived>
struct IIterator{
    virtual Derived &operator++()=0;
    virtual Derived operator++(int)=0;
    virtual Derived next() const=0;
};

/**
 * @brief 判断一个数是否是合理下标、并转成[0, bufSize)的下标值
 * @param idx 输入下标值
 * @param bufSize 数组长度
 * @param res 转成合理下标值的存储位置
 * @return 如果不合理，返回nullptr；合理返回res
 */
size_t *toBoundedIndex(size_t idx, size_t bufSize, size_t *res);

/**
 * @brief The FixedArray class
 * @attention copied by memcpy, not assigning.
 */
template<class T>
class FixedArray{
private:
    T *mArray;
    size_t len;
    
public:
    static const size_t sizeOfT = sizeof(T);
    constexpr FixedArray():mArray(nullptr), len(0ull){}
    FixedArray(std::initializer_list<T> inits)
        : mArray(inits.size()?(T*)malloc(inits.size()*sizeOfT):nullptr)
        , len(inits.size())
    {
        if(!len) return;
        memcpy(mArray, inits.begin(), len*sizeOfT);
    }
    FixedArray(const T *initVals, size_t len_)
        :mArray(len_?(T*)malloc(len_*sizeOfT):nullptr), len(len_)
    {
        if(!initVals) memset(mArray, 0, len*sizeOfT);
        else memcpy(mArray, initVals, sizeOfT*len);
    }
    template<size_t N>
    FixedArray(const T (&arr)[N])
        :mArray(N?(T*)malloc(N*sizeOfT):nullptr), len(N)
    {
        if(!N) memset(mArray, 0, N*sizeOfT);
        else memcpy(mArray, arr, sizeOfT*N);
    }
    FixedArray(size_t len_):mArray(len_?(T*)malloc(len_*sizeOfT):nullptr), len(len_){}
    FixedArray(size_t len_, const T &initVal)
        :mArray(len_?(T*)malloc(len_*sizeOfT):nullptr), len(len_)
    {
        for(size_t i=0;i<len;++i)
            mArray[i]=initVal;
    }
    FixedArray(const FixedArray &array):mArray(nullptr){
        if(this == &array) return;
//        clear();
        len = array.len;
        if(!array.mArray) return;
        mArray = (T*)malloc(sizeOfT*len);
        memcpy(mArray, array.mArray, sizeOfT*len);
    }
    FixedArray(FixedArray &&array):mArray(nullptr),len(0){
        if(this == &array) return;
//        clear();
        if(!array.mArray) return;
        len = array.len;
        mArray = array.mArray;
        array.len=0;
        array.mArray=nullptr;
    }
    void clear(){
        if(mArray){
            free(mArray);
            mArray = nullptr;
        }
        len=0;
    }
    ~FixedArray(){clear();}
    FixedArray &operator=(const FixedArray &array){
        if(this == &array) return *this;
        clear();
        if(!array.mArray) return *this;
        len = array.len;
        mArray = (T*)malloc(sizeOfT*len);
        memcpy(mArray, array.mArray, sizeOfT*len);
        return *this;
    }
    FixedArray &operator=(FixedArray &&array){
        if(this == &array) return *this;
        clear();
        if(!array.mArray) return *this;
        len = array.len;
        mArray = array.mArray;
        array.len=0;
        array.mArray=nullptr;
        return *this;
    }
    T &operator[](size_t idx) const {return at(idx);}
    T *data() const {return mArray;}
    T &at(size_t idx) const{
        if(!toBoundedIndex(idx, len, &idx))
            throw Error::outOfRange(__FILE__,__func__, idx, 0, len);
        return mArray[idx];
    }
    size_t size() const {return len;}
    bool empty() const {return !len;}
    FixedArray<T> operator+(const FixedArray<T> &ano) const{
        FixedArray<T> res(this->len + ano.len);
        memcpy(res.mArray, mArray, sizeOfT*len);
        memcpy(res.mArray+len, ano.mArray, sizeOfT*ano.len);
        return res;
    }
    bool contains(const T &ele, std::function<int(const T &, const T &)> cmp=nullptr) const{
        if(cmp==nullptr){
            for(auto &obj:*this){
                if(ele==obj)
                    return true;
            }
        }
        else{
            for(auto &obj:*this){
                if(!cmp(ele, obj))
                    return true;
            }
        }
        return false;
    }
    class iterator;
    class reverse_iterator;
    iterator begin() const{return mArray;}
    iterator end() const {return mArray+len;}
    reverse_iterator rbegin() const {return mArray+len-1;}
    reverse_iterator rend() const {return mArray-1;}
};

template<class T>
class FixedArray<T>::iterator:public IIterator<FixedArray<T>::iterator>{
private:
    using Self = FixedArray<T>::iterator;
    using Super = IIterator<Self>;
    T *ptr;
public:
    constexpr iterator():ptr(nullptr){}
    iterator(T *ptr):Super(), ptr(ptr){}
    iterator(const Self &obj):Super(), ptr(obj.ptr){}
    Self &operator++(){
        ptr++;
        return *this;
    }
    Self operator++(int){
        Self res(ptr++);
        return res;
    }
    Self next() const{
        return Self(ptr+1);
    }
    operator T*() const{return ptr;}
};

template<class T>
class FixedArray<T>::reverse_iterator:public IIterator<FixedArray<T>::reverse_iterator>{
private:
    using Self = FixedArray<T>::reverse_iterator;
    using Super = IIterator<Self>;
    T *ptr;
public:
    constexpr reverse_iterator():ptr(nullptr){}
    reverse_iterator(T *ptr):Super(), ptr(ptr){}
    reverse_iterator(const Self &obj):Super(), ptr(obj.ptr){}
    Self &operator++(){
        ptr--;
        return *this;
    }
    Self operator++(int){
        Self res(ptr--);
        return res;
    }
    Self next() const{
        return Self(ptr-1);
    }
    operator T*() const{return ptr;}
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
