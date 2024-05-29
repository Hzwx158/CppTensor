#ifndef CPPTENSOR_SHAPEDARRAY_SIMPLEVECTOR_H
#define CPPTENSOR_SHAPEDARRAY_SIMPLEVECTOR_H
#include "./base.h"
#include <cstring>
namespace tryAI{
/**
 * @brief 判断一个数是否是合理下标、并转成[0, bufSize)的下标值
 * @param idx 输入下标值
 * @param bufSize 数组长度
 * @param res 转成合理下标值的存储位置
 * @return 如果不合理，返回nullptr；合理返回res
 */
size_t *toBoundedIdx(size_t idx, size_t bufSize, size_t *res);
}
#define VECTOR_OPTIMIZE 1
#if VECTOR_OPTIMIZE
#define INIT_LIST_CONSTRUCTOR(ClassName, EleType, SaveType)\
ClassName(std::initializer_list<EleType> init_list)\
:mArray(init_list.size()?new SaveType[init_list.size()]:nullptr), __size(init_list.size())\
{\
    auto iter=init_list.begin();\
    for(size_t i=0;i<__size;++i,++iter)\
        mArray[i]=*iter;\
}
#define VECTOR_PARTIAL_SPE(ClassName, SaveClass) \
public:\
    using T=SaveClass;\
    static constexpr size_t sizeOfT=sizeof(T);\
private:\
    T *mArray;\
    size_t __size;\
    size_t __alloc;\
    void resize(size_t target, bool needCopy=true){\
        if(__alloc>=target) return;\
        while(__alloc<target){\
            __alloc<<=1;\
        }\
        auto tmp=mArray;\
        mArray=new T[__alloc];\
        if(needCopy)\
            memcpy(mArray, tmp, __size*sizeOfT);\
        delete[] tmp;\
    }\
public:\
    constexpr ClassName():mArray(nullptr),__size(0),__alloc(0){}\
    ClassName(size_t size_)\
        :mArray(size_?new T[size_]:nullptr)\
        ,__size(size_)\
        ,__alloc(size_)\
    {}\
    ClassName(size_t size_, const T &init)\
        :mArray(size_?new T[size_]:nullptr)\
        ,__size(size_)\
    {\
        if(!__size) return;\
        if((!init)||(!(init+1)))\
            memset(mArray, init, __size*sizeOfT);\
        else for(size_t i=0;i<__size;++i)\
            mArray[i]=init;\
    }\
    ClassName(const T *arr, size_t size_)\
        :mArray((arr&&size_)?new T[size_]:nullptr)\
        ,__size(arr?size_:0)\
        ,__alloc(arr?size_:0)\
    {\
        if(__size)\
            memcpy(mArray, arr, __size*sizeOfT);\
    }\
    ClassName(std::initializer_list<T> init_list)\
        :mArray(init_list.size()?new T[init_list.size()]:nullptr)\
        ,__size(init_list.size())\
        ,__alloc(init_list.size())\
    {\
        memcpy(mArray, init_list.begin(), __size*sizeOfT);\
    }\
    template<size_t N>\
    ClassName(T (&arr)[N])\
        :mArray(new T[N])\
        ,__size(N)\
        ,__alloc(N)\
    {\
        memcpy(mArray, arr, N*sizeOfT);\
    }\
    ClassName(const ClassName &obj)\
        :mArray(obj.__size?new T[obj.__size]:nullptr)\
        ,__size(obj.__size)\
        ,__alloc(obj.__size)\
    {\
        if(obj.__size)\
            memcpy(mArray, obj.mArray, __size*sizeOfT);\
    }\
    ClassName(ClassName &&obj)\
        :mArray(obj.mArray)\
        ,__size(obj.__size)\
        ,__alloc(obj.__alloc)\
    {\
        obj.mArray=nullptr;\
    }\
    ~ClassName(){\
        delete[] mArray;\
        mArray=nullptr;\
    }\
    void clear(){\
        delete[] mArray;\
        mArray=nullptr;\
        __size=__alloc=0;\
    }\
    ClassName &operator=(const ClassName &obj){\
        if(this==&obj)\
            return *this;\
        resize(obj.__size, false);\
        __size=obj.__size;\
        memcpy(mArray, obj.mArray, obj.__size*sizeOfT);\
        return *this;\
    }\
    ClassName &operator=(ClassName &&obj){\
        if(this==&obj) \
            return *this;\
        delete[] mArray;\
        mArray=obj.mArray;\
        __size=obj.__size;\
        __alloc=obj.__alloc;\
        obj.mArray=nullptr;\
        obj.__size = obj.__alloc=0;\
        return *this;\
    }\
    T &operator[](size_t idx) const{\
        if(!tryAI::toBoundedIdx(idx,__size,&idx))\
            throw std::out_of_range("From LLUVector::operator[]:\n\tOut of range");\
        return mArray[idx];\
    }\
    size_t size() const {return __size;}\
    T *data() const {return mArray;}\
    bool empty() const {return !__size;}

namespace std{

template<>
class vector<size_t>{
    VECTOR_PARTIAL_SPE(vector, long long unsigned)
public:
    //INIT_LIST_CONSTRUCTOR(vector, int, size_t)
    //INIT_LIST_CONSTRUCTOR(vector, unsigned, size_t)
    //INIT_LIST_CONSTRUCTOR(vector, long long, size_t)
    
    friend ostream &operator<<(ostream &osm, const vector<size_t> &vec){
        const auto size=vec.size();
        osm<<'{';
        for(size_t i=0;i<size;++i){
            osm<<vec[i];
            if(i+1!=size)
                osm<<',';
        }
        osm<<'}';
        return osm;
    }
};
}


/*
template<>
class vector<double>{
    VECTOR_PARTIAL_SPE(double)
public:
    INIT_LIST_CONSTRUCTOR(vector, int, double)
    INIT_LIST_CONSTRUCTOR(vector, unsigned, double)
    INIT_LIST_CONSTRUCTOR(vector, long long, double)
    INIT_LIST_CONSTRUCTOR(vector, float, double)
    INIT_LIST_CONSTRUCTOR(vector, unsigned long long, double)
};
template<class Ele>
class vector<Ele*>{
    VECTOR_PARTIAL_SPE(Ele*)
};*/
namespace tryAI{
class DoubleVector{
    VECTOR_PARTIAL_SPE(DoubleVector, double)
public:
    INIT_LIST_CONSTRUCTOR(DoubleVector, int, double)
    INIT_LIST_CONSTRUCTOR(DoubleVector, unsigned, double)
    INIT_LIST_CONSTRUCTOR(DoubleVector, long long, double)
    INIT_LIST_CONSTRUCTOR(DoubleVector, float, double)
    INIT_LIST_CONSTRUCTOR(DoubleVector, unsigned long long, double)
};
template<class Ele>
class PtrVector{
    VECTOR_PARTIAL_SPE(PtrVector, Ele*)
    void shrinkTo(size_t newSize){
        if(__size<newSize)
            throw std::runtime_error("From PtrVector::shrinkTo(size_t):\n\t shrink to bigger size");
        if(__size==newSize)
            return;
        T *tmp=mArray;
        mArray=new T[newSize];
        memcpy(mArray, tmp, newSize*sizeOfT);
        delete[] tmp;
        __size=__alloc=newSize;
    }
};
}
#undef INIT_LIST_CONSTRUCTOR
#endif
#endif