#ifndef CPPTENSOR_SHAPEDARRAY_INDEX_H
#define CPPTENSOR_SHAPEDARRAY_INDEX_H
#include "./shape.hpp"
#include "./pointer.hpp"
namespace tryAI{
class ShapedArray;
class Slice{
private:
    size_t begin;
    size_t stepSize;
    size_t end;
public:
    explicit Slice(size_t begin_=0, size_t end_=-1, size_t stepSize_=1); //explicit是必须的，不许删！
    ShapedArray toIndices(size_t length) const;
    H_OUTPUTABLE(Slice)
};
#define genIndexMacro(type) ShapedArray genIndex(type num, size_t);
genIndexMacro(size_t)
genIndexMacro(long long)
genIndexMacro(int)
genIndexMacro(unsigned)
genIndexMacro(unsigned short)
genIndexMacro(short)
genIndexMacro(const ShapedArray&)
genIndexMacro(const Slice&)
#undef genIndexMacro
class Index
{
public:
    union _Index{
        Slice slice;
        size_t idx; //为什么坚持用size_t: 因为toBoundedIdx函数会帮我们做事. 我们假定负数只会是-1等很贴近0的数
        _Index(const Slice &slice_):slice(slice_){}
        _Index(size_t idx_):idx(idx_){}
        _Index():idx(0){}
    };
private:
    _Index data;
    bool isNum;
public:
    Index(long long idx=0):data(idx), isNum(true){}
    Index(const Slice &slice):data(slice), isNum(false){}
    H_OUTPUTABLE(Index)
};

template<class T>
class SimpleShapedArray
{
private:
    UniquePointer<T> mArray;
    Shape shape;
    friend class ShapedArray;
public:
    /**
     * @brief 构造函数，初始化列表构造
     * @param init_list 初始化列表 
     * @attention
     */
    explicit SimpleShapedArray(std::initializer_list<T> init_list)
    :mArray(init_list.size()?new T[init_list.size()] : nullptr), shape(init_list.size()?Shape({init_list.size()}):Shape())
    {
        memcpy(mArray, init_list.begin(), init_list.size()*sizeof(T));
    }
    explicit SimpleShapedArray(std::initializer_list<SimpleShapedArray<T>> arrs)
    :mArray(nullptr),shape()
    {
        const size_t tCnt=arrs.size();
        if(!tCnt)
            return;
        auto begin = arrs.begin();
        for(size_t i=0;i<tCnt-1;++i)
            if(begin[i].shape!=begin[i+1].shape)
                throw std::runtime_error("From ShapedArray::ShapedArray(vector<ShapedArray>, Shape):\n\tTensors should have same shape");
        shape = (tCnt?Shape{tCnt}:Shape()) + begin[0].shape;
        mArray = new T[shape.bufSize()];
#if DEBUG
        std::cout<<"SSA Alloc @"<<static_cast<void*>(mArray)<<'['<<shape.bufSize()<<']'<<std::endl;
#endif
        const size_t stepSize = begin[0].shape.bufSize();
        for(size_t i=0;i<tCnt;++i)
        {
            memcpy(mArray+i*stepSize, begin[i].mArray, sizeof(T)*stepSize);
        }
    }
    ~SimpleShapedArray()
    {
        mArray.clear();
        shape.clear();
    }
    T &operator[](size_t offset) const 
    {
        size_t bufSize = shape.bufSize();
        if(!toBoundedIdx(offset, bufSize, &offset))
            throw std::out_of_range("From SimpleShapedArray::[](size_t):\n\t<offset> Out of range");
        return mArray[offset];
    }
    void print(std::ostream &osm=std::cout) const 
    {
        printShaped<T>(mArray, shape, osm);
    }
};
#define IDX(...) tryAI::SimpleShapedArray<tryAI::Index>{__VA_ARGS__}
}
namespace std
{
template<>
class vector<tryAI::Index>{
    VECTOR_PARTIAL_SPE(vector, tryAI::Index)
};
}
#endif