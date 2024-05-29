#ifndef CPPTENSOR_SHAPEDARRAY_REF_H
#define CPPTENSOR_SHAPEDARRAY_REF_H
#include "./shape.hpp"
namespace tryAI
{

class ShapedArray;
/**
 * @brief 引用张量
 * @attention 不新分配内存、只用别人的
*/
class ShapedRefArray{
public:
    using Number=double; //数字类型
    using NumberPtr=Number*; 
    friend class ShapedArray;
private:
    NumberPtr *mArray; //内存条，里面存的不是Number，而是别人的Number*
    Shape shape; //形状
    //用在at(func)里
    ShapedRefArray(const PtrVector<Number> &init_list={});
    //用在at(...)里
    ShapedRefArray(Number **buf, const Shape &shape_);
    //用在？？
    ShapedRefArray(const std::vector<ShapedRefArray> &refTensors);
public:
    ~ShapedRefArray();
    ShapedRefArray(const ShapedRefArray &obj)=delete;
    ShapedRefArray(ShapedRefArray &&obj);
    operator ShapedArray();
    ShapedArray asTensor() const;
    void operator=(const ShapedRefArray &obj)=delete;
    void operator=(ShapedRefArray &&obj);
    
    void operator=(const ShapedArray &obj); //用Tensor赋值，要直接更改！
    void operator+=(const ShapedArray &obj); //运算赋值的实现在shapedarray_operators.cpp里
    void operator-=(const ShapedArray &obj);
    void operator*=(const ShapedArray &obj);
    void operator/=(const ShapedArray &obj);

    void print(std::ostream &osm=std::cout) const{osm<<(*this)<<std::endl;}
    H_OUTPUTABLE(ShapedRefArray)
    template<class ...Indices>
    ShapedRefArray at(const size_t &index0, const Indices &...indices) const{
        std::vector<size_t> index({index0, (static_cast<size_t>(indices))...});
        const auto argCount=index.size();
        if(index.empty()||argCount>shape.dimNumber())
            throw std::runtime_error("From ShapedArray::at(const vector &):\n\tWrong size of index");
        auto [be,resCnt]=shape.offsetOf(index);
        PtrVector<Number> res(resCnt);
        for(size_t i=0;i<resCnt;++i)
            res[i]=mArray[be+i];
        ShapedRefArray rt(res);
        rt.shape=shape.sliced(argCount);
        return rt;
    }
    /**
     * @brief 获取某些元素，同numpy.ndarray[tuple]
     * @param index0 下标
     * @param indices 下标
     * @return 一个RefTensor
     */
    // template<class T, class ...Indices>
    // ShapedRefArray at(const list<T> &index0, const Indices &...indices){
    //     list<list<T>> index={
    //         index0, 
    //         (indices)...};
    //     auto resLen=index0.size();
    //     for(size_t i=0;i<sizeof...(Indices);++i){
    //         if(index[i+1].size()!=resLen)
    //             throw std::runtime_error("From ShapedArray::at(vector...):\n\tNot same size of index");
    //     }
    //     std::vector<ShapedRefArray> res;
    //     for(size_t i=0;i<resLen;++i){
    //         res.push_back(at(index0[i],(indices[i])...));
    //     }
    //     return res;
    // }
};

}
#endif