#pragma once
#include "shape.hpp"
#include "pointer.hpp"
#include <functional>
namespace tryAI{
#define atCond(...) \
at([&](auto &num){ return __VA_ARGS__; })
typedef std::vector<size_t> VecN;

class Tensor;
/**
 * @brief 引用张量
 * @attention 不新分配内存、只用别人的
*/
class RefTensor{
public:
    using Number=double; //数字类型
    using NumberPtr=Number*; 
    friend class Tensor;
private:
    NumberPtr *mArray; //内存条，里面存的不是Number，而是别人的Number*
    Shape shape; //形状
    RefTensor(const PtrVector<Number> &init_list={});
    RefTensor(const std::vector<RefTensor> &refTensors);
public:
    ~RefTensor();
    RefTensor(const RefTensor &obj)=delete;
    RefTensor(RefTensor &&obj);
    operator Tensor();
    Tensor asTensor() const;
    void operator=(const Tensor &obj); //用Tensor赋值，要直接更改！
    void operator=(const RefTensor &obj)=delete;
    void operator=(RefTensor &&obj);
    void print(std::ostream &osm=std::cout) const{osm<<(*this)<<std::endl;}
    H_OUTPUTABLE(RefTensor)
};

#define ASMD_OPERATOR_DEFINE(operatorName)\
Tensor operator operatorName(const Tensor &obj) const{\
    Tensor res=broadcast(*this, obj.shape);\
    Tensor tmp=broadcast(obj, res.shape);\
    auto bufSize=res.shape.bufSize();\
    for(size_t i=0;i<bufSize;++i)\
        res.mArray[i] operatorName##= tmp.mArray[i];\
    return res;\
}\
Tensor &operator operatorName##= (const Tensor &obj){\
    *this=broadcast(*this, obj.shape);\
    Tensor tmp=broadcast(obj, shape);\
    const auto bufSize=shape.bufSize();\
    for(size_t i=0;i<bufSize;++i)\
        mArray[i] operatorName##= tmp.mArray[i];\
    return *this;\
}
/**
 * @brief 张量类
 */
class Tensor
{
public:
    using Number=double; //数字类 TODO以后换成一个自动分配的Number
    using NumberPtr=UniquePointer<Number>; //用智能指针管理内存
    friend class RefTensor;
private:
    NumberPtr mArray; //内存条
    Shape shape; //用什么形状解读这一块内存
    /**
     * @brief 私有构造函数
     * @param ptr 内存条
     * @param shape_ 形状
     * @attention 不检查内存，请使用者自行分配
    */
    Tensor(NumberPtr &&ptr, Shape &&shape_):mArray(std::move(ptr)),shape(std::move(shape_)){}
    Tensor(NumberPtr &&ptr, const Shape &shape_):mArray(std::move(ptr)),shape(shape_){}
public://Memory
    /**
     * @brief 空构造函数
    */
    constexpr Tensor():mArray(nullptr), shape(){}
    /**
     * @brief 构造函数
     * @param init_list 初始化列表
     * @param shape_ 初始化形状; 默认为空，会自动补充为一维
    */
    Tensor(std::initializer_list<Number> init_list, const Shape &shape_=Shape());
    /**
     * @brief 构造函数
     * @param num 数字
    */
    Tensor(Number num, const Shape &shape_=Shape({}));
    /**
     * @brief 拷贝构造函数
     * @attention 会新分配内存
     * @param obj 另一个对象
    */
    Tensor(const Tensor &obj);
    /**
     * @brief 移动构造函数
     * @attention 不会新分配内存
     * @param obj 另一个对象
    */
    Tensor(Tensor &&obj);
    /**
     * @brief 析构函数
    */
    ~Tensor(){clear();}
    /**
     * @brief 赋值
     * @attention 会新分配内存
     * @param obj 另一个对象
     * @return 自己
    */
    Tensor &operator=(const Tensor &obj);
    /**
     * @brief 赋值
     * @attention 不会新分配内存
     * @param obj 另一个对象
     * @return 自己
    */
    Tensor &operator=(Tensor &&obj);
private:
    /**
     * @brief 清除内存
    */
    void clear();
    /**
     * @brief 广播
     * @param src 未广播的张量
     * @param shape 与src广播的形状
     * @return 广播后的张量
    */
    static Tensor broadcast(const Tensor &src, const Shape &shape);
public:
    /**
     * @brief 输出
     * @param osm 所用输出流(默认cout)
    */
    void print(std::ostream &osm=std::cout) const {osm<<*this<<std::endl;}
    H_OUTPUTABLE(Tensor)
    /**
     * @brief 更改形状
     * @param shape_ 新形状
     * @attention 元素个数要一致
    */
    void reshape(const Shape &shape_);
    /**
     * @brief 获取形状
     * @return 形状
    */
    const Shape &getShape() const {return shape;}
    /**
     * @brief 获取数据量
     * @return 数据量
    */
    size_t getSize() const {return shape.bufSize();}
    /**
     * @brief 筛选得到符合条件的值
     * @param cond 筛选函数，参数是元素
     * @return 一个RefTensor，引用符合条件的元素
    */
    RefTensor at(std::function<bool(const Number &)> cond) const;
    /**
     * @brief 获取元素，同numpy.ndarray[tuple]
     * @param index0 下标
     * @param indices 下标
     * @return 一个RefTensor
    */
    template<class ...Indices>
    RefTensor at(const size_t &index0, const Indices &...indices) const{
        VecN index({index0, (static_cast<size_t>(indices))...});
        if(index.empty()||index.size()>shape.dimNumber())
            throw std::runtime_error("From Tensor::at(const vector &):\n\tWrong size of index");
        Number *head=mArray;
        Number *be=head;
        const auto argCount=index.size();
        size_t idx;
        size_t dimSize;
        for(size_t i=0;i<argCount;++i){
            dimSize=shape.dimSizeOf(i);
            if(!toBoundedIdx(index[i],dimSize,&idx))
                throw std::out_of_range("From Tensor::at(const vector &):\n\tOut of range");
            be+=idx*shape.stepSizeOf(i);
        }
        auto resCnt=shape.stepSizeOf(argCount-1);
        PtrVector<Number> res(resCnt);
        for(size_t i=0;i<resCnt;++i)
            res[i]=be+i;
        RefTensor rt(res);
        rt.shape=shape.sliced(argCount);
        return rt;
    }
    /**
     * @brief 获取某些元素，同numpy.ndarray[tuple]
     * @param index0 下标
     * @param indices 下标
     * @return 一个RefTensor
     */
    template<class T, class ...Indices>
    RefTensor at(const std::vector<T> &index0, const Indices &...indices){
        std::vector<std::vector<T>> index={index0, std::vector<T>(indices)...};
        auto resLen=index0.size();
        for(size_t i=0;i<sizeof...(Indices);++i){
            if(index[i+1].size()!=resLen)
                throw std::runtime_error("From Tensor::at(vector...):\n\tNot same size of index");
        }
        std::vector<RefTensor> res;
        for(size_t i=0;i<resLen;++i){
            res.push_back(at(index0[i],(indices[i])...));
        }
        return res;
    }
    template<class T, class ...Indices>
    RefTensor at(const list<T> &index0, const Indices &...indices){
        return at(index0.getVector(), (indices.getVector())...);
    }
    /**
     * @brief 对每个值进行操作
     * @param func 对每个值操作的函数
     * @param ret 对每个func返回值进行的处理，结果是整个foreach的返回值；默认为空
     * @tparam Ret1 func返回值类型
     * @tparam Ret2 整个foreach的返回值类型
     * @return ret({func(), func()...})的结果;如果ret为空，则返回Ret2()
     * @attention 用lambda表达式时记得填写模板参数
    */
    template<class Ret1, class Ret2>
    Ret2 foreach(std::function<Ret1(Number &)> func, std::function<Ret2(std::vector<Ret1>)> ret=nullptr){
        const auto size=shape.bufSize();
        Number *head=mArray;
        if(ret==nullptr){
            for(size_t i=0;i<size;++i)
                func(head[i]);
            return Ret2();
        }
        std::vector<Ret1> rets(size);
        for(size_t i=0;i<size;++i)
            rets[i]=func(head[i]);
        return ret(rets);
    }
    /**
     * @brief 对每个值进行操作
     * @param func 对每个值操作的函数
     */
    void foreach(std::function<void(Number &)> func);
    ASMD_OPERATOR_DEFINE(+)
    ASMD_OPERATOR_DEFINE(-)
    ASMD_OPERATOR_DEFINE(*)
    ASMD_OPERATOR_DEFINE(/)
};
/**
 * @brief (递归实现)输出有形状的数组的函数
 * @param arr 输出内存的首地址
 * @param dim shape的第几维
 * @param shape 数组形状
 * @param osm 所用输出流，默认std::cout
 * @param output 输出每个元素的函数，默认osm<<ele
*/
template<class T>
void printShaped(
    const T *arr, size_t dim, const Shape &shape, 
    std::ostream &osm=std::cout,
    const std::function<void(std::ostream&,const T &)> &output
    =[](std::ostream &osm, const T &ele){osm<<ele;}
) {
    //std::cout<<shape<<std::endl;
    if(shape.isEmpty()){
        //空数组
        osm<<"[]";
        return;
    }
    if(dim==shape.dimNumber()){
        //递归终点，输出数字
        output(osm, *arr);
        return;
    }
    //递归：输出所有下一级数(组)
    auto dimSize=shape.dimSizeOf(dim);
    auto stepSize=shape.stepSizeOf(dim);
    auto dimNumber=shape.dimNumber();
    osm<<'[';
    for(size_t i=0;i<dimSize;++i){
        printShaped(arr+i*stepSize,dim+1,shape,osm,output);
        if(i+1==dimSize)//i==dimSize-1
            osm<<']'; //到最后一个元素了，收尾
        else if(dim+1==dimNumber)//dim==dimNumber-1
            osm<<", ";//不是最后一个元素，而且输出的是数字
        else{ //不是最后一个元素，而且输出的不是数字
            osm<<",\n";
            if(dim+2!=dimNumber) //dim!=dimNumber-2
                osm<<'\n';//多维数组多换一行，看着得劲
            for(size_t j=0;j<=dim;++j)
                osm<<' ';//换行之后补全缩进
        }
    }
}
}
/*

tuple of X: arr.at(a,b,c,d)
arr.at(a,b,c,d)
表示前四个轴上分别第a[i],b[i],c[i],d[i]个，并将i个组合起来
tensor({ arr[a[i]][b[i]][c[i]][d[i]], ... })
如果i只有一个，转为数字；i的个数是len(a)==len(b)==len(c)...

list of X: arr.at([a,b,c,d])
arr.at(list{a,b,c,d})
表示分别取下标，等价于
tensor({ arr.at(a), arr.at(b), arr.at(c), arr.at(d) })

其中，X是list或tuple都一样

*/