#pragma once
#include "./shape.hpp"
namespace tryAI{
#define atCond(...) \
at([&](auto &num){ return __VA_ARGS__; })

#define $$(X) .matmul(X)

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
    ShapedRefArray(const PtrVector<Number> &init_list={});
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
    void operator+=(const ShapedArray &obj);
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
    template<class T, class ...Indices>
    ShapedRefArray at(const list<T> &index0, const Indices &...indices){
        list<list<T>> index={
            index0, 
            (indices)...};
        auto resLen=index0.size();
        for(size_t i=0;i<sizeof...(Indices);++i){
            if(index[i+1].size()!=resLen)
                throw std::runtime_error("From ShapedArray::at(vector...):\n\tNot same size of index");
        }
        std::vector<ShapedRefArray> res;
        for(size_t i=0;i<resLen;++i){
            res.push_back(at(index0[i],(indices[i])...));
        }
        return res;
    }
};


/**
 * @brief 张量类
 */
class ShapedArray
{
public:
    using Number=double; //数字类 TODO以后换成一个自动分配的Number
    using NumberPtr=UniquePointer<Number>; //用智能指针管理内存
    friend class ShapedRefArray;
private:
    NumberPtr mArray; //内存条
    Shape shape; //用什么形状解读这一块内存
    /**
     * @brief 私有构造函数
     * @param ptr 内存条
     * @param shape_ 形状
     * @attention 不检查内存，请使用者自行分配
    */
    ShapedArray(NumberPtr &&ptr, Shape &&shape_):mArray(std::move(ptr)),shape(std::move(shape_)){}
    ShapedArray(NumberPtr &&ptr, const Shape &shape_):mArray(std::move(ptr)),shape(shape_){}
public://Memory
    /**
     * @brief 空构造函数
    */
    constexpr ShapedArray():mArray(nullptr), shape(){}
    /**
     * @brief 构造函数
     * @param init_list 初始化列表
     * @param shape_ 初始化形状; 默认为空，会自动补充为一维
    */
    ShapedArray(std::initializer_list<Number> init_list, const Shape &shape_=Shape());
    /**
     * @brief 构造函数
     * @param num 数字，初始化值
     * @param shape_ 初始化形状，不填默认为形状{}(即数字)
    */
    ShapedArray(Number num, const Shape &shape_=Shape({}));
    /**
     * @brief 构造函数，构造高维tensor
     * @param tensors 一堆形状一致的tensor
     * @param shape_ 这些tensor的形状，默认填充为{tensors.size(), (tensors[0].shape...)}
    */
    ShapedArray(const std::vector<ShapedArray> &tensors, const Shape &shape_=Shape());
    /**
     * @brief 拷贝构造函数
     * @attention 会新分配内存
     * @param obj 另一个对象
    */
    ShapedArray(const ShapedArray &obj);
    /**
     * @brief 移动构造函数
     * @attention 不会新分配内存
     * @param obj 另一个对象
    */
    ShapedArray(ShapedArray &&obj);
    /**
     * @brief 析构函数
    */
    ~ShapedArray(){clear();}
    /**
     * @brief 赋值
     * @attention 会新分配内存
     * @param obj 另一个对象
     * @return 自己
    */
    ShapedArray &operator=(const ShapedArray &obj);
    /**
     * @brief 赋值
     * @attention 不会新分配内存
     * @param obj 另一个对象
     * @return 自己
    */
    ShapedArray &operator=(ShapedArray &&obj);

    static ShapedArray zeros(const Shape &shape) {return ShapedArray(0,shape);}
    static ShapedArray ones(const Shape &shape) {return ShapedArray(1,shape);}
    static ShapedArray arange(Number be, Number en, Number step=1, const Shape &shape=Shape());
private:
    /**
     * @brief 清除内存
    */
    void clear();
public:
    /**
     * @brief 输出
     * @param osm 所用输出流(默认cout)
    */
    void print(std::ostream &osm=std::cout) const {osm<<*this<<std::endl;}
    H_OUTPUTABLE(ShapedArray)
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
    ShapedRefArray at(std::function<bool(const Number &)> cond) const;
    /**
     * @brief 获取元素，同numpy.ndarray[tuple]
     * @param index0 下标
     * @param indices 下标
     * @return 一个RefTensor
    */
    template<class ...Indices>
    ShapedRefArray at(size_t index0, Indices ...indices) const{
        std::vector<size_t> index({index0, (static_cast<size_t>(indices))...});
        const auto argCount = index.size();
        if(index.empty()||argCount>shape.dimNumber())
            throw std::runtime_error("From ShapedArray::at(const vector &):\n\tWrong size of index");
        Number *head=mArray;
        Number *be=head;
        auto [offset, resCnt] = shape.offsetOf(index);
        be+=offset;
        PtrVector<Number> res(resCnt);
        for(size_t i=0;i<resCnt;++i)
            res[i]=be+i;
        ShapedRefArray rt(res);
        rt.shape=shape.sliced(argCount);
        return rt;
    }
    /**
     * @brief 获取某些元素，同numpy.ndarray[tuple]
     * @param index0 下标
     * @param indices 下标
     * @return 一个RefTensor
     * @todo 下标应该弄成可以broadcast成一致的就OK，并且没写sliced下标
     */
    template<class T, class ...Indices>
    ShapedRefArray at(const list<T> &index0, const Indices &...indices){
        list<list<T>> index={
            index0, 
            (indices)...};
        auto resLen=index0.size();
        for(size_t i=0;i<sizeof...(Indices);++i){
            if(index[i+1].size()!=resLen)
                throw std::runtime_error("From ShapedArray::at(vector...):\n\tNot same size of index");
        }
        std::vector<ShapedRefArray> res;
        for(size_t i=0;i<resLen;++i){
            res.push_back(at(index0[i],(indices[i])...));
        }
        return res;
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
    /**
     * @brief 缩减长度是1的维度
    */
    void squeeze() {shape.squeeze();}
    /**
     * @brief 缩减长度是1的维度
     * @param dim 维度数. 如果该维度长度非1, 会报错
    */
    void squeeze(size_t dim){shape.squeeze(dim);}
    /**
     * @brief 判断是否相同
     * @param obj 另一个tensor
     * @return 相同则true
     * @attention 不是operator==
    */
    bool isEqualTo(const ShapedArray &obj) const;
    ShapedArray &operator+=(const ShapedArray &obj);
    ShapedArray &operator-=(const ShapedArray &obj);
    ShapedArray &operator*=(const ShapedArray &obj);
    ShapedArray &operator/=(const ShapedArray &obj);
    ShapedArray operator+(const ShapedArray &obj) const;
    ShapedArray operator-(const ShapedArray &obj) const;
    ShapedArray operator*(const ShapedArray &obj) const;
    ShapedArray operator/(const ShapedArray &obj) const;
    ShapedArray matmul(const ShapedArray &obj) const;

    friend ShapedArray exp(const ShapedArray &obj);
    friend ShapedArray log(const ShapedArray &obj);
    friend ShapedArray sigmoid(const ShapedArray &obj);
    friend ShapedArray sin(const ShapedArray &obj);
    friend ShapedArray cos(const ShapedArray &obj);
    friend ShapedArray tan(const ShapedArray &obj);
    friend ShapedArray cot(const ShapedArray &obj);
    friend ShapedArray sec(const ShapedArray &obj);
    friend ShapedArray csc(const ShapedArray &obj);
    friend ShapedArray asin(const ShapedArray &obj);
    friend ShapedArray acos(const ShapedArray &obj);
    friend ShapedArray atan(const ShapedArray &obj);
};
inline double abs(double a){return a>0?a:-a;}
/**
 * @brief 计算e指数
 * @param obj 输入张量
 * @return 返回一个张量
*/
ShapedArray exp(const ShapedArray &obj);
/**
 * @brief 计算e对数
 * @param obj 输入张量
 * @return 返回一个张量
*/
ShapedArray log(const ShapedArray &obj);
/**
 * @brief 激活函数sigmoid
 * @param obj 输入张量
 * @return 返回一个张量
*/
ShapedArray sigmoid(const ShapedArray &obj);

ShapedArray sin(const ShapedArray &obj);
ShapedArray cos(const ShapedArray &obj);
ShapedArray tan(const ShapedArray &obj);
ShapedArray cot(const ShapedArray &obj);
ShapedArray sec(const ShapedArray &obj);
ShapedArray csc(const ShapedArray &obj);
ShapedArray asin(const ShapedArray &obj);
ShapedArray acos(const ShapedArray &obj);
ShapedArray atan(const ShapedArray &obj);

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

TODO:
sum/squeeze


*/