#ifndef NUMCPP_SHAPED_ARRAY_HPP
#define NUMCPP_SHAPED_ARRAY_HPP
#include "./shape.hpp"
#include "../utils/pointer.hpp"
namespace numcpp{
// #define atCond(...) \
// where([&](auto &num){ return __VA_ARGS__; })

// #define $$(X) .matmul(X)


/**
 * @brief 张量类
 */
template<class DType = double>
class ShapedArray
{
public:
    using dtype = DType; //数字类 TODO以后换成一个自动分配的Number
    using UPtr=UniquePointer<DType>; //用智能指针管理内存
private:
    UPtr mArray; //内存条
    Shape shape; //用什么形状解读这一块内存
public://Memory
    /**
     * @brief 空构造函数
    */
    constexpr ShapedArray():mArray(nullptr), shape(){}
    /**
     * @brief 构造函数
     * @param num 数字，初始化值
     * @param shape_ 初始化形状，不填默认为形状{}(即数字)
    */
    ShapedArray(const DType &num, const Shape &shape_=Shape({}));
    /**
     * @brief 构造函数
     * @param init_list 初始化列表
     * @param shape_ 初始化形状; 默认为空，会自动补充为一维
    */
    explicit ShapedArray(std::initializer_list<DType> init_list, const Shape &shape_=Shape());
    /**
     * @brief 构造函数
     * @param init_vec 初始化列表
     * @param shape_ 初始化形状; 默认为空，会自动补充为一维
    */
    explicit ShapedArray(const std::vector<DType> &init_vec, const Shape &shape_=Shape());
    /**
     * @todo 改为stack
     * @brief 构造函数，构造高维tensor
     * @param tensors 一堆形状一致的tensor
     * @param shape_ 这些tensor的组合的形状，默认填充为{tensors.size()}
     * @attention 直接用initialized_list传参会出错, 受限于C++语法, 我无法更改此错误. 请只传vector类型的参数或在initialized_list前显式标注ShapedArray
    */
    explicit ShapedArray(const std::vector<ShapedArray> &tensors, const Shape &shape_=Shape());
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
    ShapedArray &operator=(
        std::conditional_t<
            std::is_pointer_v<DType>, 
            ShapedArray<std::remove_pointer_t<DType>>,
            ShapedArray<DType>
        > const &obj);
    /**
     * @brief 赋值
     * @attention 不会新分配内存
     * @param obj 另一个对象
     * @return 自己
    */
    ShapedArray &operator=(
        std::conditional_t<
            std::is_pointer_v<DType>, 
            ShapedArray<std::remove_pointer_t<DType>>,
            ShapedArray<DType>
        > &&obj);
    /**
     * @brief 把一个序列变成ShapedArray
     * @param args 一个序列，数字组成或者是ShapedArray组成
     * @return 一个ShapedArray
     * @example ShapedArray::fromSequence(1,2,3)
    */
    template<class ...Args>
    static ShapedArray fromSequence(Args... args) {
        return ShapedArray(std::vector<DType>{
            (static_cast<DType>(args))...
        });
    }
private:
    /**
     * @brief 清除内存
    */
    void clear();
    /**
     * @brief 获取元素的内部函数
     * @param index 下标(数字们)
     * @param resBegin 把被取出的地址的引用值存放的位置的起点
     * @return 存放终点，resBegin+resCnt
    */
    // DType **atByNumbers(const Shape::SizeTArray &index, DType **resBegin) const;
    
public:
    /**
     * @brief 输出
     * @param osm 所用输出流(默认cout)
    */
    void print(std::ostream &osm=std::cout) const {osm<<*this<<std::endl;}
    H_OUTPUTABLE(ShapedArray<DType>){
        if constexpr(std::is_pointer_v<DType>)
            printShaped<DType>(obj.mArray, obj.shape, osm, [](std::ostream &osm_, DType const &ele){
                osm_<<*ele;
            });
        else printShaped<DType>(obj.mArray,obj.shape,osm);
        return osm/*<<"\nshape: "<<obj.shape*/;
    }
    /**
     * @brief 更改形状
     * @param shape_ 新形状
     * @attention 元素个数要一致
    */
    void to(const Shape &shape_);
    /**
     * @brief 更改类型
     * @tparam T 更改的dtype
     */
    template<class T>
    ShapedArray<T> to();
    /**
     * @brief 更改形状，返回新的array
     * @param shape_ 新形状
     * @return 
     * @attention 元素个数要一致
    */
    ShapedArray<DType> reshape(const Shape &shape_);
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
     * @brief 获取数据条地址
     * @return 数据地址
     */
    DType *data() const {return mArray;}
    /**
     * @brief 筛选得到符合条件的值
     * @param cond 筛选函数，参数是元素
     * @return 一个RefTensor，引用符合条件的元素
    */
    // ShapedRefArray where(std::function<bool(const DType &)> cond) const;
    
    /**
     * @brief 获取某些元素，同numpy.ndarray.__getitem__
     * @param indices 下标
     * @return 一个RefTensor
     * @example python: a[[1,2], 3, 3:4]
     * c++: a.at(IDX{1,2}, 3, Slice(3,4))
     */
    template<class ...Args>
    ShapedArray<DType*> at(const Args &... indices);
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
    Ret2 foreach(std::function<Ret1(DType &)> func, std::function<Ret2(std::vector<Ret1>)> ret=nullptr){
        const auto size=shape.bufSize();
        DType *head=mArray;
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
    void foreach(std::function<void(DType &)> func);
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

    // friend ShapedArray exp(const ShapedArray &obj);
    // friend ShapedArray log(const ShapedArray &obj);
    // friend ShapedArray sigmoid(const ShapedArray &obj);
    // friend ShapedArray sin(const ShapedArray &obj);
    // friend ShapedArray cos(const ShapedArray &obj);
    // friend ShapedArray tan(const ShapedArray &obj);
    // friend ShapedArray cot(const ShapedArray &obj);
    // friend ShapedArray sec(const ShapedArray &obj);
    // friend ShapedArray csc(const ShapedArray &obj);
    // friend ShapedArray asin(const ShapedArray &obj);
    // friend ShapedArray acos(const ShapedArray &obj);
    // friend ShapedArray atan(const ShapedArray &obj);
    
    // friend ShapedArray arange(DType be, DType en, DType step, const Shape &shape);
};

template<class T>
inline Shape shapeOf(const T &obj){
    return Shape({});
}
template<class T>
inline Shape shapeOf(const ShapedArray<T> &obj){
    return obj.getShape();
}

inline double abs(double a){return a>0?a:-a;}
/**
 * @brief 计算e指数
 * @param obj 输入张量
 * @return 返回一个张量
*/
// ShapedArray exp(const ShapedArray &obj);
// /**
//  * @brief 计算e对数
//  * @param obj 输入张量
//  * @return 返回一个张量
// */
// ShapedArray log(const ShapedArray &obj);
// /**
//  * @brief 激活函数sigmoid
//  * @param obj 输入张量
//  * @return 返回一个张量
// */
// ShapedArray sigmoid(const ShapedArray &obj);

// ShapedArray sin(const ShapedArray &obj);
// ShapedArray cos(const ShapedArray &obj);
// ShapedArray tan(const ShapedArray &obj);
// ShapedArray cot(const ShapedArray &obj);
// ShapedArray sec(const ShapedArray &obj);
// ShapedArray csc(const ShapedArray &obj);
// ShapedArray asin(const ShapedArray &obj);
// ShapedArray acos(const ShapedArray &obj);
// ShapedArray atan(const ShapedArray &obj);
// ShapedArray zeros(const Shape &shape);
// ShapedArray ones(const Shape &shape);
// ShapedArray arange(ShapedArray::DType be, ShapedArray::DType en, ShapedArray::DType step=1, const Shape &shape=Shape());

}
//---------------------------------------------------实现---------------------------------------------------


#define METHOD(...) template<class DType> __VA_ARGS__ ShapedArray<DType>
#include "./_array_arr.hpp"
#include "./_array_idx.hpp"
#undef METHOD
#endif
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