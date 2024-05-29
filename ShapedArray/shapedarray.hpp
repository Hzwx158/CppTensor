#ifndef CPPTENSOR_SHAPEDARRAY__H
#define CPPTENSOR_SHAPEDARRAY__H
#include "./shapedRefArray.hpp"
#include "./index.hpp"
namespace tryAI{
#define atCond(...) \
where([&](auto &num){ return __VA_ARGS__; })

#define $$(X) .matmul(X)

class ShapedArray;
template<class T>

struct IsShapedArray{
    static constexpr bool flag = false;
};
template<>
struct IsShapedArray<ShapedArray>{
    static constexpr bool flag = true;
};

// struct ShapedArrayView
// {
//     using Number=double; //数字类 TODO以后换成一个自动分配的Number
//     const Number *mBegin;
//     Shape shape;
//     ShapedArrayView():mBegin(nullptr), shape(){}
//     ShapedArrayView(const Number *mBegin_, const Shape &shape_);
//     ShapedArrayView(const ShapedArray &shapedArray);
//     ShapedArrayView(const ShapedArrayView &)=default;
//     ShapedArrayView(ShapedArrayView &&)=default;
//     ShapedArrayView operator[](size_t idx) const;
//     ShapedArrayView &operator=(const ShapedArrayView &)=default;
//     ShapedArrayView &operator=(ShapedArrayView &&)=default;
// };

/**
 * @brief 张量类
 */
class ShapedArray
{
public:
    using Number=double; //数字类 TODO以后换成一个自动分配的Number
    using NumberPtr=UniquePointer<Number>; //用智能指针管理内存
    friend class ShapedRefArray;
    friend class ShapedArrayView;
    friend class Slice;
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
     * @param num 数字，初始化值
     * @param shape_ 初始化形状，不填默认为形状{}(即数字)
    */
    ShapedArray(Number num, const Shape &shape_=Shape({}));
    /**
     * @brief 构造函数
     * @param init_list 初始化列表
     * @param shape_ 初始化形状; 默认为空，会自动补充为一维
    */
    explicit ShapedArray(std::initializer_list<Number> init_list, const Shape &shape_=Shape());
    /**
     * @brief 构造函数
     * @param init_vec 初始化列表
     * @param shape_ 初始化形状; 默认为空，会自动补充为一维
    */
    ShapedArray(const std::vector<Number> &init_vec, const Shape &shape_=Shape());
    /**
     * @todo 改为stack
     * @brief 构造函数，构造高维tensor
     * @param tensors 一堆形状一致的tensor
     * @param shape_ 这些tensor的形状，默认填充为{tensors.size()}
     * @attention 直接用initialized_list传参会出错, 受限于C++语法, 我无法更改此错误. 请只传vector类型的参数或在initialized_list前显式标注ShapedArray
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
    /**
     * @brief 把一个序列变成ShapedArray
     * @param args 一个序列，数字组成或者是ShapedArray组成
     * @return 一个ShapedArray
     * @example ShapedArray::fromSequence(1,2,3)
    */
    template<class ...Args>
    static ShapedArray fromSequence(Args... args) {return ShapedArray(std::vector{args...});}
    /**
     * @brief 把一个序列变成ShapedArray
     * @param args 一个序列，数字组成或者是ShapedArray组成
     * @return 一个ShapedArray
     * @example ShapedArray::fromSequence(1,2,3)
    */
    template<class ...Args>
    static ShapedArray fromSequence(size_t x, Args... args)
    {
        return ShapedArray(std::vector{
            static_cast<ShapedArray::Number>(x),
            (static_cast<ShapedArray::Number>(args))...
        });
    }
    /**
     * @brief 把一个序列变成ShapedArray
     * @param args 一个序列，数字组成或者是ShapedArray组成
     * @return 一个ShapedArray
     * @example ShapedArray::fromSequence(1,2,3)
    */
    template<class ...Args>
    static ShapedArray fromSequence(int x, Args... args)
    {
        return ShapedArray(std::vector{
            static_cast<ShapedArray::Number>(x),
            (static_cast<ShapedArray::Number>(args))...
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
    Number **atByNumbers(const std::vector<size_t> &index, Number **resBegin) const;
        
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
    ShapedRefArray where(std::function<bool(const Number &)> cond) const;
    
    /**
     * @brief 获取某些元素，同numpy.ndarray.__getitem__(:tuple)
     * @param indices 下标
     * @return 一个RefTensor
     * @todo 下标应该弄成可以broadcast成一致的就OK，并且没写sliced下标
     */
    template<class ...Args>
    ShapedRefArray at(const Args &... indices)
    {
        //虽不会检查，但要求indices都是ShapedArray/数
        size_t l = sizeof...(indices);
        size_t tmp=0;
        std::vector<ShapedArray> index {
            genIndex(indices, shape.dimSizeOf(tmp++))...
        }; 
        //计算下标广播后形状、结果形状
        Shape idxShape(index[0].shape);
        for(size_t i=1;i<l;++i)
            idxShape = Shape::broadcast(idxShape, index[i].shape);
        Shape resShape = idxShape + shape.sliced(l); 
        size_t resBufSize = resShape.bufSize();
        //想一想，为什么？（以下标的形状，拼好几块不知名物体。不知名物体的形状是shape.sliced(l)，因为不知名物体是arr[i1,i2,i3...,il]）
        
        Number **ptr = new Number*[resBufSize]; //Ref的内存条
#if DEBUG
        std::cout<<"ShapedRefArray Alloc @"<<static_cast<void*>(mArray)<<'['<<size<<']'<<std::endl;
#endif
        std::vector<size_t> numIdx (l); //一组数字下标组合的存放处
        //1.tuple的每一项是数
        if(idxShape.dimNumber()==0 && idxShape.bufSize()==1)
        {
            for(size_t i=0;i<l;++i)
                numIdx[i] = static_cast<size_t>(*index[i].mArray);
            atByNumbers(numIdx, ptr); //存到ptr里
            return ShapedRefArray(ptr, resShape);
        }
        //2.tuple的每一项是ShapedArray
        /*
            这里起，由于已经知道了resBufSize，我们把index看成一维数组就好了。然后把每一组下标算出来，放道一块连续的Number**中，转为Ref
            记index = {X_k} for k in range(l)
            res = [
                M.at({X_k[i] for k in range(l)})
                for i in range(idxShape.bufSize())
            ].reshape(resShape)
        */
        size_t idxCnt = idxShape.bufSize();
        Number **currentBegin = ptr;
        for(size_t i=0;i<idxCnt;++i)
        {
            //取每种下标组合
            for(size_t j=0, offset;j<l;++j){
                auto &idx = index[j];
                offset = Shape::offsetBeforeBroadcast(i, idxShape, idx.shape); //取出
                numIdx[j] = idx.mArray[offset];
            }
            currentBegin = atByNumbers(numIdx, currentBegin);//存储, 并往后挪
        }
        if(currentBegin-ptr != resBufSize)
            throw std::runtime_error("@Author From ShapedArray::at(...)\n\t:Wrong saved size");
        return ShapedRefArray(ptr, resShape);
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
    
    friend ShapedArray arange(Number be, Number en, Number step, const Shape &shape);
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
ShapedArray zeros(const Shape &shape);
ShapedArray ones(const Shape &shape);
ShapedArray arange(ShapedArray::Number be, ShapedArray::Number en, ShapedArray::Number step=1, const Shape &shape=Shape());

#define slist tryAI::ShapedArray::fromSequence

}
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