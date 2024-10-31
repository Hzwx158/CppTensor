#ifndef NUMCPP_SHAPED_ARRAY_HPP
#define NUMCPP_SHAPED_ARRAY_HPP
#include "./shape.hpp"
#include "../utils/pointer.hpp"
namespace numcpp{
// #define atCond(...) \
// where([&](auto &num){ return __VA_ARGS__; })

// #define $$(X) .matmul(X)
template<class DType>
class ShapedArray;
template<class T>
struct Is_ShapedArray{
    constexpr static bool value = false;
};
template<class T>
struct Is_ShapedArray<ShapedArray<T>>{
    constexpr static bool value = true;
};

template<class T>
constexpr bool is_ShapedArray_v = Is_ShapedArray<T>::value;

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
    /**
     * @brief 清除内存
    */
    void clear();
public://Memory
    /**
     * @brief 空构造函数
    */
    constexpr ShapedArray():mArray(nullptr), shape(){}
    /**
     * @brief 构造函数
     * @param num 数字，初始化值
    */
    ShapedArray(const DType &num);

    explicit ShapedArray(DType *&&ptr, const Shape &shape_)
        :mArray(std::move(ptr)), shape(shape_){}
    explicit ShapedArray(DType *&&ptr, Shape &&shape_)
        :mArray(std::move(ptr)), shape(std::move(shape_)){}
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
public:
    /**
     * @brief 输出
     * @param osm 所用输出流(默认cout)
    */
    void print(std::ostream &osm=std::cout) const {osm<<*this<<std::endl;}
    // 标准输出流输出
    H_OUTPUTABLE(ShapedArray<DType>){
        if constexpr(std::is_pointer_v<DType>)
            printShaped<DType>(obj.mArray, obj.shape, osm, [](std::ostream &osm_, DType const &ele){
                _output_number(osm_, *ele);
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
    ShapedArray<T> to() const;
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
     * @brief 获取数据条偏移地址
     * @param offset 偏移量
     * @return 数据地址
     */
    DType *data(size_t offset=0) const {return mArray+offset;}
    
    /**
     * @brief 获取某些元素
     * @param indices 下标
     * @return 一个ShapedArray<DType>
     */
    template<class ...Args>
    ShapedArray<DType> at(const Args &... indices) const;
    /**
     * @brief 获取某些元素，同numpy.ndarray.__getitem__
     * @param indices 下标
     * @return 一个ShapedArray<DType*>
     * @example python: a[[1,2], 3, 3:4]
     * c++: a.at(IDX{1,2}, 3, Slice(3,4))
     */
    template<class ...Args>
    ShapedArray<DType*> at(const Args &... indices);
    /**
     * @brief 对某个位置的元素干某事
     * @param func 元素的操作函数，只接收一个参数
     * @param index 元素的坐标
     */
    template<class Functor>
    void apply_on(Functor &&func, const FixedArray<size_t> &index);
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
#define OP_DCL_CODE(opStr, opName)\
    template<class T>\
    ShapedArray<DType> &operator opStr##=(const ShapedArray<T> &obj);\
    template<class T>\
    ShapedArray<DType> &operator opStr##=(const T &num);\
    template<class T>\
    ShapedArray<op_ret_t<EOperation::opName, DType, T>> operator opStr(const ShapedArray<T> &obj) const;\
    template<class T, class useless = std::enable_if_t<!is_ShapedArray_v<T>>>\
    ShapedArray<op_ret_t<EOperation::opName, DType, T>> operator opStr(const T &obj) const;

    OP_DCL_CODE(+, ADD)
    OP_DCL_CODE(-, SUB)
    OP_DCL_CODE(*, MUL)
    OP_DCL_CODE(/, DIV)
    OP_DCL_CODE(%, MOL)
#undef OP_DCL_CODE
#define LOGICAL_OP_DCL_CODE(opStr)\
    template<class T>\
    ShapedArray<bool> operator opStr (const ShapedArray<T> &obj) const;\
    template<class T>\
    ShapedArray<bool> operator opStr (const T &obj) const;

    LOGICAL_OP_DCL_CODE(>)
    LOGICAL_OP_DCL_CODE(>=)
    LOGICAL_OP_DCL_CODE(<)
    LOGICAL_OP_DCL_CODE(<=)
    LOGICAL_OP_DCL_CODE(==)
    LOGICAL_OP_DCL_CODE(!=)
    
#undef LOGICAL_OP_DCL_CODE
    template<class T>
    ShapedArray<int> compare(const ShapedArray<T> &obj) const;
    template<class T>
    ShapedArray<int> compare(const T &obj) const;

    template<class T>
    ShapedArray<op_ret_t<EOperation::MUL, DType, T>> matmul(const ShapedArray<T> &obj) const;

    ShapedArray<double> exp() const;
    ShapedArray<double> log() const;
    ShapedArray<double> sin() const;
    ShapedArray<double> cos() const;
    ShapedArray<double> tan() const;
    ShapedArray<double> asin() const;
    ShapedArray<double> acos() const;
    ShapedArray<double> atan() const;
    ShapedArray<double> sec() const;
    ShapedArray<double> csc() const;
    ShapedArray<double> cot() const;
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
}
//---------------------------------------------------实现---------------------------------------------------

#include "./_array_arr.hpp"
#include "./_array_idx.hpp"
#include "./_array_op.hpp"
#endif