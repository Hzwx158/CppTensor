#pragma once
#include<memory>
#include"./simplevector.hpp"
namespace tryAI{

class Shape{
public:
    using Vector=std::vector<size_t>;
    /**
     * @brief 计算广播的形状
     * @param shape1 形状1
     * @param shape2 形状2
     * @return 广播后的形状；如果changed是false则为空
    */
    static Shape broadcast(const Shape &shape1, const Shape &shape2);
private:
    Vector shape; 
    Vector product; 
    void generateProduct();
public:
    /**
     * @brief 构造函数
     * @param shape_ 形状、各维度数值
     * @attention shape_为{}也不是空shape，空shape是没存东西的Tensor的shape，shape_={}时认为是存了一个数
    */
    Shape(std::initializer_list<size_t> shape_);
    /**
     * @brief 直接构造
     * @param shape_ 直接存到shape
     * @param product_ 直接存到product
     * @attention shape、product都是倒着存的，谨慎！
    */
    Shape(Vector &&shape_, Vector &&product_):shape(std::move(shape_)),product(std::move(product_)){}
    /**
     * @brief 默认构造函数，为空shape
    */
    constexpr Shape():shape(),product(){}
    Shape(const Shape &)=default;
    Shape(Shape &&)=default;
    /**
     * @brief 返回第idx个维度的长度
     * @param idx 维度数
     * @return 第idx个维度的长度
    */
    size_t operator[](size_t idx) const {return shape[idx];}
    /**
     * @brief 返回第dim个维度的长度
     * @param dim 维度数
     * @return 第dim个维度的长度
    */
    size_t dimSizeOf(size_t dim) const {return shape[dim];}
    /**
     * @brief 返回在第dim个维度的步长
     * @param dim 维度数
     * @return 在第dim个维度的步长
    */
    size_t stepSizeOf(size_t dim) const;
    /**
     * @brief 返回有多少维度
     * @return 维度数
    */
    size_t dimNumber() const {return shape.size();}
    /**
     * @brief 返回这个形状一共有多少元素
     * @return 元素总数
    */
    size_t bufSize() const {return product[0];}
    /**
     * @brief 是否为空形状(没存东西的Tensor的shape)
     * @return 空为true
    */
    bool isEmpty() const {return product.empty();}
    /**
     * @brief 清空
     */
    void clear(){shape.clear();product.clear();}
    /**
     * @brief 截取
     * @param be 起点(含)
     * @param en 终点(不含)
     * @return [be,en)的Shape
    */
    Shape sliced(size_t be, size_t en) const;
    /**
     * @brief 截取
     * @param be 起点(含)
     * @return [be,-1]的Shape
    */
    Shape sliced(size_t be) const;
    /**
     * @brief 获取shape的起始地址
     * @attention 直接操作指针会引起不必要的麻烦，慎用
    */
    size_t *getShapeData() const {return shape.data();}
    /**
     * @brief 获取product的起始地址
     * @attention 直接操作指针会引起不必要的麻烦，慎用
    */
    size_t *getProductData() const {return product.data();}
    Shape &operator=(const Shape &)=default;
    Shape &operator=(Shape &&)=default;
    bool operator==(const Shape &shape_) const;
    bool operator!=(const Shape &shape_) const {return !((*this)==shape_);}
    H_OUTPUTABLE(Shape)
};


//下标辅助
template<class T>
class list{
private:
    std::vector<T> v;
public:
    list(const std::vector<T> &vec):v(vec){}
    list(std::vector<T> &&vec):v(std::move(vec)){}
    list(std::initializer_list<T> l):v(l){}
    list(const list &obj):v(obj.v){}
    list(list &&obj):v(std::move(obj.v)){}
    operator std::vector<T>& (){return v;}
    const std::vector<T> &getVector() const {return v;}
};
#define PARTIAL_SPECIALIZE_LIST(orgClass, saveClass)\
template<>\
class list<orgClass>{\
private:\
    std::vector<saveClass> v;\
public:\
    list(const std::vector<orgClass> &vec):v(vec.size(),0){\
        for(size_t i=0;i<vec.size();++i)\
            v[i]=vec[i];\
    }\
    list(std::initializer_list<orgClass> l):v(l.size(),0){\
        auto p=l.begin();\
        for(size_t i=0;i<l.size();++i)\
            v[i]=*(p+i);\
    }\
    list(const list &obj):v(obj.v){}\
    list(list &&obj):v(std::move(obj.v)){}\
    operator std::vector<saveClass>& (){return v;}\
    const std::vector<saveClass> &getVector() const {return v;}\
};
PARTIAL_SPECIALIZE_LIST(int, size_t)
PARTIAL_SPECIALIZE_LIST(short, size_t)
PARTIAL_SPECIALIZE_LIST(unsigned int, size_t)
PARTIAL_SPECIALIZE_LIST(unsigned short, size_t)


}