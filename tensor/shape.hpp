#pragma once
#include<vector>
#include<iostream>
#include<memory>
#include"base.h"
namespace tryAI{
class Shape{
public:
    using Vector=std::vector<size_t>;
private:
    Vector shape;
    Vector product; // product[0]=shape[1]*shape[2]...; product[shape.size()]=shape[0]*... 
    void generateProduct();
public:
    /**
     * @brief 构造函数
     * @param shape_ 形状、各维度数值
     * @attention shape_为{}也不是空shape，空shape是没存东西的Tensor的shape，shape_={}时认为是存了一个数
    */
    Shape(const Vector &shape_):shape(shape_), product(){generateProduct();}
    /**
     * @brief 构造函数
     * @param shape_ 形状、各维度数值
     * @attention shape_为{}也不是空shape，空shape是没存东西的Tensor的shape，shape_={}时认为是存了一个数
    */
    Shape(Vector &&shape_):shape(std::move(shape_)), product(){generateProduct();}
    /**
     * @brief 构造函数
     * @param shape_ 形状、各维度数值
     * @attention shape_为{}也不是空shape，空shape是没存东西的Tensor的shape，shape_={}时认为是存了一个数
    */
    Shape(std::initializer_list<size_t> shape_):shape(shape_), product(){generateProduct();}
    /**
     * @brief 默认构造函数，为空shape
    */
    Shape():shape(),product(){}
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
    size_t bufSize() const {return product[shape.size()];}
    /**
     * @brief 是否为空形状(没存东西的Tensor的shape)
     * @return 空为true
    */
    bool isEmpty() const {return product.empty();}
    /**
     * @brief 清空
     */
    void clear(){shape={};product={};}
    /**
     * @brief 广播到shape_的形状
     * @param shape_ 另一个shape
     * @return 广播后的Shape
    */
    Shape broadcastedShape(const Shape &shape_) const;
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

    const Vector &asVector() const {return shape;}
    Shape &operator=(const Shape &)=default;
    Shape &operator=(Shape &&)=default;
    bool operator==(const Shape &shape_) const;
    bool operator!=(const Shape &shape_) const {return !((*this)==shape_);}
    H_OUTPUTABLE(Shape)
};
/**
 * @brief 判断一个数是否是合理下标、并转成[0, bufSize)的下标值
 * @param idx 输入下标值
 * @param bufSize 数组长度
 * @param res 转成合理下标值的存储位置
 * @return 如果不合理，返回nullptr；合理返回res
 */
size_t *toBoundedIdx(size_t idx, size_t bufSize, size_t *res);
}