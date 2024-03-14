#pragma once
#include"./simplevector.hpp"
#include <functional>
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

#define list std::vector

/**
 * @brief 输出有形状的数组的函数
 * @param arr 输出内存的首地址
 * @param shape 数组形状
 * @param osm 所用输出流，默认std::cout
 * @param output 输出每个元素的函数，默认osm<<ele
 * @todo 输出到控制台时，时间性能低于python, 甚是奇怪
*/
template<class T>
void printShaped(
    const T *arr, const Shape &shape, 
    std::ostream &osm=std::cout,
    const std::function<void(std::ostream&,const T &)> &output
    =[](std::ostream &osm, const T &ele){osm<<ele;}
) {
    if(shape.isEmpty()){
        //空数组
        osm<<"[]";
        return;
    }
    const auto shapeDimNumber=shape.dimNumber();
    if(!shapeDimNumber){
        //数字
        output(osm, *arr);
        return;
    }
    const auto shapeProduct=shape.getProductData();
    const auto shapeBufSize=shape.bufSize();
    for(size_t pos=0, i, braCnt=shapeDimNumber;pos<shapeBufSize;++pos){
        for(i=0;braCnt && i<shapeDimNumber-braCnt;++i)
            osm<<' ';
        for(i=0;i<braCnt;++i)
            osm<<'[';
        output(osm, *(arr+pos));
        for(i=0, braCnt=0; i<shapeDimNumber; ++i){
            if((pos+1)%shapeProduct[shapeDimNumber-i-1])
                break;
            ++braCnt;
            osm<<']';
        }
        if(pos+1!=shapeBufSize){
            osm<<", ";
            if(braCnt>=2)
                osm<<"\n\n";
            else if(braCnt)
                osm<<"\n";
        }
    }
}

}