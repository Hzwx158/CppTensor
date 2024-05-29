#ifndef CPPTENSOR_SHAPEDARRAY_SHAPE_H
#define CPPTENSOR_SHAPEDARRAY_SHAPE_H
#include "./simplevector.hpp"
#include "./pointer.hpp"
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

    static Shape broadcast(const std::vector<Shape> &shapes);

    /**
     * @brief 给定一个广播后形状中的偏移量，计算在广播前的Shape中，该坐标的偏移量; 时间复杂度 O(broShape.dimNumber())
     * @param broOffset 广播后的偏移量，要求在广播后的形状中合法
     * @param broShape 广播后的形状
     * @param srcShape 广播前的形状，要求要能变成广播后的形状
     * @return 返回广播前该偏移量对应的偏移量
     * @attention 不会检查srcShape和broShape是否符合广播机制，也不会对二者进行广播。请使用者调用时一定要保证srcShape可以广播到broShape中
    */
    static size_t offsetBeforeBroadcast(size_t broOffset, const Shape &broShape, const Shape &srcShape);
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
    /**
     * @brief 获取某多维坐标对应的起始偏移量，以及该坐标对应的子数组的含元素个数
     * @param index 多维坐标，如{1,0,-1}
     * @return 一个pair，first是该坐标对应的起始偏移量，second是该子数组含元素个数
    */
    std::pair<size_t,size_t> offsetOf(const std::vector<size_t> index) const;
    /**
     * @brief 缩减长度是1的维度
    */
    void squeeze();
    /**
     * @brief 缩减长度是1的维度
     * @param dim 维度数. 如果该维度长度非1, 会报错
    */
    void squeeze(size_t dim);

    Shape &operator=(const Shape &)=default;
    Shape &operator=(Shape &&)=default;
    bool operator==(const Shape &shape_) const;
    bool operator!=(const Shape &shape_) const {return !((*this)==shape_);}
    Shape operator+(const Shape &shape_) const;
    H_OUTPUTABLE(Shape)
};


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

/**
 * @brief 广播一个ShapedArray
 * @todo 时间复杂度有点复杂，我不会算，大概是sum(shape.product), 比O(bufsize)大?
 * @param src 数据源
 * @param srcShape 数据源的形状
 * @param anoShape 目标形状(可和srcShape广播的)
 * @param needBroadcast 是否需要计算anoShape和srcShape的广播结果. 默认为true
 * @return 一个pair, first是结果内存位置(一定是新分配的), second是结果形状
 * @attention 结果的内存条一定是新分配的, 记得delete
*/
template<class Number>
std::pair<Number *, Shape> broadcastShaped(const Number *src, const Shape &srcShape, const Shape &anoShape, bool needBroadcast=true)
{
    //1.需要形状非空
    if(srcShape.isEmpty()) 
        throw std::runtime_error("From broadcastShaped:\n\t<srcShape> is empty!");
    if(anoShape.isEmpty())
        throw std::runtime_error("From broadcastShaped:\n\t<anoShape> is empty!");
    const Shape &resShape = (needBroadcast?Shape::broadcast(srcShape, anoShape):anoShape);
    const auto srcBufSize=srcShape.bufSize();
    const auto srcDimNumber=srcShape.dimNumber();
    const auto resBufSize=resShape.bufSize();

    //不管怎么样，先拷贝到dest里总是必要的
    Number* dest=new Number[resBufSize];
#if DEBUG
    std::cout<<"Broadcast Alloc @"<<static_cast<void*>(dest)<<'['<<resBufSize<<']'<<std::endl;
#endif
    memcpy(dest, src, srcBufSize*sizeof(Number));

    //2.扩张结果显示和srcShape长度一致，只需返回resShape即可
    if(resBufSize==srcBufSize)
        return {dest, resShape}; //按照设定要返回new出来的内存
    
    //3.扩展结果与srcShape不一致，按规则扩张到resShape形状上
    const auto resDimNumber=resShape.dimNumber();
    const auto resProduct=resShape.getProductData();
    for(size_t rDim=0, srcCnt, copyCnt, srcStepSize, resStepSize, i, j, k;
        rDim<resDimNumber;++rDim
    ){//对每一维度(倒着)
        //计算需要复制几份(copyCnt)
        if(rDim<srcDimNumber)
            copyCnt=resShape.dimSizeOf(resDimNumber-1-rDim)/srcShape.dimSizeOf(srcDimNumber-1-rDim);
        else copyCnt=resShape.dimSizeOf(resDimNumber-1-rDim);
        if(copyCnt==1) //无需复制
            continue;
        //需要复制，说明srcDimSize是1, copyCnt==resShape.dimSizeOf(-rDim-1)
        //在原内存上的起点间隔(每个被复制的有几个元素), 没错就是resShape.stepSize[dim], 因为前面已经复制过了
        srcStepSize=resProduct[resDimNumber-rDim];//stepSizeOf(-rDim-1)
        //copyCnt=resStepSize/srcStepSize
        resStepSize=resProduct[resDimNumber-1-rDim]; //stepSizeOf(-rDim-2)，可能有[0]故不用stepSize函数
        //有几个需要被拷贝
        if(rDim<srcDimNumber)
            srcCnt=srcBufSize/srcShape.stepSizeOf(srcDimNumber-rDim-1);
        else srcCnt=1;
        for(i=0;i<srcCnt;++i){//倒着拷贝第i个
            for(j=(i+1==srcCnt);j<copyCnt;++j)//偏移正反无所谓
                memcpy(
                    dest+(srcCnt-i-1)*resStepSize+j*srcStepSize, 
                    dest+(srcCnt-i-1)*srcStepSize, 
                    srcStepSize*sizeof(Number)
                );
        }
    }
    return {dest, resShape};
}

}
#endif