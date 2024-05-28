#include "./shapedarray.hpp"
#include <cmath>
namespace tryAI{

//--------------------------运算符----------------------------------
#define ASMD_OPERATOR_DEFINE(operatorName)\
ShapedArray &ShapedArray::operator operatorName##= (const ShapedArray &obj){\
    Shape resShape = Shape::broadcast(shape, obj.shape);\
    size_t l = resShape.bufSize();\
    NumberPtr tmp = new Number[l];\
    if(DEBUG)\
        std::cout<<"Pointer Alloc @"<<static_cast<void*>(tmp)<<'['<<l<<"]\n";\
    for(size_t i=0, offset1, offset2;i<l;++i)\
    {\
        offset1 = Shape::offsetBeforeBroadcast(i, resShape, shape);\
        offset2 = Shape::offsetBeforeBroadcast(i, resShape, obj.shape);\
        tmp[i] = (mArray[offset1]) operatorName (obj.mArray[offset2]);\
    }\
    mArray = std::move(tmp);\
    shape = std::move(resShape);\
    return *this;\
}\
ShapedArray ShapedArray::operator operatorName(const ShapedArray &obj) const{\
    Shape resShape = Shape::broadcast(shape, obj.shape);\
    size_t l = resShape.bufSize();\
    NumberPtr tmp = new Number[l];\
    if(DEBUG)\
        std::cout<<"Pointer Alloc @"<<static_cast<void*>(tmp)<<'['<<l<<"]\n";\
    for(size_t i=0, offset1, offset2;i<l;++i)\
    {\
        offset1 = Shape::offsetBeforeBroadcast(i, resShape, shape);\
        offset2 = Shape::offsetBeforeBroadcast(i, resShape, obj.shape);\
        tmp[i] = (mArray[offset1]) operatorName (obj.mArray[offset2]);\
    }\
    return ShapedArray(std::move(tmp), std::move(resShape));\
}\
void ShapedRefArray::operator operatorName##= (const ShapedArray &obj){\
    Shape resShape = Shape::broadcast(shape, obj.shape);\
    if(resShape!=shape)\
        throw std::runtime_error("From ShapedRefArray::some op=\n\t<obj.shape> can't broadcast to <shape>");\
    size_t l = resShape.bufSize();\
    Number *ptr = new Number[l];\
    for(size_t i=0, offset;i<l;++i)\
    {\
        offset = Shape::offsetBeforeBroadcast(i, resShape, obj.shape);\
        ptr[i] = (*mArray[i]) operatorName (obj.mArray[offset]);\
    }\
    for(size_t i=0; i<l;++i)\
        (*mArray[i]) = ptr[i];\
    delete[] ptr;\
}
/*
    吐槽: numpy的+=做得是愚蠢至极, 同时浪费空间(N)和时间(2N), 如果省到空间0时间N, 则会导致同一位置被加两遍.
    example: a=np.array([1,2,3,4,5,6,7,8]).reshape((2,2,2))
    a[[1,1],[1,1]]+=[[1,2],[3,4]]
    输出a会得到
    [[[1,2],
      [3,4]],
     [[5,6],
      [10,12]]]
    也就是说, 只有最后一次加这个位置的信息被记录. 这怎么做到呢? 只能把a+=b变成a=a+b才能做到, 所以这同时浪费空间与时间.
    如果直接在原数组上+=, 则会导致同一位置被操作两遍. 为保证与numpy尽可能一致, 哪怕浪费时间也得按照numpy的浪费模式实现.
    @date 2024.5.28
    @author hzwx158 
*/

ASMD_OPERATOR_DEFINE(+)
ASMD_OPERATOR_DEFINE(-)
ASMD_OPERATOR_DEFINE(*)
ASMD_OPERATOR_DEFINE(/)
#undef ASMD_OPERATR_DEFINE
bool ShapedArray::isEqualTo(const ShapedArray &obj) const{
    if(shape!=obj.shape) return false;
    auto size = shape.bufSize()*sizeof(size_t);
    return !memcmp(mArray, obj.mArray, size);
}

ShapedArray ShapedArray::matmul(const ShapedArray &obj) const{
    const auto dimNum1 = shape.dimNumber();
    const auto dimNum2 = obj.shape.dimNumber();
    if(dimNum1>2 || dimNum2>2)
        throw std::runtime_error("From ShapedArray::matmul(const ShapedArray &):\n\ttoo many dims");
    if(dimNum1 == 2 && dimNum2 == 2){
        //矩阵乘矩阵，先检查形状合理否
        size_t m=shape.dimSizeOf(0), n=shape.dimSizeOf(1), l=obj.shape.dimSizeOf(1);
        if(n!=obj.shape.dimSizeOf(0))
            throw std::runtime_error("From ShapedArray::matmul(const ShapedArray &):\n\tNeed shape of(m,n)*(n,l)");
        ShapedArray res(0.,{m, l});
        auto &oMa=obj.mArray;
        auto &rMa=res.mArray;
        size_t id;
        for(size_t i=0;i<m;++i)
            for(size_t j=0;j<l;++j){
                id = res.shape.offsetOf({i,j}).first;
                for(size_t k=0;k<n;++k){
                    rMa[id] += 
                        mArray[shape.offsetOf({i,k}).first]*
                        oMa[obj.shape.offsetOf({k,j}).first];
                }
            }
        return res;
    }
    if(dimNum1 != 2 && dimNum2 != 2){
        //向量内积，需要形状一致
        if(shape!=obj.shape)
            throw std::runtime_error("From ShapedArray::matmul(const ShapedArray &):\n\tNeed same shape of 1-dim vector");
        Number res=0;
        const auto size=shape.bufSize();
        for(size_t i=0;i<size;++i)
            res += mArray[i]*obj.mArray[i];
        return ShapedArray(res);
    }
    //一个2一个1
    if(dimNum1==1){
        //搞成行向量*矩阵
        ShapedArray t1 = *this;
        t1.reshape({1, this->shape[0]});
        return t1.matmul(obj);
    }
    else{
        //搞成矩阵*列向量
        ShapedArray t1 = obj;
        t1.reshape({obj.shape[0],1});
        return matmul(t1);
    }
}



//res是obj的拷贝构造，mArray是res的内存条，bufSize是mArray的长度，i是遍历变量
#define TensorForEachFunc \
ShapedArray res(obj);\
ShapedArray::Number *mArray=res.mArray;\
size_t bufSize=res.shape.bufSize();\
for(size_t i=0;i<bufSize;++i)

ShapedArray exp(const ShapedArray &obj){
    TensorForEachFunc{
        mArray[i]=::exp(mArray[i]);
    }
    return res;
}
ShapedArray log(const ShapedArray &obj){
    TensorForEachFunc{
        mArray[i]=::log(mArray[i]);
    }
    return res;
}
ShapedArray sigmoid(const ShapedArray &obj){
    double x;
    TensorForEachFunc{
        x=::exp(mArray[i]);
        mArray[i]=x/(1+x);
    }
    return res;
}
ShapedArray sin(const ShapedArray &obj){
    TensorForEachFunc{
        mArray[i]=::sin(mArray[i]);
    }
    return res;
}
ShapedArray cos(const ShapedArray &obj){
    TensorForEachFunc{
        mArray[i]=::cos(mArray[i]);
    }
    return res;
}
ShapedArray tan(const ShapedArray &obj){
    TensorForEachFunc{
        mArray[i]=::tan(mArray[i]);
    }
    return res;
}
ShapedArray cot(const ShapedArray &obj){
    TensorForEachFunc{
        mArray[i]=1.0/::tan(mArray[i]);
    }
    return res;
}
ShapedArray sec(const ShapedArray &obj){
    TensorForEachFunc{
        mArray[i]=1.0/::cos(mArray[i]);
    }
    return res;
}
ShapedArray csc(const ShapedArray &obj){
    TensorForEachFunc{
        mArray[i]=1.0/::sin(mArray[i]);
    }
    return res;
}
ShapedArray asin(const ShapedArray &obj){
    TensorForEachFunc{
        mArray[i]=::asin(mArray[i]);
    }
    return res;
}
ShapedArray acos(const ShapedArray &obj){
    TensorForEachFunc{
        mArray[i]=::acos(mArray[i]);
    }
    return res;
}
ShapedArray atan(const ShapedArray &obj){
    TensorForEachFunc{
        mArray[i]=::atan(mArray[i]);
    }
    return res;
}

#undef TensorForEachFunc
}