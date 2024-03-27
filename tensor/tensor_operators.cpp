#include "./tensor.hpp"
#include <cmath>
namespace tryAI{

//--------------------------运算符----------------------------------
#define ASMD_OPERATOR_DEFINE(operatorName)\
Tensor &Tensor::operator operatorName##= (const Tensor &obj){\
    *this=broadcast(*this, obj.shape);\
    Tensor tmp=broadcast(obj, shape);\
    const auto bufSize=shape.bufSize();\
    for(size_t i=0;i<bufSize;++i)\
        mArray[i] operatorName##= tmp.mArray[i];\
    return *this;\
}\
Tensor Tensor::operator operatorName(const Tensor &obj) const{\
    Tensor res=broadcast(*this, obj.shape);\
    Tensor tmp=broadcast(obj, res.shape);\
    auto bufSize=res.shape.bufSize();\
    for(size_t i=0;i<bufSize;++i)\
        res.mArray[i] operatorName##= tmp.mArray[i];\
    return res;\
}
ASMD_OPERATOR_DEFINE(+)
ASMD_OPERATOR_DEFINE(-)
ASMD_OPERATOR_DEFINE(*)
ASMD_OPERATOR_DEFINE(/)
#undef ASMD_OPERATR_DEFINE

Tensor Tensor::matmul(const Tensor &obj) const{
    const auto dimNum1 = shape.dimNumber();
    const auto dimNum2 = obj.shape.dimNumber();
    if(dimNum1>2 || dimNum2>2)
        throw std::runtime_error("From Tensor::matmul(const Tensor &):\n\ttoo many dims");
    if(dimNum1 == 2 && dimNum2 == 2){
        //矩阵乘矩阵，先检查形状合理否
        size_t m=shape.dimSizeOf(0), n=shape.dimSizeOf(1), l=obj.shape.dimSizeOf(1);
        if(n!=obj.shape.dimSizeOf(0))
            throw std::runtime_error("From Tensor::matmul(const Tensor &):\n\tNeed shape of(m,n)*(n,l)");
        Tensor res(0.,{m, l});
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
            throw std::runtime_error("From Tensor::matmul(const Tensor &):\n\tNeed same shape of 1-dim vector");
        Number res=0;
        const auto size=shape.bufSize();
        for(size_t i=0;i<size;++i)
            res += mArray[i]*obj.mArray[i];
        return Tensor(res);
    }
    //一个2一个1
    if(dimNum1==1){
        //搞成行向量*矩阵
        Tensor t1 = *this;
        t1.reshape({1, this->shape[0]});
        return t1.matmul(obj);
    }
    else{
        //搞成矩阵*列向量
        Tensor t1 = obj;
        t1.reshape({obj.shape[0],1});
        return matmul(t1);
    }
}



//res是obj的拷贝构造，mArray是res的内存条，bufSize是mArray的长度，i是遍历变量
#define TensorForEachFunc \
Tensor res(obj);\
Tensor::Number *mArray=res.mArray;\
size_t bufSize=res.shape.bufSize();\
for(size_t i=0;i<bufSize;++i)

Tensor exp(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=::exp(mArray[i]);
    }
    return res;
}
Tensor log(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=::log(mArray[i]);
    }
    return res;
}
Tensor sigmoid(const Tensor &obj){
    double x;
    TensorForEachFunc{
        x=::exp(mArray[i]);
        mArray[i]=x/(1+x);
    }
    return res;
}
Tensor sin(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=::sin(mArray[i]);
    }
    return res;
}
Tensor cos(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=::cos(mArray[i]);
    }
    return res;
}
Tensor tan(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=::tan(mArray[i]);
    }
    return res;
}
Tensor cot(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=1.0/::tan(mArray[i]);
    }
    return res;
}
Tensor sec(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=1.0/::cos(mArray[i]);
    }
    return res;
}
Tensor csc(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=1.0/::sin(mArray[i]);
    }
    return res;
}
Tensor asin(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=::asin(mArray[i]);
    }
    return res;
}
Tensor acos(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=::acos(mArray[i]);
    }
    return res;
}
Tensor atan(const Tensor &obj){
    TensorForEachFunc{
        mArray[i]=::atan(mArray[i]);
    }
    return res;
}

#undef TensorForEachFunc
}