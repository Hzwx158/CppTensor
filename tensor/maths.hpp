#pragma once
#include "./tensor.hpp"
namespace tryAI{

template<class Number, class Ret=Number>
Ret qPower(Number num, size_t n){
    Ret result=1;
    while(n){
        if(n&1)
            result*=num;
        num*=num;
        n>>=1;
    }
    return result;
}
}