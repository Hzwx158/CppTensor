#include "./shapedarray/shapedarray.hpp"
#include "./shapedarray/index.hpp"
//写代码用
useStdIO
using tryAI::ShapedArray;
using tryAI::Shape;
using tryAI::constant;
using tryAI::Slice;
using tryAI::SimpleShapedArray;
// using tryAI::autograd::gradient;
// using tryAI::autograd::Variable;
//! std::vector<size_t>被重载，尽量避免使用

//计时用
#include <chrono>
#include <fstream>
using namespace std::chrono;
std::ofstream ofs("./time.txt");

template<class ...Args>
void func(Args... args){
    size_t k=0;
    std::vector<size_t> res{
        (args, k++)...
    };
    cout<<res<<endl;
}



int main()
{
//for(int _=0;_<10000;++_)
{
    decltype(system_clock::now()) __time;
    tryAI::Releaser rel(__time,[](auto &beginTime){
        auto endTime=system_clock::now();
        auto duration=duration_cast<microseconds>(endTime-beginTime);
        cout<<"\n----------------------\n"
            "using time of "<<duration.count()/1000.0<<"ms"
            "\n----------------------\n";
        //ofs<<double(duration.count())<<',';
    });
    __time=system_clock::now();
#if 1
    // auto x=Variable("x");
    // auto y=Variable("y");
    // auto z = x->add(x->mul(y));
    // cout<<"z="<<*z<<endl;
    // auto res = gradient(z, {x,y});
    // cout<<*res[0]<<'\n'<<*res[1]<<endl;
    // x->freeAllAnnoymous();
    // delete x;
    // delete y;
    
    ShapedArray t{1,2,3,4,5,6,7,8,9,10,11,12};
    t.reshape({2,2,3});
    cout<<t<<endl;
    auto p = t.at(Slice(1,2)); //t[1:2]
    std::cout<<p<<endl;

#else
    func(1,2,3,49,0);
#endif
}
    return 0;
}
/*
TODO : Slice如何被设置为单个的然后被broadcast（SimpleShapedArray已实现）？

TODO : 把ShapedArray拼接的构造函数改为ShapedArray::stack；写一个vector of vector of vector... => ShapedArray的构造函数（必要的话）
TODO : 在tensor mul 常量怎么表示？Variable怎么设置？
*/