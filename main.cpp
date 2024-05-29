#include "./shapedarray/shapedarray.hpp"
#include "./shapedarray/index.hpp"
//写代码用
useStdIO
using tryAI::ShapedArray;
using tryAI::Shape;
using tryAI::constant;
// using tryAI::autograd::gradient;
// using tryAI::autograd::Variable;
//! std::vector<size_t>被重载，尽量避免使用

//计时用
#include <chrono>
#include <fstream>
using namespace std::chrono;
std::ofstream ofs("./time.txt");



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
    
    auto t = tryAI::arange(1, 13, 1, {2,2,3});
    cout<<t<<endl;
    auto p = t.at(
        slist(slist(1),slist(0)),
        1, 
        slist(2,1)
    );
    cout<<p<<endl;
#else
    func({1,2,3,4});
#endif
}
    return 0;
}
/*
TODO : Slice怎么搞, at下标需要改成可broadcast成一致的即可

TODO : 把ShapedArray拼接的构造函数改为ShapedArray::stack；写一个vector of vector of vector... => ShapedArray的构造函数（必要的话）
TODO : 在tensor mul 常量怎么表示？Variable怎么设置？
*/