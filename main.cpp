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

void test(const std::vector<ShapedArray> &tensors){
    for(auto &mem:tensors)
        mem.print();
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
    
    cout<<Shape();
    // t.print();
#else

#endif
}
    return 0;
}
/*
TODO : Slice怎么搞, at下标需要改成可broadcast成一致的即可; 是否需要Index类？

以ShapedArray作为Index的类型：构造函数可以写成ShapedArray t{1,2,3}, 所以可以直接传参{1,2,3}

TODO : 在tensor mul 常量怎么表示？Variable怎么设置？
*/