#include "./autograd/node_op.hpp"
//写代码用
useStdIO
using tryAI::Tensor;
using tryAI::Shape;
using tryAI::constant;
using tryAI::autograd::gradient;
using tryAI::autograd::Variable;
//! std::vector<size_t>被重载，尽量避免使用

//计时用
#include <chrono>
#include <fstream>
using namespace std::chrono;
std::ofstream ofs("./time.txt");

int main()
{
//for(int _=0;_<10000;++_){
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
    auto x=*Variable("x");
    auto y=*Variable("y");
    auto z = x+*(x*y);
    cout<<"z="<<*z<<endl;
    auto res = gradient(z, {&x,&y});
    cout<<*res[0]<<'\n'<<*res[1]<<endl;
    x.freeAllAnnoymous();
#else
    std::string *p=new std::string("a");
    std::string_view s=*p;
    delete p;
    cout<<s<<endl;
#endif
// }
    return 0;
}