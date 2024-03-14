#include "./tensor/tensor.hpp"
//写代码用
useStdIO
using tryAI::Tensor;
using tryAI::Shape;
using tryAI::constant;
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
    auto t=Tensor::arange(1,13,1,{2,2,3})*constant::pi/2;
    Tensor p = tryAI::sin(t.at(list{0,1},list{0,1}));
    p.foreach([](auto &n){
        if(abs(n)<1e-10){
            n=0;
        }
    });
    cout<<p<<endl;
#else
#endif
// }
    return 0;
}