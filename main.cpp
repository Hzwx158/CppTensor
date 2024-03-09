#include "./tensor/tensor.hpp"
useStdIO
using tryAI::Tensor;
using tryAI::Shape;
using tryAI::list;
#include <chrono>
#include <fstream>
using namespace std::chrono;
// std::vector<size_t>被重载，尽量避免使用

class T{
public:
    int b;
    T(int a):b(a){}
    void clear() {printf("aaa cleared\n");}
};
void pushTest(std::vector<T> &v, size_t n){
    for(size_t i=0;i<n;++i)
        v.push_back(T(10));
}
std::ofstream ofs("./time.txt");
int main()
{
//for(int _=0;_<10000;++_){
    decltype(system_clock::now()) __time;
    tryAI::Releaser rel(__time,[](auto &beginTime){
        auto endTime=system_clock::now();
        auto duration=duration_cast<microseconds>(endTime-beginTime);
        cout<<"using time of "<<duration.count()/1000.0<<"ms\n";
        //ofs<<double(duration.count())<<',';
    });
    __time=system_clock::now();
#if 1
    auto t=new Tensor({1,2,3,4,5,6,7,8,9,10,11,12},{2,2,3});
    //auto t=new Tensor({1,2,3,4,5,6,7,8},{1,2,2,1,2});
    auto p=(*t)/Tensor({0,1,2},{1,1,3});
    cout<<p<<endl;
    delete t;
#else
    int b=0;
    cout<<1.0/b<<endl;
#endif
// }
    return 0;
}