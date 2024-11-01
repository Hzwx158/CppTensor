#include "shaped/array.hpp"
#include <chrono>

using namespace numcpp;
using namespace std::chrono;
using Time = decltype(system_clock::now());
template<class Functor>
void compute_running_time(size_t repeat_times, Functor &&f){
    with(system_clock::now(), [&repeat_times, &f](Time){
        for(size_t i=0;i<repeat_times;++i)
            f();
    }, [](Time const&start_time){
        auto end_time = system_clock::now();
        auto duration = duration_cast<microseconds>(end_time-start_time).count();
        std::cout<<"----------------------\nusing time:"<<duration/1000.0<<"ms"<<std::endl;
    });
}

//进度:todo np.where如何声明实现并且好用
int main(int argc, char **argv){compute_running_time(1,[](){

    auto mat1 = numcpp::arange(0, 16,1,{4,4});
    auto mat2 = numcpp::fill(1, {4, 1});
    mat1.matmul(mat2).print();


}); return 0;}
/** 开发进度记录
 * √各个运算的简单实现
 * inf的运算问题?
 * np.where的设计和实现
 */