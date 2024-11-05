#include <chrono>
#include "shaped/array.hpp"
#include "./utils/thread_pool.hpp"

#define TEST_CPP 0
using namespace std::chrono;
using Time = decltype(system_clock::now());
/**
 * @brief 计算某个函数的执行时间
 * @param repeat_times 反复执行的次数
 * @param f 要执行的函数，写lambda即可
 */
template<class Functor>
void compute_running_time(size_t repeat_times, Functor &&f){
    numcpp::with(system_clock::now(), [&repeat_times, &f](Time){
        for(size_t i=0;i<repeat_times;++i)
            f();
    }, [](Time const&start_time){
        auto end_time = system_clock::now();
        auto duration = duration_cast<microseconds>(end_time-start_time).count();
        std::cout<<"----------------------\nusing time:"<<duration/1000.0<<"ms"<<std::endl;
    });
}
using numcpp::constant;
//进度:todo np.where如何声明实现并且好用
int main(){
#if !TEST_CPP
    auto mat1 = numcpp::arange<int>(0, 12, 1, {4,3});
    auto mat2 = numcpp::fill<int>(1, {4,3});
    constexpr bool output = 1;
    compute_running_time(1,[&](){
        std::cout<<numcpp::sin(constant::pi*0.5)<<std::endl;
    }); 
#else

#endif  
    return 0;
} 
/** 开发进度记录
 * √各个运算的简单实现
 * np.where的设计和实现
 */