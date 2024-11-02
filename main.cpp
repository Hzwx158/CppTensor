#include "shaped/array.hpp"
#include <chrono>

using namespace numcpp;
using namespace std::chrono;
using Time = decltype(system_clock::now());
/**
 * @brief 计算某个函数的执行时间
 * @param repeat_times 反复执行的次数
 * @param f 要执行的函数，写lambda即可
 */
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
int main(){
    auto mat1 = numcpp::arange<int>(0, 12, 1, {4,3});
    auto mat2 = numcpp::fill<int>(1, {4,3});
    compute_running_time(1,[&](){
        constexpr bool output = 1;
        auto mat3 = 0 - (mat1 & mat2);
        if constexpr(output){
            std::cout<<"mat1:\n"<<mat1<<"\nmat2:\n"<<mat2<<"\nmat3:\n"<<mat3<<std::endl;
        }
    }); 
    return 0;}
/** 开发进度记录
 * √各个运算的简单实现
 * np.where的设计和实现
 */