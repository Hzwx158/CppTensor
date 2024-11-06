#include "./shaped/array.hpp"
using numcpp::constant;
using numcpp::with;
using numcpp::compute_running_time;

void test_cpp(){
    for(int i=0;i<100000;++i){
        // printf("");
    }
}


#define TEST_CPP 0
int main(){
#if !TEST_CPP
    auto mat1 = numcpp::arange<int>(0, 10000, 1, {10,1000});
    auto mat2 = numcpp::fill<int>(0, {1000,20});
    constexpr bool output = 0;
    compute_running_time(1,[&](){
        mat1.matmul(mat2).print();
    });
#else
    compute_running_time(1, [](){
        numcpp::thread::ThreadPool pool(5);
        for(int i=0;i<100;++i){
            pool.post(test_cpp);
        }
    });
    compute_running_time(1, [](){
        for(int i=0;i<100;++i)
            test_cpp();
    });
    
#endif  
    return 0;
} 

/** 开发进度记录
 * √各个运算的简单实现
 * np.where的设计和实现
 */