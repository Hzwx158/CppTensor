#include "shaped/array.hpp"
#include <chrono>

using namespace numcpp;
using namespace std::chrono;

//进度:todo np.where如何声明实现并且好用
int main(int argc, char **argv){
    auto start_time = system_clock::now();
    // begin
    
    auto arr = numcpp::arange(0, 12, 1, {2,2,3});
    arr.at(1, Slice(), Slice::to(-1)) = 100; //arr[1, :, 0:-1] = 100
    arr.print();
    numcpp::log(arr).print();

    // end
    auto end_time = system_clock::now();
    auto duration = duration_cast<microseconds>(end_time-start_time).count();
    std::cout<<"--------\nusing time:"<<duration/1000.0<<"ms"<<std::endl;
    return 0;
}
/** 开发进度记录
 * √各个运算的简单实现(差matmul、logical)
 * inf的运算问题?
 * np.where的设计和实现
 */