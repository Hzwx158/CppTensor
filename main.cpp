#include "shaped/array.hpp"
#include <chrono>

using namespace numcpp;
using namespace std::chrono;

int main(int argc, char **argv){
    auto start_time = system_clock::now();
    ShapedArray arr({1,2,3,4,5,6,7,8,9,10,11,12}, {2,2,3});
    //进度:todo at的地址赋值、RefArray的输出
    arr.at(1, Slice(), Slice::to(-1)) = ShapedArray{100,99};
    // std::cout<<arr<<std::endl;
    auto end_time = system_clock::now();
    auto duration = duration_cast<microseconds>(end_time-start_time).count();
    std::cout<<"--------\nusing time:"<<duration/1000.0<<"ms"<<std::endl;
    return 0;
}
