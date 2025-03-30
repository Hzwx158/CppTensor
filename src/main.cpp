#include "shaped/array.hpp"
#include "utils/thread_pool.hpp"
// #include "./autograd/tensor.h"

#include "utils/context.h" 
// format 支持
#include <fmt/ostream.h>
template<class T>
struct fmt::formatter<numcpp::ShapedArray<T>>: fmt::ostream_formatter{};
// template<>
// struct fmt::formatter<numcpp::autograd::CptGraphNode>: fmt::ostream_formatter{};
// 使用一些名字
using numcpp::constant;
using numcpp::FixedArray;
using numcpp::with;
using numcpp::compute_running_time;
using numcpp::thread::ThreadPool;
useStdIO;
// numcpp::thread::OStream acout(cout);


void test_cpp(int i){
    cout << fmt::format("{}\n", i);
    for(int i=0;i<100000;++i){
        
    }
}

// using namespace numcpp::auto_del;
// struct TestList: public ContextLifeBase{
// 	int a;
//     TestList *nxt;
// 	TestList(int a_=0, TestList *nxt_=nullptr)
//         :ContextLifeBase(), a{a_}, nxt{nxt_}
//     {
//         fmt::println("({0})[{1}]-[{3}]", a, (void*)this, (void*)nxt);
//     }
// 	~TestList(){
// 		fmt::println("del {}", (void*)this);
// 	}
// };

#define TEST_CPP 0
int main(){
#if !TEST_CPP
    
    auto a = numcpp::fill(1., {3,4});
    auto b = numcpp::arange(0, 12, 1, {4,3});
    fmt::println("a:{}\nb:{}\na@b:{}", a,b, a.matmul(b));
    
#else

    


    // auto x = numcpp::Tensor({1.});
    // auto y = numcpp::Tensor({2.});
    // auto z = x + y;
    // cout << z;
    
#endif

    return 0;
} 
