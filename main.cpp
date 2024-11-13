#include "./shaped/array.hpp"
#include "./thread_pool.hpp"
#include "./autograd/tensor.hpp"
using numcpp::constant;
using numcpp::with;
using numcpp::compute_running_time;
using numcpp::thread::ThreadPool;
useStdIO;
numcpp::thread::OStream acout(cout);

void test_cpp(){
    acout<<1<<endl;
    for(int i=0;i<100000;++i){
        // acout<<'\0';
    }
}


#define TEST_CPP 1
int main(){
#if !TEST_CPP
    auto mat1 = numcpp::arange<int>(0, 10000, 1, {10,1000});
    auto mat2 = numcpp::fill<int>(0, {1000,20});
    constexpr bool output = 0;
    compute_running_time(1,[&](){
        mat1.matmul(mat2).print();
    });
#else
    // with(ThreadPool(4), [](auto &pool){
    //     for(int i=0;i<100;++i){
    //         pool.post(test_cpp);
    //     }
    // });
    // acout<<"END"<<endl;
    
    using numcpp::Var;
    auto x = Var::make(1.);
    printf("x over\n");
    auto y = Var::exp(x);
    printf("y over\n");
    auto z = Var::exp(y);
    printf("z over\n");
    with(std::vector<Var*>{x.get(), y.get(), z.get()}, [&](auto &&){
        Var::compute_gradient(z);
        printf("grad over\n");
        acout << x->gradient_node << numcpp::exp(constant::e+1) << endl;
    }, [](std::vector<Var*> const &var_list){
        for(auto &var:var_list)
            var->zero_grad();
    });
    
#endif

    return 0;
} 
