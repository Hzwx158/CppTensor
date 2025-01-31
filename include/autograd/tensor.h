#ifndef NUMCPP_AUTOGRAD_TENSOR_H
#define NUMCPP_AUTOGRAD_TENSOR_H
#include "../shaped/array.hpp"
#include "../utils/shared_var.hpp"
namespace numcpp::autograd{

class Function;
class CptGraphNode{
public:
    using _Self = SharedVar<CptGraphNode>;
    using TData = ShapedArray<double>;
private:
    TData data;
    Function *op;
public:
    _Self grad_node;
    bool require_grad;
    CptGraphNode(TData const &data_, bool require_grad_ = false, Function *op_=nullptr);
    CptGraphNode(CptGraphNode const &n)=delete;
    CptGraphNode(CptGraphNode &&n);
    ~CptGraphNode();
    void zero_grad();
    void backward();
    TData const &getData() const {return data;}
    H_OUTPUTABLE(CptGraphNode){
        return osm << obj.data;
    }
};

using Tensor = SharedVar<CptGraphNode>;

struct Function{
    FixedArray<Tensor> oprands;
    constexpr Function():oprands(){}
    Function(FixedArray<Tensor> const &oprands_) noexcept:oprands(oprands_){}
    Function(FixedArray<Tensor> &&oprands_) noexcept:oprands(std::move(oprands_)){}
    Function(Function const &)=delete;
    Function(Function &&f) noexcept :oprands(std::move(f.oprands)){}
    virtual ~Function(){}
    virtual Tensor forward(void *args=nullptr) const=0;
    Tensor operator()(void *args=nullptr) const {return this->forward(args);}
    virtual FixedArray<Tensor> backward(Tensor const &grad_output, void *args=nullptr) const = 0;
};



inline std::ostream &operator<<(std::ostream &osm, Tensor const &obj){return osm << obj.value();}
Tensor operator+(Tensor const &a, Tensor const &b);

namespace _operators{

class AddOp: public Function{
public:
    AddOp(Tensor const &op1, Tensor const &op2);
    ~AddOp(){}
    Tensor forward(void *args=nullptr) const override final;
    FixedArray<Tensor> backward(Tensor const &grad_output, void *args=nullptr) const override final;
};


}
}

namespace numcpp{
    using autograd::Tensor;
}

#endif