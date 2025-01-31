#include "autograd/tensor.h"
namespace numcpp::autograd::_operators{
static FixedArray<Tensor> only_require_grad(FixedArray<Tensor> const &arr){
    FixedArray<Tensor> res = arr;
    size_t k=0;
    for(auto &t:res)
        if(t->require_grad)
            res[k++] = std::move(t);
    return res.shrink_to(k);
}
AddOp::AddOp(Tensor const &op1, Tensor const &op2)
    :Function({op1, op2})
{}
Tensor AddOp::forward(void *) const {
    return Tensor(
        oprands[0]->getData() + oprands[1]->getData(), 
        true, (Function*)this
    );
}
FixedArray<Tensor> AddOp::backward(Tensor const &grad_output, void *) const{
    // switch(oprands.size()){
    // case 0:
    //     return {};
    // case 1:
    //     return {grad_output};
    // case 2:
    //     return {grad_output, grad_output};
    // default:
    //     throw Error::author(__FILE__, __func__, "wrong size of oprands");
    // }
    return {grad_output, grad_output};
}

}