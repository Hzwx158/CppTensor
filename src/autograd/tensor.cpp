#include "autograd/tensor.h"
namespace numcpp::autograd{

CptGraphNode::CptGraphNode(TData const &data_, bool require_grad_, Function *op_)
    :data(data_)
    ,op{op_}
    ,require_grad{require_grad_}
    ,grad_node(nullptr)
{}

CptGraphNode::CptGraphNode(CptGraphNode &&n)
    :data(std::move(n.data))
    ,op(n.op)
    ,require_grad{n.require_grad}
    ,grad_node(std::move(n.grad_node))
{
    n.require_grad = false;
    n.op = nullptr;
}
CptGraphNode::~CptGraphNode(){
    this->zero_grad();
    require_grad = false;
    delete op;
    op = nullptr;
}

void CptGraphNode::zero_grad(){
    this->grad_node = nullptr;
}

void CptGraphNode::backward(){
    if(grad_node.ptr) zero_grad();
    // TODO: 在这里遍历计算图进行backward
}

Tensor operator+(Tensor const &a, Tensor const &b){
    return (new _operators::AddOp(a,b))->forward();
}

}