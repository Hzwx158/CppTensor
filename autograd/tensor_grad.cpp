#include "tensor.hpp"
namespace numcpp{
extern int OP_NODE_CNT[18];
#define CHILD_GRAD_FUNC_DEF(opName)\
static Var *_##opName##_child_grad(Var *self, int i)

CHILD_GRAD_FUNC_DEF(add){return self->gradiant_node;}
CHILD_GRAD_FUNC_DEF(addConst){return self->gradiant_node;}
CHILD_GRAD_FUNC_DEF(sub){return i ? self->gradiant_node->mul(-1): self->gradiant_node;}
CHILD_GRAD_FUNC_DEF(subConst){return self->gradiant_node;}
CHILD_GRAD_FUNC_DEF(rsubConst){return self->gradiant_node->mul(-1);}
CHILD_GRAD_FUNC_DEF(mul){
    if(i==0)
        return self->gradiant_node->mul(self->getChild(1));
    else return self->gradiant_node->mul(self->getChild(0));
}
CHILD_GRAD_FUNC_DEF(mulConst){return self->gradiant_node->mul(*self->getConstAttr());}
CHILD_GRAD_FUNC_DEF(div){return nullptr;}
CHILD_GRAD_FUNC_DEF(divConst){return nullptr;}
CHILD_GRAD_FUNC_DEF(rdivConst){return nullptr;}
CHILD_GRAD_FUNC_DEF(powConst){return nullptr;}
CHILD_GRAD_FUNC_DEF(rpowConst){return nullptr;}
CHILD_GRAD_FUNC_DEF(exp){
    return self->gradiant_node->mul(self);
}
CHILD_GRAD_FUNC_DEF(log){return nullptr;}
CHILD_GRAD_FUNC_DEF(sin){return nullptr;}
CHILD_GRAD_FUNC_DEF(cos){return nullptr;}
CHILD_GRAD_FUNC_DEF(tan){return nullptr;}


#undef CHILD_GRAD_FUNC_DEF
typedef Var *(*TChildGradFunc)(Var *self, int i);

constexpr TChildGradFunc CHILD_GRAD_FUNC[]={
    [Var::PLACEHOLDER_OP] = nullptr,
    [Var::ADD_OP] = &_add_child_grad,
    [Var::ADD_CONST_OP] = &_addConst_child_grad,
    [Var::SUB_OP] = &_sub_child_grad,
    [Var::SUB_CONST_OP] = &_subConst_child_grad,
    [Var::RSUB_CONST_OP] = &_rsubConst_child_grad,
    [Var::MUL_OP] = &_mul_child_grad,
    [Var::MUL_CONST_OP] = &_mulConst_child_grad,
    [Var::DIV_OP] = &_div_child_grad,
    [Var::DIV_CONST_OP] = &_divConst_child_grad,
    [Var::RDIV_CONST_OP] = &_rdivConst_child_grad,
    [Var::POW_CONST_OP] = &_powConst_child_grad,
    [Var::RPOW_CONST_OP] = &_rpowConst_child_grad,
    [Var::EXP_OP] = &_exp_child_grad,
    [Var::LOG_OP] = &_log_child_grad,
    [Var::SIN_OP] = &_sin_child_grad,
    [Var::COS_OP] = &_cos_child_grad,
    [Var::TAN_OP] = &_tan_child_grad
};
Var *Var::gradOfChild(Var *self, int i)
{
    if(self->op_type==PLACEHOLDER_OP)
        return make(Value(0), false);
    return CHILD_GRAD_FUNC[self->op_type](self, i);
}

void Var::compute_gradiant(Var *self){
    // 先计算逆拓扑序
    std::vector<Var*> topo_order{};
    std::unordered_set<Var*> visited;
    dfs(self, topo_order, visited);
    self->gradiant_node = make(1., false);
    // 按逆拓扑序访问
    for(auto iter = topo_order.rbegin(); iter!=topo_order.rend(); ++iter){
        auto node = *iter;
        // 把梯度传给每个孩子
        for(int i=0; i<OP_NODE_CNT[node->op_type]; ++i){
            auto child = node->getChild(i);
            child->gradiant_node = child->gradiant_node->add(gradOfChild(node, i));
        }
    }
}

}