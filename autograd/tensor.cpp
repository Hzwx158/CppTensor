#include "tensor.hpp"
namespace numcpp{
int OP_NODE_CNT[]={
    [Var::PLACEHOLDER_OP]=0, 
    [Var::ADD_OP]=2,
    [Var::ADD_CONST_OP]=1,
    [Var::SUB_OP]=2,
    [Var::SUB_CONST_OP]=1,
    [Var::RSUB_CONST_OP]=1,
    [Var::MUL_OP]=2,
    [Var::MUL_CONST_OP]=1,
    [Var::DIV_OP]=2,
    [Var::DIV_CONST_OP]=1,
    [Var::RDIV_CONST_OP]=1,
    [Var::POW_CONST_OP]=1,
    [Var::RPOW_CONST_OP]=1,
    [Var::EXP_OP]=1,
    [Var::LOG_OP]=1,
    [Var::SIN_OP]=1,
    [Var::COS_OP]=1,
    [Var::TAN_OP]=1,
};
//--------------------------------内存管理----------------------------------
#define DEBUG 1

Var::Var(Value const &val, Op _op_type, bool require_grad_)
    :op_type(_op_type), require_grad{require_grad_}, has_parent{false}, value(val)
{
#if DEBUG
    std::cout<<"new Var @"<<(void*)this<<std::endl;
#endif
    compute();
    if(!isConst())
        zero_grad();
}
Var::Var(Value &&val, Op _op_type, bool require_grad_)
    :op_type(_op_type), require_grad{require_grad_}, has_parent{false}, value(std::move(val))
{
#if DEBUG
    std::cout<<"new Var @"<<(void*)this<<std::endl;
#endif
    compute();
    if(!isConst())
        zero_grad();
}
Var *Var::make(Op op_type, Var *left, Var *right){
    if(OP_NODE_CNT[op_type]!=2)
        throw Error::wrong(__FILE__, __func__, "op type");
    // 把left right都标记为有父亲的
    left->has_parent = right->has_parent = true;
    // placement new技术来分配内存
    char *mem = (char*)::operator new(sizeof(Var)+sizeof(Var*)*2);
    *(Var**)(mem+sizeof(Var)) = left;
    *(Var**)(mem+sizeof(Var)+sizeof(Var*)) = right;
    return new(mem) Var(Value(0), op_type);    
}
Var *Var::make(Op op_type, Var *left, Value *const_attr){
    if(op_type==PLACEHOLDER_OP)
        return new Var(Value(0),PLACEHOLDER_OP);
    if(OP_NODE_CNT[op_type]==2)
        throw Error::wrong(__FILE__,__func__,"op type");
    // left标记为有父亲的
    left->has_parent = true;
    // placement new 来分配内存
    char *mem = (char*)::operator new(sizeof(Var)+sizeof(Var*)+sizeof(Value*)*hasConstAttr(op_type));
    *(Var**)(mem+sizeof(Var)) = left;
    if(hasConstAttr(op_type))
        *(Value**)(mem+sizeof(Var)+sizeof(Var*)) = const_attr;
    return new(mem) Var(Value(0), op_type);
}
void Var::free(Var *var){
    if(!var) return;
    if(!var->isConst()){
        // std::cout << "var"<<*var<<std::endl;
        // std::cout << "grd"<<*var->gradiant_node<<std::endl;
        Var::free(var->gradiant_node);
    }
    if(var->has_parent) return;
    // 先计算逆拓扑序
    std::vector<Var*> topo_order{};
    std::unordered_set<Var*> visited;
    dfs(var, topo_order, visited);
    // 遍历拓扑序
    for(auto &ptr:topo_order){
        // 不可以直接delete，会回收不全，要按照下面这个方式delete
        ptr->~Var();
        ::operator delete(ptr);
    }
}
Var::~Var(){
#if DEBUG
    std::cout<<"del Var @"<<(void*)this<<std::endl;
#endif
}

void dfs(Var *node, std::vector<Var*> &res, std::unordered_set<Var*> &visited){
    if((!node)||visited.count(node)) return;
    visited.emplace(node);
    for(int i=0; i<OP_NODE_CNT[node->op_type]; ++i){
        auto child = node->getChild(i);
        if(child) dfs(child, res, visited);
    }
    res.push_back(node);
}
//--------------------------------运算函数----------------------------------
typedef Var::Value (Var::Value::*T_OP_COMPUTE_FUNC_2)(Var::Value const &) const;
typedef Var::Value (Var::Value::*T_OP_COMPUTE_FUNC_1)() const;
static T_OP_COMPUTE_FUNC_2 OP_COMPUTE_FUNC[]={
    [Var::PLACEHOLDER_OP] = nullptr,
    [Var::ADD_OP] = &Var::Value::operator+,
    [Var::ADD_CONST_OP] = &Var::Value::operator+,
    [Var::SUB_OP] = &Var::Value::operator-,
    [Var::SUB_CONST_OP] = &Var::Value::operator-,
    [Var::RSUB_CONST_OP] = &Var::Value::operator-,
    [Var::MUL_OP] = &Var::Value::operator*,
    [Var::MUL_CONST_OP] = &Var::Value::operator*,
    [Var::DIV_OP] = &Var::Value::operator/,
    [Var::DIV_CONST_OP] = &Var::Value::operator/,
    [Var::RDIV_CONST_OP] = &Var::Value::operator/,
    [Var::POW_CONST_OP] = nullptr, // @todo
    [Var::RPOW_CONST_OP] = nullptr,
    [Var::EXP_OP] = (T_OP_COMPUTE_FUNC_2)&Var::Value::exp,
    [Var::LOG_OP] = (T_OP_COMPUTE_FUNC_2)&Var::Value::log,
    [Var::SIN_OP] = (T_OP_COMPUTE_FUNC_2)&Var::Value::sin,
    [Var::COS_OP] = (T_OP_COMPUTE_FUNC_2)&Var::Value::cos,
    [Var::TAN_OP] = (T_OP_COMPUTE_FUNC_2)&Var::Value::tan,
};
//--------------------------------运算符----------------------------------

void Var::compute(){
    if(op_type==PLACEHOLDER_OP)
        return;
    getChild(0)->compute();
    Value const *p_op1 = &getChild(0)->value;
    if(OP_NODE_CNT[op_type]==1 && (!hasConstAttr(op_type))){
        value = (p_op1->*(T_OP_COMPUTE_FUNC_1)OP_COMPUTE_FUNC[op_type])();
        return;
    }
    Value const *p_op2;
    if(hasConstAttr(op_type))
        p_op2 = getConstAttr();
    else{
        getChild(1)->compute();
        p_op2 = &getChild(1)->value;
    }
    if(isReverse(op_type))
        std::swap(p_op1, p_op2);
    value = (p_op1->*OP_COMPUTE_FUNC[op_type])(*p_op2);
}
Var *Var::add(Var *var2) const{
    if(this->isConst()){
        if(var2->isConst())
            return Var::make(this->value+var2->value, false);
        return var2->add(this->value); 
    }
    if(var2->isConst())
        return this->add(var2->value);
    return Var::make(Var::ADD_OP, (Var*)this, var2);
}
Var *Var::add(Var::Value const &val) const{
    if(this->isConst())
        return Var::make(this->value + val, false);
    return Var::make(Var::ADD_CONST_OP, (Var*)this, new Var::Value(val));
}
Var *Var::add(Var::Value &&val) const{
    if(this->isConst())
        return Var::make(this->value + val, false);
    return Var::make(Var::ADD_CONST_OP, (Var*)this, new Var::Value(std::move(val)));
}

Var *Var::mul(Var *var2) const{
    if(this->isConst()){
        if(var2->isConst())
            return Var::make(this->value*var2->value, false);
        return var2->mul(this->value); 
    }
    if(var2->isConst())
        return this->mul(var2->value);
    return Var::make(Var::MUL_OP, (Var*)this, var2);
}
Var *Var::mul(Var::Value const &val) const{
    if(this->isConst())
        return Var::make(this->value * val, false);
    return Var::make(Var::MUL_CONST_OP, (Var*)this, new Var::Value(val));
}
Var *Var::mul(Var::Value &&val) const{
    if(this->isConst())
        return Var::make(this->value * val, false);
    return Var::make(Var::MUL_CONST_OP, (Var*)this, new Var::Value(std::move(val)));
}

Var *Var::exp(Var *var){
    if(var->isConst())
        return Var::make(var->value.exp(), false);
    return Var::make(Var::EXP_OP, var);
}



}