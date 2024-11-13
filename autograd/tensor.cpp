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
    :op_type(_op_type), require_grad{require_grad_}, value(val)
{
#if DEBUG
    std::cout<<"new Var @"<<(void*)this<<std::endl;
#endif
    compute();
    if(!isConst())
        zero_grad();
}
Var::Var(Value &&val, Op _op_type, bool require_grad_)
    :op_type(_op_type), require_grad{require_grad_}, value(std::move(val))
{
#if DEBUG
    std::cout<<"new Var @"<<(void*)this<<std::endl;
#endif
    compute();
    if(!isConst())
        zero_grad();
}
Var::Ptr Var::make(Op op_type, Ptr const &left, Ptr const &right){
    if(OP_NODE_CNT[op_type]!=2)
        throw Error::wrong(__FILE__, __func__, "op type");
    // placement new技术来分配内存
    char *mem = (char*)::operator new(sizeof(Var)+sizeof(Ptr)*2);
    new(mem+sizeof(Var)) Ptr(left);
    new(mem+sizeof(Var)+sizeof(Ptr)) Ptr(right);
    return Ptr(new(mem) Var(Value(0), op_type), &Var::deleter);    
}
Var::Ptr Var::make(Op op_type, Ptr const &left, Value *const_attr){
    if(op_type==PLACEHOLDER_OP)
        return ZERO();
    if(OP_NODE_CNT[op_type]==2)
        throw Error::wrong(__FILE__,__func__,"op type");
    // placement new 来分配内存
    char *mem = (char*)::operator new(sizeof(Var)+sizeof(Ptr)+sizeof(Value*)*hasConstAttr(op_type));
    new(mem+sizeof(Var)) Ptr(left);
    if(hasConstAttr(op_type))
        *(Value**)(mem+sizeof(Var)+sizeof(Ptr)) = const_attr;
    return Ptr(new(mem) Var(Value(0), op_type), &Var::deleter);
}
void Var::deleter(Var *var){
    if(!var) return;
    // std::cout << "var"<<*var<<std::endl;
    // std::cout << "grd"<<*var->gradient_node<<std::endl;
    switch(OP_NODE_CNT[var->op_type]){
    case 2:
        var->getChild(1).~Ptr();
        var->getChild(0).~Ptr();
        break;
    case 1:
        if(hasConstAttr(var->op_type))
            delete var->getConstAttr();
        var->getChild(0).~Ptr();
        break;
    default:
        break;
    }
    var->~Var();
    ::operator delete(var);
#if DEBUG
    std::cout<<"del Var @"<<(void*)var<<std::endl;
#endif
}
Var::~Var(){}

void dfs(Var::Ptr const &node, std::vector<Var::Ptr> &res, std::unordered_set<Var*> &visited){
    if((!node)||visited.count(node.get())) return;
    visited.emplace(node.get());
    for(int i=0; i<OP_NODE_CNT[node->op_type]; ++i){
        Var::Ptr const &child = node->getChild(i);
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

Var::Ptr operator+(Var::Ptr const &var1, Var::Ptr const &var2){
    if(var1->isConst()){
        if(var2->isConst())
            return Var::make(var1->value+var2->value, false);
        return var2 + (var1->value); 
    }
    if(var2->isConst())
        return var1 + (var2->value);
    return Var::make(Var::ADD_OP, var1, var2);
}
Var::Ptr operator+(Var::Ptr const &var, Var::Value const &val){
    if(var->isConst())
        return Var::make(var->value + val, false);
    return Var::make(Var::ADD_CONST_OP, var, new Var::Value(val));
}
Var::Ptr operator+(Var::Ptr const &var, Var::Value &&val){
    if(var->isConst())
        return Var::make(var->value + val, false);
    return Var::make(Var::ADD_CONST_OP, var, new Var::Value(std::move(val)));
}

Var::Ptr operator*(Var::Ptr const &var1, Var::Ptr const &var2){
    if(var1->isConst()){
        if(var2->isConst())
            return Var::make(var1->value*var2->value, false);
        return var2 * (var1->value); 
    }
    if(var2->isConst())
        return var1 * (var2->value);
    return Var::make(Var::MUL_OP, var1, var2);
}
Var::Ptr operator*(Var::Ptr const &var, Var::Value const &val){
    if(var->isConst())
        return Var::make(var->value * val, false);
    return Var::make(Var::MUL_CONST_OP, var, new Var::Value(val));
}
Var::Ptr operator*(Var::Ptr const &var, Var::Value &&val){
    if(var->isConst())
        return Var::make(var->value * val, false);
    return Var::make(Var::MUL_CONST_OP, var, new Var::Value(std::move(val)));
}

Var::Ptr Var::exp(Ptr const &var){
    if(var->isConst())
        return Var::make(var->value.exp(), false);
    return Var::make(Var::EXP_OP, var);
}



}