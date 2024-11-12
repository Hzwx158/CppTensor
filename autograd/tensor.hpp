#ifndef NUMCPP_AUTOGRAD_TENSOR_HPP
#define NUMCPP_AUTOGRAD_TENSOR_HPP
// #include <memory>
#include <unordered_set>
#include "../shaped/array.hpp"
namespace numcpp{
class Var{
public:
    using Value = ShapedArray<double>;
    enum Op{
        PLACEHOLDER_OP,
        ADD_OP, ADD_CONST_OP,
        SUB_OP, SUB_CONST_OP, RSUB_CONST_OP,
        MUL_OP, MUL_CONST_OP,
        DIV_OP, DIV_CONST_OP, RDIV_CONST_OP,
        POW_CONST_OP, RPOW_CONST_OP,
        EXP_OP, LOG_OP,
        SIN_OP, COS_OP, TAN_OP,
    };
    Var(Var const &obj)=delete;
    Var(Var &&obj)=delete;
    // using Ptr = std::shared_ptr<Var>;
    Op op_type;
    bool require_grad;
    bool has_parent;
    Var *gradiant_node;
    Value value;
private:
    char data[0];
    /**
     * @brief 构造函数(私有)
     * @param val 初始值
     * @param _op_type 该节点符号类型
     * @param require_grad 是否需要求梯度(默认是)
     */
    explicit Var(Value const &val, Op _op_type, bool require_grad=true);
    /**
     * @brief 构造函数(私有)
     * @param val 初始值
     * @param _op_type 该节点符号类型
     * @param require_grad 是否需要求梯度(默认是)
     */
    explicit Var(Value &&val, Op _op_type, bool require_grad=true);
    /**
     * @brief 析构函数(私有)
     */
    ~Var();
public:
    /**
     * @brief 计算当前`value`
     */
    void compute();
    /**
     * @brief 清空梯度
     */
    void zero_grad(){gradiant_node = make(0, false);}

    
    /**
     * @brief 获取第i（0,1）个孩子的shared_ptr
     * @param i 第i个孩子，0或1
     * @return 一个Ptr const &
     */
    Var *getChild(int i) const{
        return *(Var**)(data + sizeof(Var*)*i);
    }
    /**
     * @brief 获取const_attr
     * @return const_attr
     */
    Value *getConstAttr() const{
        return *(Value**)(data + sizeof(Var*));
    }
    /**
     * @brief 构造一个节点
     * @param op_type 运算符类型
     * @param left 左节点
     * @param right 右节点
     * @return 一个shared_ptr，新节点
     */
    static Var *make(Op op_type, Var *left, Var *right);
    /**
     * @brief 构造一个节点
     * @param op_type 运算符类型，默认无
     * @param left 孩子节点(默认无)
     * @param const_attr 一个Value *类型的常量值(默认无)
     * @return 一个shared_ptr，新节点
     */
    static Var *make(Op op_type=PLACEHOLDER_OP, Var *left=nullptr, Value *const_attr=nullptr);
    /**
     * @brief 构造一个指定初始值的叶子节点
     * @param val 一个值
     * @param require_grad 需要求导(默认true)
     * @return 一个shared_ptr
     */
    static Var *make(Value const &val, bool require_grad=true){
        return new Var(val, PLACEHOLDER_OP, require_grad);
    }
    /**
     * @brief 构造一个指定初始值的叶子节点
     * @param val 一个值
     * @param require_grad 需要求导(默认true)
     * @return 一个shared_ptr
     */
    static Var *make(Value &&val, bool require_grad=true){
        return new Var(std::move(val), PLACEHOLDER_OP,require_grad);
    }
    /**
     * @brief 释放一个节点的函数，是make构造节点的shared_ptr的deleter
     * @param var 一个节点指针
     */
    static void free(Var *var);

public:
    /**
     * @brief 判断该节点是否是常量值节点(即该节点既是叶子节点也不需要求导)
     * @return 是则true
     */
    bool isConst() const{return (!require_grad)&&(op_type==PLACEHOLDER_OP);}
    H_OUTPUTABLE(Var const *){
        return osm<<"Var*("<<obj->value<<')';
    }
    H_OUTPUTABLE(Var){
        std::ostringstream oss;
        oss << "Var@"<<(void*)&obj << "{\n\t";
        oss << "value: "<< obj.value << ",\n\t";
        oss << "grad@"<<(void*)(obj.gradiant_node)<<",\n\t";
        oss << "grad.value:"<<obj.gradiant_node->value<<"\n}";
        return osm << oss.str();
    }
    Var *add(Var *var) const;
    Var *add(Value const &val) const;
    Var *add(Value &&val) const;
    friend Var *add(Value const &val, Var *var){return var->add(val);}
    friend Var *add(Value &&val, Var *var){return var->add(std::move(val));}
    Var *mul(Var *var) const;
    Var *mul(Value const &val) const;
    Var *mul(Value &&val) const;
    friend Var *mul(Value const &val, Var *var){return var->mul(val);}
    friend Var *mul(Value &&val, Var *var){return var->mul(std::move(val));}
    
    static Var *exp(Var *var);
    // Var *mul(Var *var);
    // Var *mul(Value const &val);
    // Var *mul(Value &&val);
    // Var *exp();
    /**
     * @brief (传给)第i个孩子的梯度
     * @param self 本节点
     * @param i 0或1，孩子节点的下标
     * @return 一个shared_ptr
     */
    static Var *gradOfChild(Var *self, int i);

    static void compute_gradiant(Var *self);
};
/**
 * @brief 判断一个符号有无const_attr
 * @param op_type 符号
 * @return 有则true
 */
inline bool hasConstAttr(Var::Op op_type){
    switch(op_type){
    case Var::ADD_CONST_OP:
    case Var::SUB_CONST_OP:
    case Var::MUL_CONST_OP:
    case Var::DIV_CONST_OP:
    case Var::POW_CONST_OP:
    case Var::RDIV_CONST_OP:
    case Var::RSUB_CONST_OP:
    case Var::RPOW_CONST_OP:
        return true;
    default:
        return false;
    }
}
/**
 * @brief 判断一个符号是否是反的
 * @param op_type 符号
 * @return 是则true
 */
inline bool isReverse(Var::Op op_type){
    switch(op_type){
    case Var::RDIV_CONST_OP:
    case Var::RPOW_CONST_OP:
    case Var::RSUB_CONST_OP:
        return true;
    default:
        return false;
    }
}

void dfs(Var *node, std::vector<Var*> &res, std::unordered_set<Var*> &visited);
}
#endif