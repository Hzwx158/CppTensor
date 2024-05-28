#pragma once
#include "../tensor/tensor.hpp"
#include <string>
#include <map>
namespace tryAI::autograd{
struct ExprNode;
struct Operator{
    using ValueType = Tensor; //TODO: change to tensor
    using Nodes = PtrVector<const ExprNode>;
    /**
     * @brief 计算主人node的孩子们的"脑袋上"的梯度
     * @param node 调用这个op.grad的node，有这个参数是因为含Const的Op需要
     * @param pre_grad node"脑袋上"的梯度
     * @return 孩子们脑袋上的梯度
    */
    virtual Nodes grad(const ExprNode *node, const ExprNode *pre_grad) const = 0;
    /**
     * @brief 计算子树的值
     * @param node 子树起点/根
     * @param varValues 孩子们的值
     * @return 计算的值

    virtual ValueType compute(const ExprNode *node, const std::vector<ValueType> &varValues) const = 0;
    /*
    /**
     * @brief 创建一个以本Op为.op的ExprNode对象
     * @param nodes 孩子们
     * @return ExprNode对象，被new的
    */
    virtual ExprNode *makeNode(const Nodes &nodes) const{
        throw std::runtime_error("From Operator::makeNode:\n\tUnfinished");
    }
    /**
     * @brief 判断本op所在结点是不是叶子结点/是不是PlaceHolder
     * @return 是则true
    */
    virtual bool isLeaf() const {return false;}
};


struct PlaceHolderOp: Operator{
    using ValueType = Operator::ValueType;
    using Nodes = PtrVector<const ExprNode>;
    Nodes grad(const ExprNode *node, const ExprNode *pre_grad) const override final{return {};}
    bool isLeaf() const override final{return true;}
};
#define Autograd_Op_Decl(ClassName)\
struct ClassName: Operator{\
    using ValueType = Operator::ValueType;\
    using Nodes = PtrVector<const ExprNode>;\
    Nodes grad(const ExprNode *node, const ExprNode *pre_grad) const override final;\
    ExprNode *makeNode(const Nodes &nodes) const override final; \
};

Autograd_Op_Decl(OnesLikeOp)
Autograd_Op_Decl(ZerosLikeOp)
Autograd_Op_Decl(AddOp)
Autograd_Op_Decl(MulOp)
/*
Autograd_Op_Decl(AddConstOp)
Autograd_Op_Decl(SubOp)
Autograd_Op_Decl(SubConstOp)
Autograd_Op_Decl(RSubConstOp)
Autograd_Op_Decl(MulConstOp)
Autograd_Op_Decl(DivOp)
Autograd_Op_Decl(DivConstOp)
Autograd_Op_Decl(RDivConstOp)
Autograd_Op_Decl(ExpOp)
Autograd_Op_Decl(LogOp)*/
#undef Autograd_Op_Decl

#ifndef DEFINE_VARS
extern const PlaceHolderOp *placeHolderOp;
extern const OnesLikeOp *onesLikeOp;
extern const ZerosLikeOp *zerosLikeOp;
extern const AddOp *addOp;
extern const MulOp *mulOp;
/*
extern const AddConstOp *addConstOp;
extern const SubOp *subOp;
extern const SubConstOp *subConstOp;
extern const MulConstOp *mulConstOp;
extern const DivOp *divOp;
extern const DivConstOp *divConstOp;
extern const RDivConstOp *rdivConstOp;
extern const ExpOp *expOp;
extern const LogOp *logOp;*/
#endif

struct ExprNode //data是Operator*、const_value的Tree Node
{
    using ValueType = Operator::ValueType;
    PtrVector<const ExprNode> children; //一般只有俩，万一有二班呢（？）
    const Operator * const op;
    ValueType value;
    const bool needGrad; //脑袋上有没有梯度

    ExprNode(
        const ValueType &value_ = ValueType(0), 
        const Operator *op_ = placeHolderOp, 
        const PtrVector<const ExprNode> &children_ = {},
        bool needGrad_=true);
    ~ExprNode();
    /**
     * @brief 静态函数，释放目前所有的匿名结点(非Leaf结点)
    */
    static void freeAllAnnoymous();

    H_OUTPUTABLE(ExprNode)
    ExprNode *add(const ExprNode *node) const {
        return addOp->makeNode({this, node});
    }
    ExprNode *mul(const ExprNode *node) const {
        return mulOp->makeNode({this, node});
    }
    ExprNode *mul(const ValueType &value_) const {
        return mulOp->makeNode({})//TODO: 常量怎么表示？
    }
};

/**
 * @brief 求导，dy/dx
 * @param expression 表达式树根节点，y
 * @param variables 自变量们，xs
 * @return 导数表达式们，对应x的位次存储
*/
PtrVector<const ExprNode> gradient(const ExprNode *expression, const PtrVector<const ExprNode> &variables);
/**
 * @brief 求值
 * @param expression 表达式树根节点
 * @param varValueMap 变量和对应的值
 * @return 值
*/
Operator::ValueType compute(const ExprNode *expression, const std::map<const ExprNode *, Tensor> &varValueMap);
inline ExprNode *Variable(const ExprNode::ValueType &value){
    auto res = new ExprNode(value);
    return res;
}


// const int TTTTT = sizeof(ExprNode);




}