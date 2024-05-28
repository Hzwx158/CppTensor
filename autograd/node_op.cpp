#include "./node_op.hpp"
#include <set>
#include <algorithm>
namespace tryAI::autograd{
#define DEFINE_VARS
const ZerosLikeOp *zerosLikeOp = new ZerosLikeOp();
const OnesLikeOp *onesLikeOp = new OnesLikeOp();
const PlaceHolderOp *placeHolderOp = new PlaceHolderOp();
const AddOp *addOp = new AddOp();
const MulOp *mulOp = new MulOp();
#undef DEFINE_VARS
/*
#define DefineCompute(ClassName) \
Operator::ValueType ClassName::compute(\
    const ExprNode *node,\
    const std::vector<ValueType> &varValues\
) const*/
#define DefineGrad(ClassName) \
Operator::Nodes ClassName::grad(\
    const ExprNode *node,\
    const ExprNode *pre_grad\
) const
#define DefineMakeNode(ClassName)\
ExprNode *ClassName::makeNode(const Nodes &nodes) const
//OnesLikeOp
DefineGrad(OnesLikeOp){
    assert(node->needGrad);
    return {zerosLikeOp->makeNode({node->children[0]})};
    //不管孩子需不需要求导，都是zeroLike(child)
}
DefineMakeNode(OnesLikeOp){
    assert(nodes.size()==1);
    auto &node=nodes[0];
    return new ExprNode(
        Tensor::ones(node->value.getShape()),
        onesLikeOp,{node}
    );
}
//ZerosLikeOp
DefineGrad(ZerosLikeOp){
    assert(node->needGrad);
    return {zerosLikeOp->makeNode({node->children[0]})};
}
DefineMakeNode(ZerosLikeOp){
    assert(nodes.size()==1);
    auto &node=nodes[0];
    return new ExprNode(
        Tensor::zeros(node->value.getShape()),
        zerosLikeOp,{node}
    );
}
//AddOp
DefineGrad(AddOp){
    auto &child0 = node->children[0];
    auto &child1 = node->children[1];
    if(child0->needGrad && child1->needGrad)
        return Nodes{pre_grad, pre_grad};
    else if(node->needGrad)
        return {pre_grad};
    return {};
}
DefineMakeNode(AddOp){
    assert(nodes.size()==2);
    auto &node0 = nodes[0];
    auto &node1 = nodes[1];
    return new ExprNode(
        node0->value+node1->value,
        addOp, nodes,
        node0->needGrad || node1->needGrad
    );
}
//MulOp
DefineGrad(MulOp){
    auto &child0 = node->children[0];
    auto &child1 = node->children[1];
    if(child0->needGrad && child1->needGrad)
        return Nodes{
            pre_grad->mul(child1), 
            pre_grad->mul(child0)
        };
    else if(child0->needGrad)
        return {pre_grad->mu};//TODO: 常量怎么表示？
    else if(child1->needGrad)
        //...
        return ababa;
    return {};
}
DefineMakeNode(MulOp){
    assert(nodes.size()==2);
    return new ExprNode(
        nodes[0]->value*nodes[1]->value,
        mulOp, nodes
    );
}
//TODO:other op
//#undef DefineCompute
#undef DefineGrad
#undef DefineMakeNode
//ExprNode
static std::vector<const ExprNode *> annoymous_ptrs;
ExprNode::ExprNode(
    const ValueType &value_, 
    const Operator *op_, 
    const PtrVector<const ExprNode> &children_,
    bool needGrad_
):children(children_), op(op_), value(value_), needGrad(needGrad_){
    //std::cout<<"ExprNode.create@"<<static_cast<void*>(this)<<std::endl;
    if(!op->isLeaf())
        annoymous_ptrs.push_back(this);
}
ExprNode::~ExprNode(){ //TODO 现在这个不是dfs
    //std::cout<<"ExprNode.delete@"<<static_cast<void*>(this)<<std::endl;
}
void ExprNode::freeAllAnnoymous(){
    for(auto &p:annoymous_ptrs)
        delete p;
    annoymous_ptrs.clear();
}
std::ostream &operator<<(std::ostream &osm, const ExprNode &obj){
    return osm<<"{@"<<&obj<<", value:"<<obj.value<<'}';
}


//gradient

void dfs(const ExprNode *node, std::vector<const ExprNode *> &order_list, std::set<const ExprNode*> &visited){
    if(visited.count(node)) return;
    visited.insert(node);
    size_t cCnt=node->children.size();
    for(size_t i=0;i<cCnt;++i)
        dfs(node->children[i], order_list, visited);
    order_list.push_back(node);
}
/**
 * @brief 获取一个树的拓扑序
 * @param root 树根
 * @return 一个节点指针数组，按拓扑序存储
*/
PtrVector<const ExprNode> topo_order(const ExprNode *root){
    std::vector<const ExprNode*> res;
    std::set<const ExprNode *> visited;
    dfs(root, res, visited);
    return PtrVector(res.data(), visited.size());
}
PtrVector<const ExprNode> gradient(
    const ExprNode *expression, 
    const PtrVector<const ExprNode> &variables
){
    auto topo_order_list = topo_order(expression); //获取拓扑序（逆拓扑序指所有爹完事才是我）
    size_t cnt = topo_order_list.size();
    std::map<const ExprNode*, const ExprNode *> var_grad_map{{
        topo_order_list[-1],  //初始化，脑袋上是onesLike
        onesLikeOp->makeNode({topo_order_list[-1]})
    }}; //节点->脑袋上的梯度们的和
    const ExprNode *cur, *cur_grad, *child;
    PtrVector<const ExprNode> child_grads; 
    size_t cCnt;
    for(size_t i=0;i<cnt;++i){ //逆拓扑序
        cur = topo_order_list[cnt-i-1]; //现在计算结点
        if(cur->op->isLeaf()) //是叶子，不用算了，直接用脑袋上的就行
            continue;
        cur_grad = var_grad_map.at(cur); //现在节点脑袋上的梯度和
        //逆拓扑序保证了cur的所有爹都算过了，现在算cur的所有孩子
        child_grads = cur->op->grad(cur, cur_grad); //每个孩子脑袋上的梯度
        cCnt = cur->children.size();
        assert(cCnt==child_grads.size());
        for(size_t j=0, jj=0;jj<cCnt;++jj){
            child = cur->children[j];
            if(!child->needGrad)//这个孩子不求导，略过
                continue;
            //把孩子脑袋上的梯度加在一起
            if(var_grad_map.count(child)){
                auto &tmp = var_grad_map.at(child);
                tmp = tmp->add(child_grads[j]);
            }
            else var_grad_map.insert({child, child_grads[j]});
            ++j;
        }
    }
    cnt = variables.size();
    PtrVector<const ExprNode> res(cnt);
    for(size_t i=0;i<cnt;++i){
        res[i]=var_grad_map[variables[i]];
    }
    return res;
}
Operator::ValueType compute(const ExprNode *expression, const std::map<const ExprNode *, Tensor> &varValueMap){
    //TODO
    auto topo_order_list = topo_order({expression});
    return 0;
}

}