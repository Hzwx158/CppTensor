#ifndef NUMCPP_UTILS_CONTEXT_H
#define NUMCPP_UTILS_CONTEXT_H
#include <stack>
namespace numcpp::auto_del{

struct ContextLifeBase;

class Context{
private:
    Context *parent;
    std::stack<ContextLifeBase*> ptr_stack;
    Context(Context *p):parent{p}, ptr_stack{}{}
    static Context *current;
public:
    friend class ContextLifeBase;
    ~Context();
    /**
     * @brief 向context添加一个被管理指针
     * @param ptr 指针
     */
    void add(ContextLifeBase *ptr) {ptr_stack.push(ptr);}
    /**
     * @brief 开启一个新的context
     */
    static void open();
    /**
     * @brief 关闭当前context，退回上一个context
     */
    static void close();
};
#ifdef NUMCPP_UTILS_CONTEXT_CPP
Context *Context::current = nullptr;
// #else
// extern Context *Context::current;
#endif

/**
 * @brief 被Context管理的指针类必须继承自这一基类
 */
struct ContextLifeBase{
    ContextLifeBase(){Context::current->add(this);}
    virtual ~ContextLifeBase(){}
};
/**
 * @brief 实现Context释放的语法糖类
 * @example 见下面示例：
 * {
 *  ContextUpdater _;
 *  T *p = new T();
 * }
 * 等价于
 * Context::open();
 * T *p=new T();
 * Context::close();
 */
struct ContextUpdater{
    ContextUpdater(){Context::open();}
    ~ContextUpdater(){Context::close();}
};

}
#endif