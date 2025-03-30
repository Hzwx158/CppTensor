#define NUMCPP_UTILS_CONTEXT_CPP
#include "utils/context.h"
namespace numcpp::auto_del{

Context::~Context(){
    ContextLifeBase *tmp;
    while(!ptr_stack.empty()){
        tmp = ptr_stack.top();
        delete tmp;
        ptr_stack.pop();
    }
}
void Context::open(){
    current = new Context(current);
}
void Context::close(){
    auto t = current->parent;
    delete current;
    current = t;
}

}
#undef NUMCPP_UTILS_CONTEXT_CPP