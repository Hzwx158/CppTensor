#ifndef NUMCPP_UTILS_POINTER_HPP
#define NUMCPP_UTILS_POINTER_HPP
#include "./base.h"
namespace numcpp
{
DEF_TYPE_JUDGE(void);

#define DEBUG 0
/**
 * @brief 智能指针（自己实现版）
 * @tparam T 指针指向类型
 */
template<class T>
class UniquePointer{
private:
    std::enable_if_t<!is_void_v<T>, T> *ptr;
public:
    constexpr UniquePointer(decltype(nullptr)=nullptr):ptr(nullptr){}
    /**
     * @brief 构造函数
     * @param ptr_ 被保管的指针
     * @attention ptr_必须是被new出来的, 否则会被delete两遍
    */
    constexpr explicit UniquePointer(T *&&ptr_):ptr(ptr_){}
    UniquePointer(const UniquePointer &)=delete;
    UniquePointer(UniquePointer &&uPtr):ptr(uPtr.ptr){
        uPtr.ptr=nullptr;
    }
    ~UniquePointer(){if(ptr) clear();}
    void clear(){
#if DEBUG
        if(ptr)
            std::cout<<"Pointer Free @"<<static_cast<void*>(ptr)<<std::endl;
#endif
        delete[] ptr;
        ptr=nullptr;    
    }
    operator T*() const {return ptr;}
    UniquePointer &operator=(const UniquePointer &)=delete;
    UniquePointer &operator=(UniquePointer &&uPtr){
        if(this==&uPtr)
            return *this;
        if(ptr)
            clear();
        ptr=uPtr.ptr;
        uPtr.ptr=nullptr;
        return *this;
    }
    UniquePointer &operator=(decltype(nullptr)){
        if(ptr)
            clear();
        return *this;
    }
};


}
#endif