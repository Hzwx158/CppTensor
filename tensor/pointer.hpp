#pragma once
namespace tryAI{

#define DEBUG 0
template<class T>
class UniquePointer{
private:
    T *ptr;
public:
    constexpr UniquePointer(decltype(nullptr)=nullptr):ptr(nullptr){}
    /**
     * @brief 构造函数
     * @param ptr_ 被保管的指针
     * @attention ptr_必须是被new出来的
    */
    UniquePointer(T *ptr_):ptr(ptr_){}
    UniquePointer(const UniquePointer &)=delete;
    UniquePointer(UniquePointer &&uPtr):ptr(uPtr.ptr){
        uPtr.ptr=nullptr;
    }
    ~UniquePointer(){if(ptr) clear();}
    void clear(){
#if DEBUG
        std::cout<<"Tensor Free @"<<static_cast<void*>(ptr)<<std::endl;
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
    /**
     * @brief 重设内容
     * @param ptr_ 管理的指针
     * @attention 原来管理的ptr不会被自动释放掉，会造成泄露！
    */
    void reset(T *ptr_){
        ptr=ptr_;
    }
};
}