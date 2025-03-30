/** 
 * @author hzwx158
 * 这个文件主要实现了一个类似python变量的自动回收机制
 * 相比shared_ptr的自动回收，本文件实现的`numcpp::Var`类是以变量形式出现（而不是指针）
 * 对于运算符(目前实现的是部分双目运算符)非常友好
 */
#ifndef NUMCPP_UTILS_SHARED_VAR_HPP
#define NUMCPP_UTILS_SHARED_VAR_HPP
#include <atomic>
#include <cstdio>
#include "utils/base.h"

namespace numcpp::auto_del{

// #define LIST_ALL_OPERATOR_2(F)\
// F(+, add)\
// F(*, mul)\
// F(/, div)\
// F(-, sub)\
// F(%, mod)\
// F(^, xor)\
// F(&, bitand)\
// F(|, bitor)\
// F(<<, lsh)\
// F(>>, rsh)\
// // F(==, eq)\
// // F(!=, neq)\
// // F(>=, ge)\
// // F(>, gt)\
// // F(<, lt)\
// // F(<=, le)\
// // F(&&, and)\
// // F(||, or)

// #define X2_OPERATABLE(op, opName)\
// template<class T1, class T2>\
// class is_##opName##able{\
// private:\
// 	template<class U1, class U2>\
// 	static decltype(std::declval<U1>() op std::declval<U2>(), std::true_type()) test(int);\
// 	template<class, class>\
// 	static std::false_type test(...);\
// public:\
// 	static constexpr bool value = decltype(test<T1, T2>(0))::value;\
// };\
// template<class T1, class T2>\
// inline constexpr bool is_##opName##able_v = is_##opName##able<T1, T2>::value;\
// template<class T1, class T2>\
// using opName##_res_t = decltype(std::declval<T1>() op std::declval<T2>());

// LIST_ALL_OPERATOR_2(X2_OPERATABLE)

// #undef X2_OPERATABLE


/**
 * @brief 共享变量的基类
 */
struct _VarBase{
	std::atomic<size_t> refcnt;
    /**
     * @brief 构造函数：引用计数为1
     * @attention 不要用这个来构造，用`make`函数来构造
     */
	constexpr _VarBase():refcnt{1ull}{}
    /**
     * @brief 析构函数
     */
	virtual ~_VarBase(){
		printf("free %p, refcnt %llu\n", (void*)this, refcnt.load());
	}

    /**
     * @brief 定制一个共享变量的指针
     * @tparam T 默认是_VarBase，一般这里要填_VarWrapper<Var>
     * @tparam 被打包的函数的构造函数参数
     * @param args 被打包的函数的构造函数(Var类型)参数
     * @return 一个T*类型的指针
     */
	template<class T=_VarBase, class ...Args>
	static T *make(Args&&... args){
		std::enable_if_t<std::is_base_of_v<_VarBase, T>, T> *res = nullptr;
        // 要求必须是_VarBase的子类
		if constexpr(sizeof...(Args))
			res = new T(std::forward<Args>(args)...);
		else res = new T();
		printf("make %p\n", (void*)res);
		return res;
	}
    /**
     * @brief 释放一个共享变量
     * @param ptr 共享变量的地址
     * @return 如果ptr的引用计数归零、被释放掉了，则返回true；否则返回false
     */
	static bool free(_VarBase *ptr){
		if(!ptr) return false;
		auto tmp = ptr->refcnt.load();
		while(!ptr->refcnt.compare_exchange_weak(tmp, tmp-1));
		if(!ptr->refcnt){
			// fmt::println("free {}", (void*)ptr);
			delete ptr;
			return true;
		}
		return false;
	}
    /**
     * @brief 赋值函数；即a被释放一次，然后被赋为b，b引用计数加一次
     * @param a 被赋值变量
     * @param b 用以赋值的变量
     */
	static void assign(_VarBase *&a, _VarBase const *b){
		if(a==b) return;
		_VarBase::free(a);
		if(!b) return;
		a = (_VarBase*)b;
		a->refcnt.fetch_add(1);
	}
};

/**
 * @brief 带一个被打包变量的类，类似于一个shared_ptr
 * @tparam T 被打包变量类型
 */
template<class T>
class _VarWrapper:public _VarBase{
private:
	T __real_var;
public:
    /**
     * @brief 构造函数
     * @param args T类型构造函数的参数
     */
	template<class ...Args>
	_VarWrapper(Args... args):__real_var(args...){}
    _VarWrapper(const T &v):__real_var(v){}
	_VarWrapper(T &&v):__real_var(std::move(v)){}
	_VarWrapper(const _VarWrapper &)=delete;
	_VarWrapper(_VarWrapper &&w):__real_var(std::move(w.__real_var)){}
	~_VarWrapper(){}
    /**
     * @brief 获取被包裹变量
     * @return 被包裹变量的引用
     */
	T const & value() const {return __real_var;}
    /**
     * @brief 获取被包裹变量
     * @return 被包裹变量的引用
     */
	T &value() {return __real_var;}
};

/**
 * @brief 真正被提供给外面的类
 * @tparam T 被打包变量的类型
 */
template<class T>
struct Var{
	_VarWrapper<T> *ptr; // 维护这个wrapper指针，掌握被包裹变量的生命周期
	
    /**
     * @brief 构造函数，指向一个null
     */
	constexpr Var(std::nullptr_t &):ptr{nullptr}{}
    /**
     * @brief 构造函数，指向一个null
     */
	constexpr Var(std::nullptr_t &&):ptr{nullptr}{}

    /**
     * @brief 构造函数
     * @param args T构造函数参数
     */
	template<class ...Args>
	Var(Args&&... args)
		:ptr(_VarBase::make<_VarWrapper<T>>(std::forward<Args>(args)...))
	{
		// printf("HI\n");
	}

    /**
     * @brief 根据wrapper指针初始化一个Var
     * @param p 任意一个wrapper指针
     * @attention 如果类型一致，则p引用计数加一；如果类型不一致，新创造一个类型一致的wrapper
     */
	template<class U=T>
	explicit Var(_VarWrapper<U> *p){
		if constexpr(std::is_same_v<T, U>){
            // 这段相当于assign((T*&)nullptr, p)
			p->refcnt.fetch_add(1);
			ptr = p;
		} else{
			ptr = _VarBase::make<_VarWrapper<T>>(p->value());
		}
	}
	~Var(){
		if(!ptr) return;
		_VarBase::free(ptr);
		ptr = nullptr;
	}
	Var(const Var &v):Var(v.ptr){}
	Var(Var &&v):ptr{v.ptr}{v.ptr = nullptr;}

	// Var &operator=(Var const &)=delete;
	// Var &operator=(Var &&)=delete;
	/**
     * @brief 赋值函数。
     * @param a 用以赋值的变量
     * @return *this
     */
	template<class U>
	Var<T> &operator=(U const &a){
		// printf("ptr = %p\n", (void*)this->ptr);
		if constexpr(std::is_same_v<decltype(nullptr), U>){	
			if(ptr) this->~Var();
		} else {
			if(ptr)
				_VarBase::free(this->ptr);
			ptr = _VarBase::make<_VarWrapper<T>>(a);
		}
		return *this;
	}
    /**
     * @brief 赋值函数。
     * @attention 浅拷贝，而非深拷贝
     * @param v 用以赋值的Var
     * @return *this
     */
	template<class U>
	Var<T> &operator=(const Var<U> &v){
		if((void*)this!=(void*)&v){
			_VarBase::free(ptr);
			ptr = _VarBase::make<_VarWrapper<T>>(v.ptr->value());
		}
		return *this;
	}
    /**
     * @brief 赋值函数。
     * @attention 浅拷贝，而非深拷贝
     * @param v 用以赋值的Var
     * @return *this
     */
	Var<T> &operator=(const Var<T> &v){
		if((void*)this!=(void*)&v)
			_VarBase::assign((_VarBase*&)ptr, v.ptr);
		return *this;
	}
	/**
     * @brief 赋值函数。
     * @attention 浅拷贝，而非深拷贝
     * @param v 用以赋值的Var
     * @return *this
     */
	template<class U>
	Var<T> &operator=(Var<U> &&v){
		if((void*)this==(void*)&v) 
			return *this;
		_VarBase::free(ptr);
		ptr = _VarBase::make<_VarWrapper<T>>(v.ptr->value());
		_VarBase::free(v.ptr);
		v.ptr = nullptr;
		return *this;
	}
	/**
     * @brief 赋值函数。
     * @attention 浅拷贝，而非深拷贝
     * @param v 用以赋值的Var
     * @return *this
     */
	Var<T> &operator=(Var<T> &&v){
		if((void*)this==(void*)&v) 
			return *this;
		_VarBase::free(ptr);
		ptr = v.ptr;
		v.ptr = nullptr;
		return *this;
	}

	void swap(Var<T> &v){
		if((void*)this == (void*)&v) return;
		std::swap(ptr, v.ptr);
	}
    /**
     * @brief 获取被包裹变量
     * @return 被包裹变量的引用
     */
	T const &value() const{return ptr->value();}
    /**
     * @brief 获取被包裹变量
     * @return 被包裹变量的引用
     */
	T &value(){return ptr->value();}
	T *operator->() {return ptr?&ptr->value():nullptr;}
	const T *operator->() const {return ptr?&ptr->value():nullptr;}
	size_t refcnt() const {return ptr?ptr->refcnt.load():0;}
};

#define OPERATOR2_DEF(op, opTag)\
template<class T, class U=T>\
Var<op_ret_t<EOperation::opTag, T, U>> operator op (const Var<T> &v1, const Var<U> &v2){\
	return Var<op_ret_t<EOperation::opTag, T, U>>(v1.ptr->value() op v2.ptr->value());\
}\
template<class T, class U=T>\
Var<op_ret_t<EOperation::opTag, T, U>> operator op (const Var<T> &v, const U &a){\
	return Var<op_ret_t<EOperation::opTag, T, U>>(v.ptr->value() op a);\
}\
template<class T, class U=T>\
Var<op_ret_t<EOperation::opTag, U, T>> operator op (const U &a, const Var<T> &v){\
	return Var<op_ret_t<EOperation::opTag, U, T>>(a op v.ptr->value());\
}\
template<class T, class U=T>\
Var<T> &operator op##=(Var<T> &v, const U &a){\
	_VarBase::assign((_VarBase*&)v.ptr, Var<T>((v op a).ptr).ptr);\
	return v;\
}

OPERATOR2_DEF(+, ADD);
OPERATOR2_DEF(-, SUB);
OPERATOR2_DEF(*, MUL);
OPERATOR2_DEF(/, DIV);
OPERATOR2_DEF(%, MOL);
OPERATOR2_DEF(&, BIT);
OPERATOR2_DEF(|, BIT);
OPERATOR2_DEF(^, BIT);
OPERATOR2_DEF(<<, BIT);
OPERATOR2_DEF(>>, BIT);

#undef OPERATOR2_DEF

template<class T>
_VarWrapper(_VarWrapper<T>) -> _VarWrapper<T>;

}

namespace numcpp{
    template<class T>
    using SharedVar = auto_del::Var<T>;
}

namespace std{
	template<typename T>
	constexpr bool is_default_constructible_v<numcpp::SharedVar<T>> = is_default_constructible_v<T>;

}
#endif