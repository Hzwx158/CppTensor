#pragma once
#include <iostream>
#include <stdexcept>

namespace tryAI{
/**
 * @brief 只运行一次助手 
 * @param retval 多次运行时的返回值，可以不填
*/
#define H_RUN_ONCE(retval) \
static bool __runOnceHelper=false;\
if(__runOnceHelper) return retval;\
__runOnceHelper=true;

//可输出类标签
#define H_OUTPUTABLE(ClassName) \
friend std::ostream & operator<<(std::ostream &osm, const ClassName &obj);

#define useStdIO using std::cin;using std::cout;using std::endl;

#define TENSOR_RES_OP_DECL(ClassName, operatorName) ClassName &operator operatorName##= (const ClassName &obj);\
ClassName operator operatorName(const ClassName &obj) const {\
    ClassName res(*this);\
    return res operatorName##= obj;\
}

template<class T>
class list{
private:
    std::vector<T> v;
public:
    list(const std::vector<T> &vec):v(vec){}
    list(std::vector<T> &&vec):v(std::move(vec)){}
    list(std::initializer_list<T> l):v(l){}
    list(const list &obj):v(obj.v){}
    list(list &&obj):v(std::move(obj.v)){}
    operator std::vector<T>& (){return v;}
    const std::vector<T> &getVector() const {return v;}
};

#define PARTIAL_SPECIALIZE_LIST(orgClass, saveClass)\
template<>\
class list<orgClass>{\
private:\
    std::vector<saveClass> v;\
public:\
    list(const std::vector<orgClass> &vec):v(vec.size(),0){\
        for(size_t i=0;i<vec.size();++i)\
            v[i]=vec[i];\
    }\
    list(std::initializer_list<orgClass> l):v(l.size(),0){\
        auto p=l.begin();\
        for(size_t i=0;i<l.size();++i)\
            v[i]=*(p+i);\
    }\
    list(const list &obj):v(obj.v){}\
    list(list &&obj):v(std::move(obj.v)){}\
    operator std::vector<saveClass>& (){return v;}\
    const std::vector<saveClass> &getVector() const {return v;}\
};
PARTIAL_SPECIALIZE_LIST(int, size_t)
PARTIAL_SPECIALIZE_LIST(short, size_t)
PARTIAL_SPECIALIZE_LIST(unsigned int, size_t)
PARTIAL_SPECIALIZE_LIST(unsigned short, size_t)


}