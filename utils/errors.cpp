#include "./errors.h"
namespace numcpp{
std::string Error::ullToStr(size_t num){
    if(!num) return "0";
    std::string str = "";
    while(num){
        str += (num%10)+'0';
        num/=10;
    }
    return str;
}
std::string Error::llToStr(long long num){
    if(!num) return "0";
    std::string str = num>0?"":"-";
    num = (num>0?num:-num);
    while(num){
        str += (num%10)+'0';
        num/=10;
    }
    return str;
}
std::string Error::_from(SCR file, SCR func){
    return "From <"+file+">, func <"+func+">:\n\t";
}
Error::OR Error::outOfRange(SCR file, SCR func, long long idx, size_t be_, size_t en_){
    return OR(_from(file, func)
        +"Legal Range: ["+ullToStr(be_)+", "+ullToStr(en_)
        +"), but got <"+llToStr(idx)+">."
    );
}
Error::RE Error::wrong(SCR file, SCR func, SCR context){
    return RE(_from(file, func)+context);
}
Error::RE Error::author(SCR file, SCR func, SCR context){
    return RE(_from(file, func)+context);
}
Error::RE Error::unfinished(SCR file, SCR func){
    return RE(_from(file, func)+"Unfinished");
}
}