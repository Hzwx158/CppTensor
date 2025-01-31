#ifndef NUMCPP_UTILS_ERRORS_H
#define NUMCPP_UTILS_ERRORS_H
#include <string>
#include <stdexcept>
namespace numcpp{
    
struct Error{
    using RE = std::runtime_error;
    using OR = std::out_of_range;
    using SCR = std::string const &;
    /**
     * @brief 把无符号整数变为字符串
     * @param num 无符号整数
     * @return 字符串
     */
    static std::string ullToStr(size_t num);
    /**
     * @brief 把符号整数变为字符串
     * @param num 符号整数
     * @return 字符串
     */
    static std::string llToStr(long long num);
    /**
     * @brief out of range错误, idx不在[`be_`, `en_`)之中
     * @param file 报错文件
     * @param func 报错函数名
     * @param idx 超出范围的下标
     * @param be_ 合法下标的起始
     * @param en_ 合法下标的终止
     * @return std::out_of_range
     */
    static OR outOfRange(SCR file, SCR func, long long idx, size_t be_, size_t en_);
    /**
     * @brief runtime error
     * @param file 报错文件
     * @param func 报错函数名
     * @param context 报错内容
     * @return std::runtime_error
     */
    static RE wrong(SCR file, SCR func, SCR context);
    /**
     * @brief runtime error, 但是正常来讲不会出现的错误、需要通知作者的错误
     * @param file 报错文件
     * @param func 报错函数名
     * @param context 报错内容
     * @return std::runtime_error
     */
    static RE author(SCR file, SCR func, SCR context);
    /**
     * @brief 未完成
     * @param file 报错文件
     * @param func 报错函数名
     */
    static RE unfinished(SCR file, SCR func);
    /**
     * @brief 除0错误
     * @param file 报错文件
     * @param func 报错函数名
     */
    static RE divByZero(SCR file, SCR func);
private:
    /**
     * @brief 格式化来源字符串
     * @param file 报错文件
     * @param func 报错函数名
     * @return From <file>, line <line>
     */
    static std::string _from(SCR file, SCR func);
};
}
#endif