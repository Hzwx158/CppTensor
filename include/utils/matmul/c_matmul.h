#ifndef NUMCPP_UTILS_C_MATMUL_H
#define NUMCPP_UTILS_C_MATMUL_H
typedef unsigned long long size_t;
typedef long long LLong;
typedef unsigned int UInt;
typedef unsigned short UShort;
typedef unsigned char UChar;

#define MATMUL_DECL(T1, T2, Ret) int _matmul_##T1##_##T2 (T1 *mat1, T2 *mat2, Ret *res, size_t a, size_t b, size_t c);

#define ALL_MUL_C_TYPE(F)\
/* double */\
F(double, char, double)\
F(double, UChar, double)\
F(double, short, double)\
F(double, UShort, double)\
F(double, int, double)\
F(double, UInt, double)\
F(double, LLong, double)\
F(double, size_t, double)\
F(double, float, double)\
F(double, double, double)\
/* float */\
F(float, char, float)\
F(float, UChar, float)\
F(float, short, float)\
F(float, UShort, float)\
F(float, int, float)\
F(float, UInt, float)\
F(float, LLong, float)\
F(float, size_t, float)\
F(float, float, float)\
F(float, double, double)\
/* ull */\
F(size_t, char, size_t)\
F(size_t, UChar, size_t)\
F(size_t, short, size_t)\
F(size_t, UShort, size_t)\
F(size_t, int, size_t)\
F(size_t, UInt, size_t)\
F(size_t, LLong, size_t)\
F(size_t, size_t, size_t)\
F(size_t, float, float)\
F(size_t, double, double)\
/* ll */\
F(LLong, char, LLong)\
F(LLong, UChar, LLong)\
F(LLong, short, LLong)\
F(LLong, UShort, LLong)\
F(LLong, int, LLong)\
F(LLong, UInt, LLong)\
F(LLong, LLong, LLong)\
F(LLong, size_t, size_t)\
F(LLong, float, float)\
F(LLong, double, double)\
/* unsigned */\
F(UInt, char, UInt)\
F(UInt, UChar, UInt)\
F(UInt, short, UInt)\
F(UInt, UShort, UInt)\
F(UInt, int, UInt)\
F(UInt, UInt, UInt)\
F(UInt, LLong, LLong)\
F(UInt, size_t, size_t)\
F(UInt, float, float)\
F(UInt, double, double)\
/* int */\
F(int, char, int)\
F(int, UChar, int)\
F(int, short, int)\
F(int, UShort, int)\
F(int, int, int)\
F(int, UInt, UInt)\
F(int, LLong, LLong)\
F(int, size_t, size_t)\
F(int, float, float)\
F(int, double, double)\
/* UShort */\
F(UShort, char, int)\
F(UShort, UChar, int)\
F(UShort, short, int)\
F(UShort, UShort, int)\
F(UShort, int, int)\
F(UShort, UInt, UInt)\
F(UShort, LLong, LLong)\
F(UShort, size_t, size_t)\
F(UShort, float, float)\
F(UShort, double, double)\
/* short */\
F(short, char, int)\
F(short, UChar, int)\
F(short, short, int)\
F(short, UShort, int)\
F(short, int, int)\
F(short, UInt, UInt)\
F(short, LLong, LLong)\
F(short, size_t, size_t)\
F(short, float, float)\
F(short, double, double)\
/* UChar */\
F(UChar, char, int)\
F(UChar, UChar, int)\
F(UChar, short, int)\
F(UChar, UShort, int)\
F(UChar, int, int)\
F(UChar, UInt, UInt)\
F(UChar, LLong, LLong)\
F(UChar, size_t, size_t)\
F(UChar, float, float)\
F(UChar, double, double)\
/* UChar */\
F(char, char, int)\
F(char, UChar, int)\
F(char, short, int)\
F(char, UShort, int)\
F(char, int, int)\
F(char, UInt, UInt)\
F(char, LLong, LLong)\
F(char, size_t, size_t)\
F(char, float, float)\
F(char, double, double)

ALL_MUL_C_TYPE(MATMUL_DECL)
#undef MATMUL_DECL
#endif