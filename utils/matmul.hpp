#include "./errors.h"
#include "./base.h"
namespace numcpp::linalg{

/**
 * @brief 矩阵乘法实现
 * @param mat1 矩阵1, 形状为(`a`, `b`)
 * @param mat2 矩阵2, 形状为(`b`, `c`)
 * @param res 结果存放矩阵, 形状为(`a`, `c`)
 * @param a 形状
 * @param b 形状
 * @param c 形状
 * @return 0表示正常, 1表示出现了正常不会出现的奇怪错误
 */
template<class T1, class T2, class Ret = op_ret_t<EOperation::MUL, T1, T2>>
int _matmul(
    T1 *mat1, T2 *mat2, 
    op_ret_t<EOperation::MUL, T1, T2> *res, 
    size_t a, size_t b, size_t c
){
    if(a*b*c==0) return 0;
    if((!mat1)||(!mat2)||(!res)) return 0;
    size_t i,j;
    T1 *a0k_p, *a1k_p, *a2k_p, *a3k_p;

    for(i=0;i+4<=a;i+=4){
        j=0;
        for(;j+4<=c;j+=4){
            Ret c00=0,c01=0,c02=0,c03=0,
            c10=0,c11=0,c12=0,c13=0,
            c20=0,c21=0,c22=0,c23=0,
            c30=0,c31=0,c32=0,c33=0;
            Ret a0k, a1k, a2k, a3k, bk0, bk1, bk2, bk3;
            a0k_p = mat1+i*b;
            a1k_p = mat1+(i+1)*b;
            a2k_p = mat1+(i+2)*b;
            a3k_p = mat1+(i+3)*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; bk1 = mat2[k*c+1+j]; bk2 = mat2[k*c+2+j]; bk3 = mat2[k*c+3+j];
                a0k = *a0k_p++; a1k = *a1k_p++; a2k = *a2k_p++; a3k = *a3k_p++;
                c00 += a0k*bk0; c01 += a0k*bk1; c02 += a0k*bk2; c03 += a0k*bk3;
                c10 += a1k*bk0; c11 += a1k*bk1; c12 += a1k*bk2; c13 += a1k*bk3;
                c20 += a2k*bk0; c21 += a2k*bk1; c22 += a2k*bk2; c23 += a2k*bk3;
                c30 += a3k*bk0; c31 += a3k*bk1; c32 += a3k*bk2; c33 += a3k*bk3;
            }
            res[i*c+j] += c00; res[i*c+j+1] += c01; res[i*c+j+2] += c02; res[i*c+j+3] += c03;
            res[(i+1)*c+j] += c10; res[(i+1)*c+j+1] += c11; res[(i+1)*c+j+2] += c12; res[(i+1)*c+j+3] += c13;
            res[(i+2)*c+j] += c20; res[(i+2)*c+j+1] += c21; res[(i+2)*c+j+2] += c22; res[(i+2)*c+j+3] += c23;
            res[(i+3)*c+j] += c30; res[(i+3)*c+j+1] += c31; res[(i+3)*c+j+2] += c32; res[(i+3)*c+j+3] += c33;
        }
        if(j==c) continue;
        //还有一部分, 为了效率开始屎山
        size_t rest_cnt = c-j;
        switch(rest_cnt){
        case 3:{
            Ret c00=0,c01=0,c02=0,
            c10=0,c11=0,c12=0,
            c20=0,c21=0,c22=0,
            c30=0,c31=0,c32=0;
            Ret a0k, a1k, a2k, a3k, bk0, bk1, bk2;
            a0k_p = mat1+i*b;
            a1k_p = mat1+(i+1)*b;
            a2k_p = mat1+(i+2)*b;
            a3k_p = mat1+(i+3)*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; bk1 = mat2[k*c+1+j]; bk2 = mat2[k*c+2+j];
                a0k = *a0k_p++; a1k = *a1k_p++; a2k = *a2k_p++; a3k = *a3k_p++;
                c00 += a0k*bk0; c01 += a0k*bk1; c02 += a0k*bk2;
                c10 += a1k*bk0; c11 += a1k*bk1; c12 += a1k*bk2;
                c20 += a2k*bk0; c21 += a2k*bk1; c22 += a2k*bk2;
                c30 += a3k*bk0; c31 += a3k*bk1; c32 += a3k*bk2;
            }
            res[i*c+j] += c00; res[i*c+j+1] += c01; res[i*c+j+2] += c02;
            res[(i+1)*c+j] += c10; res[(i+1)*c+j+1] += c11; res[(i+1)*c+j+2] += c12;
            res[(i+2)*c+j] += c20; res[(i+2)*c+j+1] += c21; res[(i+2)*c+j+2] += c22;
            res[(i+3)*c+j] += c30; res[(i+3)*c+j+1] += c31; res[(i+3)*c+j+2] += c32;
            break;
        }
        case 2:{
            Ret c00=0,c01=0,
            c10=0,c11=0,
            c20=0,c21=0,
            c30=0,c31=0;
            Ret a0k, a1k, a2k, a3k, bk0, bk1;
            a0k_p = mat1+i*b;
            a1k_p = mat1+(i+1)*b;
            a2k_p = mat1+(i+2)*b;
            a3k_p = mat1+(i+3)*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; bk1 = mat2[k*c+1+j];
                a0k = *a0k_p++; a1k = *a1k_p++; a2k = *a2k_p++; a3k = *a3k_p++;
                c00 += a0k*bk0; c01 += a0k*bk1; 
                c10 += a1k*bk0; c11 += a1k*bk1; 
                c20 += a2k*bk0; c21 += a2k*bk1; 
                c30 += a3k*bk0; c31 += a3k*bk1; 
            }
            res[i*c+j] += c00; res[i*c+j+1] += c01; 
            res[(i+1)*c+j] += c10; res[(i+1)*c+j+1] += c11; 
            res[(i+2)*c+j] += c20; res[(i+2)*c+j+1] += c21; 
            res[(i+3)*c+j] += c30; res[(i+3)*c+j+1] += c31; 
            break;
        }
        case 1:{
            Ret c00=0,
            c10=0,
            c20=0,
            c30=0;
            Ret a0k, a1k, a2k, a3k, bk0;
            a0k_p = mat1+i*b;
            a1k_p = mat1+(i+1)*b;
            a2k_p = mat1+(i+2)*b;
            a3k_p = mat1+(i+3)*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; 
                a0k = *a0k_p++; a1k = *a1k_p++; a2k = *a2k_p++; a3k = *a3k_p++;
                c00 += a0k*bk0; 
                c10 += a1k*bk0; 
                c20 += a2k*bk0; 
                c30 += a3k*bk0;  
            }
            res[i*c+j] += c00; 
            res[(i+1)*c+j] += c10; 
            res[(i+2)*c+j] += c20; 
            res[(i+3)*c+j] += c30; 
            break;
        }
        default: return 1;
        }
    }
    if(i==a) return 0;
    //还有一部分，为了效率屎山
    size_t rest_i_cnt = a-i;
    // copy and change from up
    switch(rest_i_cnt){
    case 3:{
        j=0;
        for(;j+4<=c;j+=4){
            Ret c00=0,c01=0,c02=0,c03=0,
            c10=0,c11=0,c12=0,c13=0,
            c20=0,c21=0,c22=0,c23=0;
            Ret a0k, a1k, a2k, bk0, bk1, bk2, bk3;
            a0k_p = mat1+i*b;
            a1k_p = mat1+(i+1)*b;
            a2k_p = mat1+(i+2)*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; bk1 = mat2[k*c+1+j]; bk2 = mat2[k*c+2+j]; bk3 = mat2[k*c+3+j];
                a0k = *a0k_p++; a1k = *a1k_p++; a2k = *a2k_p++;
                c00 += a0k*bk0; c01 += a0k*bk1; c02 += a0k*bk2; c03 += a0k*bk3;
                c10 += a1k*bk0; c11 += a1k*bk1; c12 += a1k*bk2; c13 += a1k*bk3;
                c20 += a2k*bk0; c21 += a2k*bk1; c22 += a2k*bk2; c23 += a2k*bk3;
            }
            res[i*c+j] += c00; res[i*c+j+1] += c01; res[i*c+j+2] += c02; res[i*c+j+3] += c03;
            res[(i+1)*c+j] += c10; res[(i+1)*c+j+1] += c11; res[(i+1)*c+j+2] += c12; res[(i+1)*c+j+3] += c13;
            res[(i+2)*c+j] += c20; res[(i+2)*c+j+1] += c21; res[(i+2)*c+j+2] += c22; res[(i+2)*c+j+3] += c23;
        }
        if(j==c) return 0;
        size_t rest_cnt = c-j;
        switch(rest_cnt){
        case 3:{
            Ret c00=0,c01=0,c02=0,
            c10=0,c11=0,c12=0,
            c20=0,c21=0,c22=0;
            Ret a0k, a1k, a2k, bk0, bk1, bk2;
            a0k_p = mat1+i*b;
            a1k_p = mat1+(i+1)*b;
            a2k_p = mat1+(i+2)*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; bk1 = mat2[k*c+1+j]; bk2 = mat2[k*c+2+j];
                a0k = *a0k_p++; a1k = *a1k_p++; a2k = *a2k_p++; 
                c00 += a0k*bk0; c01 += a0k*bk1; c02 += a0k*bk2;
                c10 += a1k*bk0; c11 += a1k*bk1; c12 += a1k*bk2;
                c20 += a2k*bk0; c21 += a2k*bk1; c22 += a2k*bk2;
            }
            res[i*c+j] += c00; res[i*c+j+1] += c01; res[i*c+j+2] += c02;
            res[(i+1)*c+j] += c10; res[(i+1)*c+j+1] += c11; res[(i+1)*c+j+2] += c12;
            res[(i+2)*c+j] += c20; res[(i+2)*c+j+1] += c21; res[(i+2)*c+j+2] += c22;
            break;
        }
        case 2:{
            Ret c00=0,c01=0,
            c10=0,c11=0,
            c20=0,c21=0;
            Ret a0k, a1k, a2k, bk0, bk1;
            a0k_p = mat1+i*b;
            a1k_p = mat1+(i+1)*b;
            a2k_p = mat1+(i+2)*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; bk1 = mat2[k*c+1+j];
                a0k = *a0k_p++; a1k = *a1k_p++; a2k = *a2k_p++;
                c00 += a0k*bk0; c01 += a0k*bk1; 
                c10 += a1k*bk0; c11 += a1k*bk1; 
                c20 += a2k*bk0; c21 += a2k*bk1;
            }
            res[i*c+j] += c00; res[i*c+j+1] += c01; 
            res[(i+1)*c+j] += c10; res[(i+1)*c+j+1] += c11; 
            res[(i+2)*c+j] += c20; res[(i+2)*c+j+1] += c21;
            break;
        }
        case 1:{
            Ret c00=0,
            c10=0,
            c20=0;
            Ret a0k, a1k, a2k, bk0;
            a0k_p = mat1+i*b;
            a1k_p = mat1+(i+1)*b;
            a2k_p = mat1+(i+2)*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; 
                a0k = *a0k_p++; a1k = *a1k_p++; a2k = *a2k_p++;
                c00 += a0k*bk0; 
                c10 += a1k*bk0; 
                c20 += a2k*bk0;
            }
            res[i*c+j] += c00; 
            res[(i+1)*c+j] += c10; 
            res[(i+2)*c+j] += c20;
            break;
        }
        default: return 1;
        }
        break;
    }
    case 2:{
        j=0;
        for(;j+4<=c;j+=4){
            Ret c00=0,c01=0,c02=0,c03=0,
            c10=0,c11=0,c12=0,c13=0;
            Ret a0k, a1k, bk0, bk1, bk2, bk3;
            a0k_p = mat1+i*b;
            a1k_p = mat1+(i+1)*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; bk1 = mat2[k*c+1+j]; bk2 = mat2[k*c+2+j]; bk3 = mat2[k*c+3+j];
                a0k = *a0k_p++; a1k = *a1k_p++;
                c00 += a0k*bk0; c01 += a0k*bk1; c02 += a0k*bk2; c03 += a0k*bk3;
                c10 += a1k*bk0; c11 += a1k*bk1; c12 += a1k*bk2; c13 += a1k*bk3;
            }
            res[i*c+j] += c00; res[i*c+j+1] += c01; res[i*c+j+2] += c02; res[i*c+j+3] += c03;
            res[(i+1)*c+j] += c10; res[(i+1)*c+j+1] += c11; res[(i+1)*c+j+2] += c12; res[(i+1)*c+j+3] += c13;
        }
        if(j==c) return 0;
        size_t rest_cnt = c-j;
        switch(rest_cnt){
        case 3:{
            Ret c00=0,c01=0,c02=0,
            c10=0,c11=0,c12=0;
            Ret a0k, a1k, bk0, bk1, bk2;
            a0k_p = mat1+i*b;
            a1k_p = mat1+(i+1)*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; bk1 = mat2[k*c+1+j]; bk2 = mat2[k*c+2+j];
                a0k = *a0k_p++; a1k = *a1k_p++;
                c00 += a0k*bk0; c01 += a0k*bk1; c02 += a0k*bk2;
                c10 += a1k*bk0; c11 += a1k*bk1; c12 += a1k*bk2;
            }
            res[i*c+j] += c00; res[i*c+j+1] += c01; res[i*c+j+2] += c02;
            res[(i+1)*c+j] += c10; res[(i+1)*c+j+1] += c11; res[(i+1)*c+j+2] += c12;
            break;
        }
        case 2:{
            Ret c00=0,c01=0,
            c10=0,c11=0;
            Ret a0k, a1k, bk0, bk1;
            a0k_p = mat1+i*b;
            a1k_p = mat1+(i+1)*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; bk1 = mat2[k*c+1+j];
                a0k = *a0k_p++; a1k = *a1k_p++;
                c00 += a0k*bk0; c01 += a0k*bk1; 
                c10 += a1k*bk0; c11 += a1k*bk1;
            }
            res[i*c+j] += c00; res[i*c+j+1] += c01; 
            res[(i+1)*c+j] += c10; res[(i+1)*c+j+1] += c11;
            break;
        }
        case 1:{
            Ret c00=0,
            c10=0;
            Ret a0k, a1k, bk0;
            a0k_p = mat1+i*b;
            a1k_p = mat1+(i+1)*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; 
                a0k = *a0k_p++; a1k = *a1k_p++; 
                c00 += a0k*bk0; 
                c10 += a1k*bk0; 
            }
            res[i*c+j] += c00; 
            res[(i+1)*c+j] += c10;
            break;
        }
        default: return 1;
        }
        break;
    }
    case 1:{
        j=0;
        for(;j+4<=c;j+=4){
            Ret c00=0,c01=0,c02=0,c03=0;
            Ret a0k, bk0, bk1, bk2, bk3;
            a0k_p = mat1+i*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; bk1 = mat2[k*c+1+j]; bk2 = mat2[k*c+2+j]; bk3 = mat2[k*c+3+j];
                a0k = *a0k_p++;
                c00 += a0k*bk0; c01 += a0k*bk1; c02 += a0k*bk2; c03 += a0k*bk3;
            }
            res[i*c+j] += c00; res[i*c+j+1] += c01; res[i*c+j+2] += c02; res[i*c+j+3] += c03;
        }
        if(j==c) return 0;
        size_t rest_cnt = c-j;
        switch(rest_cnt){
        case 3:{
            Ret c00=0,c01=0,c02=0;
            Ret a0k, bk0, bk1, bk2;
            a0k_p = mat1+i*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; bk1 = mat2[k*c+1+j]; bk2 = mat2[k*c+2+j];
                a0k = *a0k_p++;
                c00 += a0k*bk0; c01 += a0k*bk1; c02 += a0k*bk2;
            }
            res[i*c+j] += c00; res[i*c+j+1] += c01; res[i*c+j+2] += c02;
            break;
        }
        case 2:{
            Ret c00=0,c01=0;
            Ret a0k, bk0, bk1;
            a0k_p = mat1+i*b;
            for(size_t k=0;k<b;++k){
                bk0 = mat2[k*c+j]; bk1 = mat2[k*c+1+j];
                a0k = *a0k_p++;
                c00 += a0k*bk0; c01 += a0k*bk1; 
            }
            res[i*c+j] += c00; res[i*c+j+1] += c01;
            break;
        }
        case 1:{
            Ret c00=0;
            a0k_p = mat1+i*b;
            for(size_t k=0;k<b;++k){
                c00 += (*a0k_p++)*mat2[k*c+j];
            }
            res[i*c+j] += c00;
            break;
        }
        default: return 1;
        }
        break;
    }
    default: return 1;
    }
    return 0;
}




template<class T1, class T2, class Ret = op_ret_t<EOperation::MUL, T1, T2>>
void _naive_matmul(
    T1 *mat1, T2 *mat2, 
    Ret *res, 
    size_t a, size_t b, size_t c
){
    if(a*b*c==0) return;
    if((!mat1)||(!mat2)||(!res)) return;
    for(size_t i=0;i<a;++i){
        for(size_t j=0;j+4<=c;++j){
            Ret tmp = 0;
            for(size_t k=0;k<b;++k)
                tmp += mat1[i*b+k]*mat2[k*c+j];
            res[i*c+j] = tmp;
        }
    }
}

}