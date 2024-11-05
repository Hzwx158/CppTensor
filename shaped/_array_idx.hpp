#ifndef NUMCPP_SHAPED_INDEX_HPP
#define NUMCPP_SHAPED_INDEX_HPP
#include "./array.hpp"
namespace numcpp{

/**
 * @brief 获取bool下标对应的数字下标。注意，这里的下标和np.where不一样，是每个点的下标
 * @param boolArray bool下标
 * @return 每个点的下标
 */
std::vector<FixedArray<size_t>> where(ShapedArray<bool> const &boolArray)
{
    Shape const &shape = boolArray.getShape();
    auto ptr = boolArray.data();
    const auto bufSize = boolArray.getSize();
    std::vector<FixedArray<size_t>> res(bufSize, FixedArray<size_t>{});
    size_t k=0;
    for(size_t i=0;i<bufSize;++i){
        if(ptr[i]){
            res[k++] = shape.indexOf(i); 
            // std::cout<< res[k-1] <<std::endl;
        }
    }
    res.erase(res.begin()+k, res.end());
    res.shrink_to_fit();
    return res;
}

template<class Number>
static inline ShapedArray<long long> to_shaped(Number n){
    return ShapedArray<long long>(static_cast<long long>(n));
}
static inline ShapedArray<Slice> to_shaped(Slice const &slc){
    return ShapedArray<Slice>(slc);
}
template<class DType>
static inline ShapedArray<DType> const &to_shaped(ShapedArray<DType> const &obj){
    return obj;
}

/**
 * @brief 获取每个所取点的tuple of size_t下标和Shape.
 * `$src` [`N`-sized tuple of `$idx`] = $(`idx` + `src`[`N`:])（在没有slice的情况下）
 * @param shape 被取下标的ShapedArray的shape
 * @param index0, indices 下标
 * @return 一个vector和一个Shape；vector每个元素代表每个点的下标, Shape代表结果的形状.
 */
template<class Index0, class ...Indices>
std::pair<
    std::vector<FixedArray<size_t>>,
    Shape
> getUllIndices(
    Shape const &shape, 
    Index0 const &index0, 
    Indices const &... indices
){
    size_t dimSize0 = shape.dimSizeOf(0);
    // 1.先把非tuple of xxx的解决了
    if constexpr(sizeof...(Indices)==0 && (!is_ShapedArray_v<Index0>)){
        if constexpr(std::is_same_v<Index0, Slice>){
            // 一个Slice下标：直接调用函数获取下标序列
            auto idx = index0.getIndices(dimSize0);
            std::vector<FixedArray<size_t>> res;
            for(size_t &i:idx)
                res.push_back(FixedArray<size_t>({i}));
            return {std::move(res), Shape({idx.size()})+shape.sliced(1)};
        }
        else{
            // 一个size_t下标
            size_t idx;
            if(!toBoundedIndex(index0, dimSize0, &idx))
                throw Error::outOfRange(__FILE__, __func__, index0, 0, dimSize0);
            return {std::vector{FixedArray{idx}}, shape.sliced(1)};
        }
    }
    else{ //此else不可省，to编译
    // 2.都转为ShapedArray下标
    std::tuple<decltype(to_shaped(index0)),decltype(to_shaped(std::declval<Indices const &>()))...> 
    shaped_indices = {
        to_shaped(index0),
        to_shaped(indices)...
    };
    // 3.计算广播后shape
    static auto get_broadcast_shape = [](auto const &... args)->Shape
    {
        // ([](auto const &a){
        //     std::cout<<"&b:"<<&a<<&a.getShape()<<std::endl;
        // }(args),...);
        return Shape::broadcast({numcpp::shapeOf(args)...});
    };
    // std::apply([](auto const &...t){
    //     ([](auto const &a){
    //         std::cout<<"&a:"<<&a<<"\tshape:"<<&a.getShape()<<std::endl;
    //     }(t),...);
    // }, shaped_indices);
    Shape final_shape = std::apply(get_broadcast_shape, shaped_indices);
    // 4.特判：如果是Slice，递归
    if constexpr(std::is_same_v<Index0, Slice>){
        auto [rest, rest_shape] = getUllIndices(shape.sliced(1), indices...);
        auto cur_indices = index0.getIndices(dimSize0);
        std::vector<FixedArray<size_t>> res{};
        for(auto &cur_idx:cur_indices){
            auto tmp = FixedArray<size_t>({cur_idx});
            for(auto &rest_idc:rest){
                res.push_back(tmp + rest_idc);
            }
        }
        return {std::move(res), Shape({cur_indices.size()})+rest_shape};
    }
    else{
    // 5.特判：如果是tuple of number/slice    
    if(final_shape.isNumberShape()){
        // 4.1 后面有Slice, 递归
        if constexpr(is_ShapedArray_v<Index0> || (is_ShapedArray_v<Indices> || ...)); //必须加这句，不然编译不过
        else if constexpr((std::is_same_v<Indices, Slice> || ...)){
            //本次是size_t, 直接结合即可
            auto [rest, rest_shape] = getUllIndices(shape.sliced(1), indices...);
            auto tmp = FixedArray<size_t>(1Ull,(size_t)index0);
            for(auto &rest_idc:rest){
                rest_idc = tmp + rest_idc;
            }
            return {rest, rest_shape};
        }
        // 4.2 tuple of number：直接返回
        else return {
            std::vector{FixedArray<size_t>{(size_t)index0, static_cast<size_t>(indices)...}},
            shape.sliced(sizeof...(Indices)+1)
        };
    }
    // 6.对tuple of iterable的每一个any，取at的结果
    size_t idx_final_offset=0;
    auto get_i_th_tuple_helper = [&final_shape, &shape, &idx_final_offset](auto const &...args) 
    {
        //该函数用于计算tuple of any的每一个的第i项。
        return getUllIndices(shape,
            (*(args.data()+Shape::offsetBeforeBroadcast( //计算的是每一个的第i项
                idx_final_offset, final_shape, args.getShape()
            )))...
        );
    };
    size_t final_bufSize = final_shape.bufSize();
    std::vector<FixedArray<size_t>> res;
    bool flag = true;
    Shape res_shape;
    for(; idx_final_offset<final_bufSize; ++idx_final_offset){
        auto tmp = std::apply(get_i_th_tuple_helper, shaped_indices);
        if(flag){
            res_shape = final_shape + tmp.second;
            // std::cout<<"second:"<<tmp.second<<std::endl;
            flag = false;
        }
        for(auto &m:tmp.first){
            res.push_back(std::move(m));
        }
    }
    return {res, res_shape};

}}}


template<class DType>
template<class ...Args>
ShapedArray<DType*> ShapedArray<DType>::at(Args const &... indices)
{
    auto [index_array, res_shape] = getUllIndices(this->shape, indices...);
    ShapedArray<DType*> res = numcpp::fill<DType*>(nullptr, res_shape);
    DType **dst = res.data();
    for(Shape::SizeTArray &idx:index_array){
        auto [offset, eleCnt] = shape.offsetOf(idx);
        for(size_t i=0; i<eleCnt; ++i)
            *(dst++) = mArray + offset + i;
    }
    return res;
}
template<class DType>
template<class ...Args>
ShapedArray<DType> ShapedArray<DType>::at(Args const &... indices) const
{
    auto [index_array, res_shape] = getUllIndices(this->shape, indices...);
    ShapedArray<DType> res = numcpp::fill<DType>(DType(), res_shape);
    DType *dst = res.data();
    for(Shape::SizeTArray &idx:index_array){
        auto [offset, eleCnt] = shape.offsetOf(idx);
        memcpy(dst, mArray+offset, eleCnt*sizeof(DType));
        dst+=eleCnt;
    }
    return res;
}
template<class DType>
ShapedArray<DType> ShapedArray<DType>::at(ShapedArray<bool> const &cond) const
{
    auto indices = numcpp::where(cond);
    size_t point_cnt = indices.size();
    if(!indices.size())
        return {};
    Shape const &cond_shape = cond.getShape();
    ShapedArray<DType> res = numcpp::fill(DType(), Shape{point_cnt*shape.stepSizeOf(indices[0].size()-1)});
    auto *ptr = res.data();
    for(auto &index:indices){
        auto [offset, cnt] = cond_shape.offsetOf(index);
        memcpy(ptr, mArray+offset, sizeof(DType)*cnt);
        ptr += cnt;
    }
    return res;
}
template<class DType>
ShapedArray<DType*> ShapedArray<DType>::at(ShapedArray<bool> const &cond)
{
    auto indices = numcpp::where(cond);
    size_t point_cnt = indices.size();
    if(!indices.size())
        return {};
    auto res = numcpp::fill<DType*>(nullptr, Shape{point_cnt*shape.stepSizeOf(indices[0].size()-1)});
    auto *ptr = res.data();
    for(auto &index:indices){
        auto [offset, eleCnt] = shape.offsetOf(index);
        // std::cout<<"offset:"<<offset<<"\neleCnt:"<<eleCnt<<std::endl;
        for(size_t i=0; i<eleCnt; ++i)
            *(ptr++) = mArray + offset + i;
    }
    return res;
}

template<class DType>
template<class Functor>
void ShapedArray<DType>::apply_on(Functor &&func, FixedArray<size_t> const &index){
    size_t bufSize = shape.bufSize();
    if(index.size()==0)
        for(size_t i=0;i<bufSize;++i)
            func(mArray[i]);
    else{
        auto [offset, ele_cnt] = shape.offsetOf(index);
        for(size_t i=0;i<ele_cnt;++i)
            func(mArray[offset+i]);
    }
}



#define IDX(...) ShapedArray{ __VA_ARGS__ }

}
#endif