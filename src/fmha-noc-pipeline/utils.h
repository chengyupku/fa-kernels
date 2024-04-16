#pragma once

#include "cutlass/cutlass.h"
#include "cute/numeric/integral_constant.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"

constexpr int log2OfPowerOfTwo(int n) {
  return (n <= 1) ? 0 : 1 + log2OfPowerOfTwo(n / 2);
}

namespace noc {

////////////////////////////////////////////////////////////////////////////////////////////////////

using namespace cute;

template <typename EmptyBarrier>
inline __device__ void 
wait_empty(EmptyBarrier* empty_barrier_ptr_, uint32_t phase) {
  uint32_t done;
  done = (*empty_barrier_ptr_).test_wait(phase);
  if (not done) {
    (*empty_barrier_ptr_).wait(phase);
  }
}

template <typename FullBarrier>
inline __device__ void 
dsmem_copy_prepare(FullBarrier* full_barrier_ptr_, uint32_t transaction_bytes, uint32_t cta_id) {
  (*full_barrier_ptr_).arrive_and_expect_tx(transaction_bytes, cta_id);
}

template <typename FullBarrier>
inline __device__ void 
self_prepare(FullBarrier* full_barrier_ptr_, uint32_t transaction_bytes) {
    (*full_barrier_ptr_).arrive_and_expect_tx(transaction_bytes);
}

template <typename FullBarrier>
inline __device__ void 
consumer_wait(FullBarrier* full_barrier_ptr_, uint32_t phase) {
  uint32_t done;
  done = (*full_barrier_ptr_).test_wait(phase);
  if (not done) {
    (*full_barrier_ptr_).wait(phase);
  }
}

template <typename EmptyBarrier>
inline __device__ void 
arrive_empty(EmptyBarrier* empty_barrier_ptr_) {
  (*empty_barrier_ptr_).arrive();
}

template <typename EmptyBarrier>
inline __device__ void 
arrive_empty(EmptyBarrier* empty_barrier_ptr_, uint32_t cta_id) {
  (*empty_barrier_ptr_).arrive(cta_id);
}

CUTE_HOST_DEVICE void
dsmem_copy_func(uint32_t src_int_addr, 
                uint32_t remote_addr, 
                uint32_t smem_int_mbar, 
                uint32_t transaction_bytes)
{
  asm volatile (
      "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
      " [%0], [%1], %2, [%3];"
      :
      : "r"(remote_addr), "r"(src_int_addr), "r"(transaction_bytes), "r"(smem_int_mbar)
      : "memory"
  );
}

template<typename T>
struct SumOp {
__device__ inline T operator()(T const & x, T const & y) { return x + y; }
};

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void thread_reduce_2d(Tensor<Engine0, Layout0> &tensor0, Tensor<Engine1, Layout1> &tensor1, Operator &op) {
  // static_assert(Layout0::rank == 2, "Only support 2D Tensor");
  // static_assert(Layout1::rank == 2, "Only support 2D Tensor");
  CUTE_STATIC_ASSERT_V(size<0>(tensor0) == size<0>(tensor1));
  CUTE_STATIC_ASSERT_V(size<1>(tensor0) == size<1>(tensor1));
  CUTE_STATIC_ASSERT_V(size<2>(tensor0) == size<2>(tensor1));
  #pragma unroll
  for (int mi = 0; mi < size<0>(tensor0); mi++) {
    #pragma unroll
    for (int ni = 0; ni < size<1>(tensor0); ni++) {
      #pragma unroll
      for (int ki = 0; ki < size<2>(tensor0); ki++) {
        tensor0(mi, ni, ki) = op(tensor0(mi, ni, ki), tensor1(mi, ni, ki));
      }
    }
  }
}

template<typename Engine0, typename Layout0, typename Engine1, typename Layout1, typename Operator>
__device__ inline void reduce_in_place_(Tensor<Engine0, Layout0> &tensor0, Tensor<Engine1, Layout1> &tensor1, Operator &op) {
  thread_reduce_2d(tensor0, tensor1, op);
}

// tensor0 += tensor1
template<typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__device__ inline void reduce_in_place(Tensor<Engine0, Layout0> &tensor0, Tensor<Engine1, Layout1> &tensor1){
  SumOp<float> sum_op;
  reduce_in_place_(tensor0, tensor1, sum_op);
}

template<typename Layout>
inline __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
  // TD [2023-08-13]: Idk why but get<0, 1>(l) doesn't work for Cutlass 3.2, I'm getting
  // "int_tuple.hpp(74): error: conversion to inaccessible base class"
  // return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
  return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)), make_layout(get<0>(get<0>(l)), get<2>(l)));
};

}  // end namespace noc

constexpr int PatternLen = 4;
constexpr int IntraSplitNum = 2;

struct block_iter_id {
  int8_t x, iter;
};

__device__ constexpr int8_t tile_order[IntraSplitNum][PatternLen] = {
  {0, 1, 2, 3},
  {1, 0, 2, 3},
};
// __device__ constexpr block_iter_id srcKV[2][PatternLen] = {
//   {{-1, -1},{-1, -1}},
//   {{-1, -1},{-1, -1}},
// };

// __device__ constexpr block_iter_id dstKV[2][PatternLen] = {
//   {{-1, -1},{-1, -1}},
//   {{-1, -1},{-1, -1}},
// };
__device__ constexpr block_iter_id srcKV[IntraSplitNum][PatternLen] = {
  {{-1, -1},{1, 0},{-1, -1},{-1, -1}},
  {{-1, -1},{0, 0},{-1, -1},{-1, -1}},
};

__device__ constexpr block_iter_id dstKV[IntraSplitNum][PatternLen] = {
  {{1, 1},{-1, -1},{-1, -1},{-1, -1}},
  {{0, 1},{-1, -1},{-1, -1},{-1, -1}},
};