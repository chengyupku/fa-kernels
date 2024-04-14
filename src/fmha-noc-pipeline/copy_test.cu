#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/tensor.hpp>
#include <cutlass/cluster_launch.hpp>

#include "cutlass/numeric_types.h"
#include <cutlass/cutlass.h>

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/transform/collective/sm90_wgmma_transpose.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/helper_cuda.hpp"
#include "cutlass/util/print_error.hpp"

#include "utils/cuda_launch.hpp"
#include "utils/fmha_cutlass.hpp"
#include "utils/random.hpp"

#include "gemm/copy_tensor.hpp"
#include "gemm/gemm_tensor.hpp"

#include "online_softmax.h"
#include "reg2reg.h"

#include "fmha_nopipe.h"
#include "fmha_pipe_nows.h"
#include "fmha_pipe_ws.h"
#include "ss_helper.h"
using namespace cute;

constexpr int WARMUP = 0;
constexpr int ITERATION = 10;
using bM = _64;
using bN = _64;
using bNx = _128;
using bKblock = _256;
using AccumType = half_t;
using SmemType = half_t;
using MmaA = half_t;
using MmaB = half_t;
using MmaC = float;
using MmaTileShape = Layout<Shape<_1, _1, _1>>;

namespace ops {
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
}

// __global__ void copy_test_kernel() {
//   extern __shared__ int data[];
//   using TiledMma0 = decltype(cute::make_tiled_mma(
//       ss_op_selector_custom<MmaA, MmaB, MmaC, Shape<bM, bN, bKblock>>(),
//       MmaTileShape{}));

//   auto synchronize = [&] () { cutlass::arch::NamedBarrier::sync(size(TiledMma0{}), 1); };
//   auto tileShapeS = make_shape(bM{}, bN{});
//   auto smemLayoutAtomS = cute::GMMA::Layout_K_SW128_Atom<SmemType>{};
//   auto smemLayoutS = tile_to_shape(
//       smemLayoutAtomS,
//       make_shape(bM{}, bNx{}, _1{}));

//   TiledMma0 tiledMma0;
//   auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);

//   Tensor sS =
//       make_tensor(make_smem_ptr(reinterpret_cast<SmemType*>(data) + 0       ), smemLayoutS);
//   Tensor sR =
//       make_tensor(make_smem_ptr(reinterpret_cast<SmemType*>(data) + size(sS)), smemLayoutS);
//   Tensor tSrS = partition_fragment_C(tiledMma0, tileShapeS);
//   for (int i = 0; i < size(tSrS); i++) {
//     tSrS(i) = float(threadIdx.x) + float(i) * 0.1;
//   }
  
//   // auto tSrXLayout = make_layout(make_shape(make_shape(_2{},_2{},_16{}),_1{},_1{}), make_stride(make_stride(_1{},_2{},_4{}),_0{},_0{}));
//   // Tensor tSrX = make_tensor(make_smem_ptr(reinterpret_cast<SmemType*>(tSrS.data())), tSrXLayout);


//   using CopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>;
//   TiledCopy tiled_copy_C_atom = make_tiled_copy_C_atom(CopyAtomC{}, tiledMma0);

//   // (t)hread-partition for (r)egister to (s)mem copy (tRS_)
//   TiledCopy tiled_r2s = make_tiled_copy_S(Copy_Atom<SM90_U32x4_STSM_N,SmemType>{}, tiled_copy_C_atom);
//   ThrCopy thread_r2s = tiled_r2s.get_slice(threadIdx.x);
//   // Tensor tRS_rAcc = thread_r2s.retile_S(tSrX);                                   // ((R2S,R2S_V),MMA_M,MMA_N)
//   Tensor tRS_sS   = thread_r2s.partition_D(sS);                                       // (R2S,R2S_M,R2S_N,PIPE_D)

//   // Allocate D registers
//   Layout tRS_rS_layout = make_layout(take<0,3>(shape(thread_r2s.partition_S(sS))));
//   Tensor tRS_rS = make_tensor(make_smem_ptr(reinterpret_cast<SmemType*>(tSrS.data())), tRS_rS_layout);                                          // (R2S,R2S_M,R2S_N)


//   // Tensor tSsS = threadMma0.partition_C(sS(_,_,0));
//   // Tensor tSsR = threadMma0.partition_C(sR(_,_,0));
  
//   // copy(DefaultCopy{}, tSrS, tSsS);
//   copy(tiled_r2s, tRS_rS, tRS_sS(_,_,_,0));
//   synchronize();

//   // if (cute::thread0()) {
//   //   print("tSrS:"); print_tensor(tSrS); print("\n"); 
//   //   // print("tSrX:"); print_tensor(tSrX); print("\n"); 
//   //   // print("tRS_rAcc:"); print_tensor(tRS_rAcc); print("\n"); 
//   //   // print("tRS_rS:"); print_tensor(tRS_rS); print("\n"); 
//   //   // print("tRS_sS:"); print_tensor(tRS_sS); print("\n");
//   //   // print("sS:"); print_tensor(sS); print("\n");
//   // }

//   clear(tSrS);
//   // if (cute::thread0()) {
//   //   print("tSrS:"); print_tensor(tSrS); print("\n"); 
//   // }

//   TiledCopy tiled_s2r = make_tiled_copy_S(Copy_Atom<SM75_U32x4_LDSM_N, SmemType>{}, tiled_copy_C_atom);
//   ThrCopy thread_s2r = tiled_s2r.get_slice(threadIdx.x);
//   Tensor tSR_sC = thread_s2r.partition_S(sS);                                  // (S2R,S2R_M,S2R_N,PIPE_C)

//   Tensor tRS_rC = make_tensor(make_smem_ptr(reinterpret_cast<SmemType*>(tSrS.data())), tRS_rS_layout);
//   Tensor tSR_rC = thread_s2r.retile_D(tRS_rC);                                                   // (S2R,S2R_M,S2R_N)
//   copy(tiled_s2r, tSR_sC(_,_,_,0), tSR_rC);
//   // if (cute::thread0()) {
//   //   print("tSrS:"); print_tensor(tSrS); print("\n"); 
//   // }
// }

__global__ void bug_copy_kernel() {
  extern __shared__ char data[];
  using bNX = _128;
  auto smemLayoutS =
      tile_to_shape(GMMA::Layout_K_SW128_Atom<cutlass::half_t>{}, make_shape(bM{}, bNX{}));
  using CopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>;
  using MmaTileShape = Layout<Shape<_1, _1, _1>>;
  Tensor sS =
      make_tensor(make_smem_ptr(reinterpret_cast<SmemType *>(data)), smemLayoutS);
  Tensor sR =
      make_tensor(make_smem_ptr(reinterpret_cast<SmemType *>(data) + size(sS)), smemLayoutS);
  
  TiledCopy tiled_copy_C_atom = make_tiled_copy_C_atom(CopyAtomC{}, cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<cutlass::half_t, cutlass::half_t, float, Shape<_64, _128, _256>>(),
  MmaTileShape{}));
  TiledCopy tiled_r2s = make_tiled_copy_S(Copy_Atom<SM90_U32x4_STSM_N,SmemType>{}, tiled_copy_C_atom);
  ThrCopy thread_r2s = tiled_r2s.get_slice(threadIdx.x);
  Tensor tRS_sS   = thread_r2s.partition_D(sS);                                       // (R2S,R2S_M,R2S_N,PIPE_D)
  Layout tRS_rS_layout = make_layout(take<0,3>(shape(thread_r2s.partition_S(sS))));
  
  using TiledMma0 = decltype(cute::make_tiled_mma(
      cute::GMMA::ss_op_selector<MmaA, MmaB, MmaC, Shape<bM, bN, bKblock>>(),
      MmaTileShape{}));
  TiledMma0 tiledMma0;
  auto tileShapeS = make_shape(bM{}, bN{});

  #pragma unroll
  for (uint64_t blockIdxY = 0; blockIdxY < 64; ++blockIdxY) {
    Tensor tSrS = partition_fragment_C(tiledMma0, tileShapeS);
    Tensor tSrR = make_fragment_like(tSrS);
    Tensor tRS_rS = make_tensor(make_smem_ptr(reinterpret_cast<SmemType*>(tSrS.data())), tRS_rS_layout);                                          // (R2S,R2S_M,R2S_N)
    copy(tiled_r2s, tRS_rS, tRS_sS);

    __syncthreads();


    TiledCopy tiled_s2r = make_tiled_copy_S(Copy_Atom<SM75_U32x4_LDSM_N, SmemType>{}, tiled_copy_C_atom);
    ThrCopy thread_s2r = tiled_s2r.get_slice(threadIdx.x);
    Tensor tSR_sC = thread_s2r.partition_S(sR);                                  // (S2R,S2R_M,S2R_N,PIPE_C)
    Tensor tRS_rC = make_tensor(make_smem_ptr(reinterpret_cast<SmemType*>(tSrR.data())), tRS_rS_layout);
    Tensor tSR_rC = thread_s2r.retile_D(tRS_rC);                                                   // (S2R,S2R_M,S2R_N)
    copy(tiled_s2r, tSR_sC, tSR_rC);

    Tensor A = flatten(make_tensor(tSrS(_,0,0).data(), tSrS(_,0,0).layout()));
    Tensor B = flatten(make_tensor(tSrR(_,0,0).data(), tSrR(_,0,0).layout()));
    ops::reduce_in_place(A, B);
  }
}

int main() {
  dim3 grid(64, 2, 16);
  dim3 block(128, 1, 1);
  int dyn_smem = 98336;
  cudaFuncSetAttribute(bug_copy_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);
  // cudaFuncSetAttribute(bug_copy_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  for (int i = 0; i < WARMUP; i++) {
    bug_copy_kernel<<<grid, block, dyn_smem>>>();
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
  for (int i = 0; i < ITERATION; i++) {
    bug_copy_kernel<<<grid, block, dyn_smem>>>();
  }
  // cudaDeviceSynchronize();
  cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time, start, stop);
	printf("latency: %f ms\n", time / float(ITERATION));
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
  return 0;
}
