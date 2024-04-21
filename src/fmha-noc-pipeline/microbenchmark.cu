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
#include "utils.h"
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

// __global__ void __cluster_dims__(1,2,1) warpspecialize_noc() {
//   extern __shared__ char data[];
//   using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
//   using EmptyBarrier = cutlass::arch::ClusterBarrier;

//   FullBarrier  *send_mbar_ptr = reinterpret_cast<FullBarrier *>(data + 32768);
//   EmptyBarrier *recv_mbar_ptr = reinterpret_cast<EmptyBarrier*>(data + 32768 + 64);
//   uint32_t producer_phase = 0;
//   uint32_t consumer_phase = 0;
//   if (threadIdx.x == 0) {
//     send_mbar_ptr[0].init(1);
//     recv_mbar_ptr[0].init(1);
//   }

//   namespace cg = cooperative_groups;
//   cg::cluster_group cluster = cg::this_cluster();
//   dim3 cluster_shape = cluster.dim_blocks();
//   uint32_t clusterBlockRank = cluster.block_rank();

//   auto synchronize = [&] () { cutlass::arch::NamedBarrier::sync(128, 1); };

//   cluster.sync();
//   if (threadIdx.x >= 128) {
//     for (int i = 0; i < 64; i++) {
//       if (threadIdx.x == 128) {
//         noc::arrive_empty(recv_mbar_ptr, clusterBlockRank ^ 1);
//         noc::wait_empty(recv_mbar_ptr, producer_phase);
//       }
//       synchronize();
//       if (threadIdx.x == 128) {
//         uint32_t transaction_bytes = 16384;
//         uint32_t src_int_addr = cast_smem_ptr_to_uint(data);
//         uint32_t smem_int_mbar = set_block_rank(cast_smem_ptr_to_uint(send_mbar_ptr), clusterBlockRank ^ 1);
//         uint32_t remote_addr = set_block_rank(cast_smem_ptr_to_uint(data + transaction_bytes), clusterBlockRank ^ 1);
//         // noc::wait_empty(recv_mbar_ptr, producer_phase);
//         producer_phase ^= 1;
//         noc::dsmem_copy_prepare(send_mbar_ptr, transaction_bytes, clusterBlockRank ^ 1);
//         noc::dsmem_copy_func(src_int_addr, remote_addr, smem_int_mbar, transaction_bytes);
//       }
//       // Wait until receiving all data
//       if (threadIdx.x == 128 + 32 * 1) {
//         noc::consumer_wait(send_mbar_ptr, consumer_phase);
//         consumer_phase ^= 1;
//       }
//       synchronize();
//     }
//   }

//   cluster.sync();
// }

// int main() {
//   dim3 grid(64, 2, 16);
//   dim3 block(256, 1, 1);
//   int dyn_smem = 98688;
//   cudaFuncSetAttribute(warpspecialize_noc, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);
//   cudaFuncSetAttribute(warpspecialize_noc, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
//   for (int i = 0; i < WARMUP; i++) {
//     warpspecialize_noc<<<grid, block, dyn_smem>>>();
//   }
//   cudaDeviceSynchronize();

//   cudaEvent_t start, stop;
// 	cudaEventCreate(&start);
// 	cudaEventCreate(&stop);
// 	cudaEventRecord(start);
//   for (int i = 0; i < ITERATION; i++) {
//     warpspecialize_noc<<<grid, block, dyn_smem>>>();
//   }
//   // cudaDeviceSynchronize();
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);
//   float time;
//   cudaEventElapsedTime(&time, start, stop);
//   cudaError_t error = cudaGetLastError();
//   if (error != cudaSuccess) {
//     printf("CUDA error: %s\n", cudaGetErrorString(error));
//   }
//   printf("latency: %f ms\n", time / float(ITERATION));
//   cudaEventDestroy(start);
//   cudaEventDestroy(stop);
//   return 0;
// }



// __global__ void __cluster_dims__(1,2,1) non_warpspecialize_noc() {
//   extern __shared__ char data[];
//   using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
//   using EmptyBarrier = cutlass::arch::ClusterBarrier;

//   FullBarrier  *send_mbar_ptr = reinterpret_cast<FullBarrier *>(data + 32768);
//   EmptyBarrier *recv_mbar_ptr = reinterpret_cast<EmptyBarrier*>(data + 32768 + 64);
//   uint32_t producer_phase = 0;
//   uint32_t consumer_phase = 0;
//   if (threadIdx.x == 0) {
//     send_mbar_ptr[0].init(1);
//     recv_mbar_ptr[0].init(1);
//   }

//   namespace cg = cooperative_groups;
//   cg::cluster_group cluster = cg::this_cluster();
//   dim3 cluster_shape = cluster.dim_blocks();
//   uint32_t clusterBlockRank = cluster.block_rank();

//   cluster.sync();
//   for (int i = 0; i < 64; i++) {
//     if (threadIdx.x == 0) {
//       noc::arrive_empty(recv_mbar_ptr, clusterBlockRank ^ 1);
//       noc::wait_empty(recv_mbar_ptr, producer_phase);
//     }
//     __syncthreads();
//     if (threadIdx.x == 0) {
//       uint32_t transaction_bytes = 16384;
//       uint32_t src_int_addr = cast_smem_ptr_to_uint(data);
//       uint32_t smem_int_mbar = set_block_rank(cast_smem_ptr_to_uint(send_mbar_ptr), clusterBlockRank ^ 1);
//       uint32_t remote_addr = set_block_rank(cast_smem_ptr_to_uint(data + transaction_bytes), clusterBlockRank ^ 1);
//       // noc::wait_empty(recv_mbar_ptr, producer_phase);
//       producer_phase ^= 1;
//       noc::dsmem_copy_prepare(send_mbar_ptr, transaction_bytes, clusterBlockRank ^ 1);
//       noc::dsmem_copy_func(src_int_addr, remote_addr, smem_int_mbar, transaction_bytes);
//     }
//     // Wait until receiving all data
//     if (threadIdx.x == 32) {
//       noc::consumer_wait(send_mbar_ptr, consumer_phase);
//       consumer_phase ^= 1;
//     }
//     __syncthreads();
//   }

//   cluster.sync();
// }

// int main() {
//   dim3 grid(64, 2, 16);
//   dim3 block(128, 1, 1);
//   int dyn_smem = 98688;
//   cudaFuncSetAttribute(non_warpspecialize_noc, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);
//   cudaFuncSetAttribute(non_warpspecialize_noc, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
//   for (int i = 0; i < WARMUP; i++) {
//     non_warpspecialize_noc<<<grid, block, dyn_smem>>>();
//   }
//   cudaDeviceSynchronize();

//   cudaEvent_t start, stop;
// 	cudaEventCreate(&start);
// 	cudaEventCreate(&stop);
// 	cudaEventRecord(start);
//   for (int i = 0; i < ITERATION; i++) {
//     non_warpspecialize_noc<<<grid, block, dyn_smem>>>();
//   }
//   // cudaDeviceSynchronize();
//   cudaEventRecord(stop);
//   cudaEventSynchronize(stop);
//   float time;
//   cudaEventElapsedTime(&time, start, stop);
//   cudaError_t error = cudaGetLastError();
//   if (error != cudaSuccess) {
//     printf("CUDA error: %s\n", cudaGetErrorString(error));
//   }
//   printf("latency: %f ms\n", time / float(ITERATION));
//   cudaEventDestroy(start);
//   cudaEventDestroy(stop);
//   return 0;
// }

__global__ void __cluster_dims__(2,1,1) dim_1024() {
  extern __shared__ char data[];
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;

  FullBarrier  *send_mbar_ptr = reinterpret_cast<FullBarrier *>(data + 32768);
  EmptyBarrier *recv_mbar_ptr = reinterpret_cast<EmptyBarrier*>(data + 32768 + 64);
  uint32_t producer_phase = 0;
  uint32_t consumer_phase = 0;
  if (threadIdx.x == 0) {
    send_mbar_ptr[0].init(1);
    recv_mbar_ptr[0].init(1);
  }

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  dim3 cluster_shape = cluster.dim_blocks();
  uint32_t clusterBlockRank = cluster.block_rank();

  cluster.sync();
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < 64; i++) {
    uint32_t src_group_size = 1;
    uint32_t src_id = (clusterBlockRank + src_group_size) % (src_group_size << 1) 
                    + (clusterBlockRank / (src_group_size << 1)) * (src_group_size << 1);
    if (threadIdx.x == 0) {
      // if (threadIdx.x==0 && blockIdx.x==0 && blockIdx.z==0) {
      //   printf("blockIdx.y:%d, iter:%d:0\n", blockIdx.y, i);
      // }
      noc::arrive_empty(recv_mbar_ptr, src_id);
      // if (threadIdx.x==0 && blockIdx.x==0 && blockIdx.z==0) {
      //   printf("blockIdx.y:%d, iter:%d:1\n", blockIdx.y, i);
      // }
      noc::wait_empty(recv_mbar_ptr, producer_phase);
      // if (threadIdx.x==0 && blockIdx.x==0 && blockIdx.z==0) {
      //   printf("blockIdx.y:%d, iter:%d:2\n", blockIdx.y, i);
      // }
      producer_phase ^= 1; 
    }
    // cluster.sync();
    // __syncthreads();
  }

  cluster.sync();
}

int main() {
  dim3 grid(2, 64, 16);
  dim3 block(128, 1, 1);
  int dyn_smem = 98688;
  cudaFuncSetAttribute(dim_1024, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);
  cudaFuncSetAttribute(dim_1024, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  for (int i = 0; i < WARMUP; i++) {
    dim_1024<<<grid, block, dyn_smem>>>();
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
  for (int i = 0; i < ITERATION; i++) {
    dim_1024<<<grid, block, dyn_smem>>>();
  }
  // cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time, start, stop);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
  }
  printf("latency: %f ms\n", time / float(ITERATION));
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}
