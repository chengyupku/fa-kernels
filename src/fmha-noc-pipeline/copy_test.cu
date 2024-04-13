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

constexpr int WARMUP = 10;
constexpr int ITERATION = 1000;
using bM = _128;
using bN = _64;
using bKblock = _256;
using AccumType = float;
using MmaA = half_t;
using MmaB = half_t;
using MmaC = float;
using MmaTileShape = Layout<Shape<_2, _1, _1>>;

__global__ void copy_test_kernel() {
  extern __shared__ int data[];
  using TiledMma0 = decltype(cute::make_tiled_mma(
      ss_op_selector_custom<MmaA, MmaB, MmaC, Shape<bM, bN, bKblock>>(),
      MmaTileShape{}));

  auto synchronize = [&] () { cutlass::arch::NamedBarrier::sync(size(TiledMma0{}), 1); };
  auto tileShapeS = make_shape(bM{}, bN{});
  auto smemLayoutAtomS = cute::GMMA::Layout_K_SW32_Atom<MmaC>{};
  auto smemLayoutS = tile_to_shape(
      smemLayoutAtomS,
      make_shape(bM{}, bN{}, _1{}));

  TiledMma0 tiledMma0;
  auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);

  Tensor sS =
      make_tensor(make_smem_ptr(reinterpret_cast<AccumType*>(data) + 0       ), smemLayoutS);
  Tensor sR =
      make_tensor(make_smem_ptr(reinterpret_cast<AccumType*>(data) + size(sS)), smemLayoutS);
  Tensor tSrS = partition_fragment_C(tiledMma0, tileShapeS);


  Tensor tSsS = threadMma0.partition_C(sS(_,_,0));
  Tensor tSsR = threadMma0.partition_C(sR(_,_,0));
  copy(tSrS, tSsS);
  synchronize();
}

int main() {
  dim3 grid(1024, 1, 1);
  dim3 block(256, 1, 1);
  int dyn_smem = 227 * 1024;
  cudaFuncSetAttribute(copy_test_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_smem);
  // cudaFuncSetAttribute(copy_test_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  for (int i = 0; i < WARMUP; i++) {
    copy_test_kernel<<<grid, block, dyn_smem>>>();
  }
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
  for (int i = 0; i < ITERATION; i++) {
    copy_test_kernel<<<grid, block, dyn_smem>>>();
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
