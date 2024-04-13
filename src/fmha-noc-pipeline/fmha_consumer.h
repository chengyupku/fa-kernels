#pragma once

#include "online_softmax.h"
#include "reg2reg.h"
#include "shared_storage.h"
#include "noc_config.h"
#include "utils.h"

// FMHA Consumer does GEMMs and softmax
template <class Gemm1Type, class AccumType, class SoftType, class Gemm2Type,
          class TiledMma0, class TiledMma1, class TileShapeS, class GmemLayoutS,
          typename TensorQ, typename TensorK, typename TensorS,
          typename TensorV, typename TensorO, typename TensorSS, typename TensorSR,
          typename RegLayout, typename Reg2Reg,
          typename RowMax, typename RowSum, typename FullBarrier, typename EmptyBarrier>
__device__ static void //__launch_bounds__(128, 2)
fmhaForwardConsumer(Gemm1Type const *Q, Gemm1Type const *K, Gemm2Type const *V,
                    Gemm1Type *S, const TensorQ &tSrQ, const TensorK &tSrK,
                    TensorS &&tSrS, const TensorV &tOrV, TensorO &tOrO,
                    const RegLayout &tOrPLayout, Reg2Reg & reg2reg, RowMax &rowMax, RowSum &rowSum,
                    const TileShapeS &tileShapeS,
                    const GmemLayoutS &gmemLayoutS, float scale, int blockIdxY,
                    const TiledMma0 &tiledMma0, const TiledMma1 &tiledMma1,
                    FullBarrier* send_mbar_ptr, EmptyBarrier* recv_mbar_ptr,
                    TensorSS& sS, TensorSR& sR,
                    uint32_t& producer_phase, uint32_t& consumer_phase,
                    const AccumType &, const SoftType &) {

  using namespace cute;

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  dim3 cluster_shape = cluster.dim_blocks();
  uint32_t clusterBlockRank = cluster.block_rank();

  auto synchronize = [&] () { cutlass::arch::NamedBarrier::sync(size(TiledMma0{}), 1); };

  // NOTICE: threadIdx.x may be a bug
  auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);

  clear(tSrS);

  // Issue GEMM-I.
  cfk::gemm(tiledMma0, tSrQ, tSrK, tSrS);

  constexpr int NoCIter = log2OfPowerOfTwo(SplitNum);
  for (int iter = 0; iter < NoCIter/*log(KDimSplitNum)+1*/; iter++) {
    uint32_t src_group_size = iter == NoCIter - 1 ? 1 : 1 << (iter + 1);
    uint32_t dst_group_size = 1 << iter;
    uint32_t src_id = (clusterBlockRank + src_group_size) % (src_group_size << 1) 
                    + (clusterBlockRank / (src_group_size << 1)) * (src_group_size << 1);
    uint32_t dst_id = (clusterBlockRank + dst_group_size) % (dst_group_size << 1) 
                    + (clusterBlockRank / (dst_group_size << 1)) * (dst_group_size << 1);

    if (threadIdx.x == 128) {
      noc::arrive_empty(recv_mbar_ptr, src_id);
    }

    Tensor tSsS = threadMma0.partition_C(sS(_,_,0));
    Tensor tSsR = threadMma0.partition_C(sR(_,_,0));
    copy(tSrS, tSsS);
    synchronize();

    // Send local (partial) acc_s to neighbour
    // cluster.sync();
    if (threadIdx.x == 128) {
      uint32_t transaction_bytes = size(sS(_,_,0)) * sizeof(AccumType)/*sizeof float*/;
      uint32_t src_int_addr = cast_smem_ptr_to_uint(sS(_,_,0).data().get().get());
      uint32_t smem_int_mbar = set_block_rank(cast_smem_ptr_to_uint(send_mbar_ptr), dst_id);
      uint32_t remote_addr = set_block_rank(cast_smem_ptr_to_uint(sR(_,_,0).data().get().get()), dst_id);
      noc::wait_empty(recv_mbar_ptr, producer_phase);
      producer_phase ^= 1;
      noc::dsmem_copy_prepare(send_mbar_ptr, transaction_bytes, dst_id);
      noc::dsmem_copy_func(src_int_addr, remote_addr, smem_int_mbar, transaction_bytes);
    }
    // Wait until receiving all data
    if (threadIdx.x == 128 + 32 * 1) {
      noc::consumer_wait(send_mbar_ptr, consumer_phase);
      consumer_phase ^= 1;
    }
    synchronize();

    Tensor tSrR = make_fragment_like(tSrS);
    copy(tSsR, tSrR);
    synchronize();
    
    Tensor A = flatten(make_tensor(tSrS(_,0,0).data(), tSrS(_,0,0).layout()));
    Tensor B = flatten(make_tensor(tSrR(_,0,0).data(), tSrR(_,0,0).layout()));
    // if (cute::thread0()) {print("A:"); print(A.layout()); print("\n");}
    // if (cute::thread0()) {print("B:"); print(B.layout()); print("\n");}
    noc::reduce_in_place(A, B);
    synchronize();
  }

// Required for verification ONLY.
#ifdef COPYOUTMM0
  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.z % H);
  auto blockIdxB = uint64_t(blockIdx.z / H);
  auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);
  Tensor mS = make_tensor(make_gmem_ptr(S), gmemLayoutS);
  auto blkCoordS = make_coord(blockIdxX, blockIdxY, blockIdxH, blockIdxB);
  Tensor gS = local_tile(mS, tileShapeS, blkCoordS);
  Tensor tSgS = threadMma0.partition_C(gS);
  copy(tSrS, tSgS);
#endif

  if (blockIdxY == 0) { // Compute Online Softmax and NO Output Rescaling.
    onlineSoftmaxAndRescale<true, SoftType>(rowMax, rowSum, tSrS, tOrO, scale);
  } else { // Compute Online Softmax and Output Rescaling.
    onlineSoftmaxAndRescale<false, SoftType>(rowMax, rowSum, tSrS, tOrO, scale);
  }
  warpgroup_fence_operand(tSrS);

  // ISSUE GEMM-II with Operand A from RMEM.
  // Convert Operand A from SoftType [=float or half] to Gemm2Type [=half_t or
  // fp8] before issuing.
  auto tSrSPrec = convert_type<Gemm2Type, AccumType>(tSrS);
  // Invoke additional register permute/shuffle if GEMM-II is FP8.
#ifdef GEMM2FP8
  reg2reg(tSrSPrec);
#endif
  auto tOrP = make_tensor(tSrSPrec.data(), tOrPLayout);
  warpgroup_fence_operand(tSrS);
  // Issue GEMM-II.
#if 0
  if (cute::thread0()) {
        print("\n");
        print(tOrPLayout);
        print("\n");
        print(tOrV.layout());
        print("\n");
   }
#endif

  cfk::gemm(tiledMma1, tOrP, tOrV, tOrO);
}