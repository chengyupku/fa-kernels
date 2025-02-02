/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*! \file
    \brief Driver for the pipelined and warp-specialized FMHA kernel.

    Based on the CUTLASS unit test for the PipelineTmaAsync class
    as it would be used in a warp-specialized loop.
*/

#pragma once

#define KERNEL_DBG_TRACE false

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/tensor.hpp>

#include <cutlass/cluster_launch.hpp>
#include <cutlass/util/reference/host/gemm.h>

#include "cutlass/core_io.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/print_error.hpp"

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/reg_reconfig.h"
// #include "cutlass/pipeline/pipeline.hpp"
#include "async_pipeline.hpp"
#include "fmha_consumer.h"
#include "fmha_epilogue.h"
#include "fmha_producer.h"
#include "shared_storage.h"
#include "noc_config.h"
#include "utils.h"

using namespace cute;
using namespace cutlass;

template<class ScheduleStorage>
CUTLASS_DEVICE void static
init_schedule(ScheduleStorage& shared_schedules) {
  if (threadIdx.x == 0) {
    for (int i = 0; i < IntraSplitNum; i++) {
      for (int j = 0; j < PatternLen; j++) {
        shared_schedules.tileOrder[i][j] = tile_order[i][j];
        shared_schedules.srcKV[i][j] = srcKV[i][j];
        shared_schedules.dstKV[i][j] = dstKV[i][j];
      }
    }
  }
}

template <class Gemm1Type, class AccumType, class SoftType, class Gemm2Type,
          class OutputType, class TiledMma0, class TiledMma1, class TiledMmaCvt0,
          class TiledCopyQ, class TileShapeQ, class GmemLayoutQ, class SmemLayoutQ,
          class TiledCopyK, class TileShapeK, class GmemLayoutK, class SmemLayoutK, 
          class TileShapeS, class GmemLayoutS, class SmemLayoutS, class SmemLayoutPS, 
          class TiledCopyV, class TileShapeV, class GmemLayoutV, class SmemLayoutV, 
          class SmemLayoutVt, class TiledCopyO, class TileShapeO, class GmemLayoutO,
          class SmemLayoutO, class GmemLayoutDummyKV, class TiledCopyDummyK, class TiledCopyDummyV, 
          class GmemLayoutMI, class ClusterShape>
__global__ static void __launch_bounds__(384, 1)
fmhaForwardPipelinedWspl(
    Gemm1Type const *Q, CUTE_GRID_CONSTANT TiledCopyQ const tmaLoadQ,
    TileShapeQ tileShapeQ, GmemLayoutQ gmemLayoutQ, SmemLayoutQ smemLayoutQ,
    Gemm1Type const *K, CUTE_GRID_CONSTANT TiledCopyK const tmaLoadK,
    TileShapeK tileShapeK, GmemLayoutK gmemLayoutK, SmemLayoutK smemLayoutK,
    Gemm1Type *S, TileShapeS tileShapeS, GmemLayoutS gmemLayoutS,
    SmemLayoutS smemLayoutS, SmemLayoutPS smemLayoutPS, int nTilesOfK, Gemm2Type *V,
    CUTE_GRID_CONSTANT TiledCopyV const tmaLoadV, TileShapeV tileShapeV,
    GmemLayoutV gmemLayoutV, SmemLayoutV smemLayoutV, SmemLayoutVt smemLayoutVt,
    OutputType *O, CUTE_GRID_CONSTANT TiledCopyO const tmaStoreO,
    TileShapeO tileShapeO, GmemLayoutO gmemLayoutO, SmemLayoutO smemLayoutO,
    GmemLayoutDummyKV gmemLayoutDummyKV,
    CUTE_GRID_CONSTANT TiledCopyDummyK const tmaLoadDummyK,
    CUTE_GRID_CONSTANT TiledCopyDummyV const tmaLoadDummyV,
    SoftType *mi_ptr, SoftType *sPrimePtr, GmemLayoutMI gmemLayoutMi,
    float scale) {
  extern __shared__ char shared_memory[];
  using MainloopPipeline = typename cutlass::PipelineTmaNoCAsync<stageCount, PatternLen>;
  // Change to this to use with CUTLASS 3.3 Pipeline API
  // using MainloopPipeline =
  //     typename cutlass::PipelineTmaAsync<stageCount, ClusterShape>;
  using PipelineState = typename cutlass::PipelineState<stageCount>;
  using PhysicalPipelineState = typename cutlass::PipelineState<stageCount>;
  using LogicalPipelineState = typename cutlass::PipelineState<PatternLen>;
  using BarrierType = typename MainloopPipeline::ProducerBarrierType;

  using SharedStorage =
      SharedStorage<Gemm1Type, Gemm2Type, OutputType, SmemLayoutQ, SmemLayoutK,
                    SmemLayoutS, SmemLayoutPS, SmemLayoutV, SmemLayoutO, ClusterShape>;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(shared_memory);

  int warp_group_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
  int warp_idx_in_warpgroup =
      __shfl_sync(0xffffffff, (threadIdx.x / 32) % 4, 0);
  int warp_group_thread_idx = threadIdx.x % 128;
  dim3 bid = cute::block_id_in_cluster();

  auto cluster_shape_ = ClusterShape{};

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  dim3 cluster_shape = cluster.dim_blocks();
  uint32_t clusterBlockRank = cluster.block_rank();

  // Unlike the unit test we always set this variable to 1
  // independent of cluster size.
  uint32_t const NumProducers = 1;

  // Get only TMA tensor mQ outside of producer loops.
  Tensor mQ = tmaLoadQ.get_tma_tensor(shape(gmemLayoutQ));

  // Compute TMA transaction bytes
  // constexpr int per_cta_bytes =
  //     size(tileShapeK) * sizeof_bits_v<Gemm1Type> / 8 +
  //     size(tileShapeV) * sizeof_bits_v<Gemm2Type> / 8;
  constexpr int per_cta_bytes_k = size(tileShapeK) * sizeof_bits_v<Gemm1Type> / 8;
  constexpr int per_cta_bytes_v = size(tileShapeV) * sizeof_bits_v<Gemm2Type> / 8;
  uint32_t const TmaTransactionBytesK = per_cta_bytes_k * NumProducers;
  uint32_t const TmaTransactionBytesV = per_cta_bytes_v * NumProducers;

  // Construct SMEM tensors.
  Tensor sQ =
      make_tensor(make_smem_ptr(shared_storage.smem_q.data()), smemLayoutQ);
  Tensor sO =
      make_tensor(make_smem_ptr(shared_storage.smem_o.data()), smemLayoutO);
  Tensor sK =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_k.data()), smemLayoutK);
#ifdef SINSMEM
  Tensor sS =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_s.data()), smemLayoutS);
#else
  // Just a dummy sS (with smem_v). It's required only for shape later.
  Tensor sS =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_k.data()), smemLayoutS);
  Tensor sNocS =
      make_tensor(make_smem_ptr(reinterpret_cast<cutlass::half_t*>(shared_storage.smem_s.data())), smemLayoutPS);
      // make_tensor(make_smem_ptr(reinterpret_cast<cutlass::half_t*>(shared_storage.kv.smem_k.data()) + 0       ), smemLayoutPS);
  Tensor sNocR =
      make_tensor(make_smem_ptr(reinterpret_cast<cutlass::half_t*>(shared_storage.smem_r.data())), smemLayoutPS);
      // make_tensor(make_smem_ptr(reinterpret_cast<cutlass::half_t*>(shared_storage.kv.smem_k.data()) + size(sNocS)), smemLayoutPS);
#endif
  Tensor sV =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_v.data()), smemLayoutV);

  // Tensor for V Transpose; used in GEMM-II.
  Tensor sVt =
      make_tensor(make_smem_ptr(shared_storage.kv.smem_v.data()), smemLayoutVt);

  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.z % H);
  auto blockIdxB = uint64_t(blockIdx.z / H);

  // No pipelining for copying the block of Q.

  // Get the block of Q for this CTA using the block coordinates
  auto blkCoordQ = make_coord(blockIdxX, 0, blockIdxH, blockIdxB);
  Tensor gQ = local_tile(mQ(_,_,clusterBlockRank / cluster_shape.x,_,_), tileShapeQ, blkCoordQ);

  // Partition the copying of source tiles for Q among threads.
  auto cta_tmaQ = tmaLoadQ.get_slice(0);
  Tensor tQgQX = cta_tmaQ.partition_S(gQ);

  // Group the REST_X modes and the TMA_X modes to easily iterate through the
  // tiles
  Tensor tQgQ = group_modes<1, rank(tQgQX)>(tQgQX); // (TMA,REST)
  auto kTiles = size<1>(tQgQ);
  assert(kTiles == 1);
  assert(kTiles == size<2>(gQ));

  // Partition the copying of dest tile for Q among threads.
  Tensor tQsQX = cta_tmaQ.partition_D(sQ);
  Tensor tQsQ = group_modes<1, rank(tQsQX)>(tQsQX);

  // Copy Q tile from GMEM to SMEM.
  uint64_t *tma_load_mbar = shared_storage.tma_load_mbar;
  cfk::barrierInit(tma_load_mbar[0], 1); // This is REQUIRED.
  cfk::copy(tQgQ(_, 0), tQsQ(_, 0), tmaLoadQ, tma_load_mbar[0]);
  cute::wait_barrier(tma_load_mbar[0], 0); // This is REQUIRED.


  // In the WS kernel, we still initialize matmul objects
  // outside of the consumer body. This is done to avoid a
  // synchronization problem with the QINRMEM flag enabled.
  TiledMma0 tiledMma0;
  TiledMma1 tiledMma1;
  TiledMmaCvt0 tiledMmaCvt0;
  auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);
  auto threadMma1 = tiledMma1.get_thread_slice(threadIdx.x);

  // Allocate "fragments/descriptors"
  // for first matmul.
  Tensor tSrQ = threadMma0.partition_fragment_A(sQ);
  Tensor tSrK = threadMma0.partition_fragment_B(sK);
  Tensor tSrS = partition_fragment_C(tiledMma0, tileShapeS);
  clear(tSrS);

#ifdef QINRMEM
  Tensor tSsQ = threadMma0.partition_A(sQ);
  cfk::copy_nosync(tSsQ, tSrQ);
  warpgroup_fence_operand(tSrQ);
#endif
  // Allocate "fragments/descriptors"
  // for second matmul.
  // Note: S becomes P.
  Tensor tOrV = threadMma1.partition_fragment_B(sVt);
  Tensor tOrS = threadMma1.partition_fragment_A(sS(_, _, 0));
  auto tOrPLayout = ReshapeTStoTP()(tSrS, tOrS);
  auto reg2reg = ReorgCFp8toAFp8();

  // FMHA OUTPUT (GEMM-II) accumulator.
  Tensor tOrO = partition_fragment_C(tiledMma1, tileShapeO);
  clear(tOrO);
  // Allocate space for per-thread rowMax and rowSum in rmem.
  Tensor rowMax = make_tensor<SoftType>(Shape<Int<2 * size<1>(tSrS)>>{});
  Tensor rowSum = make_fragment_like(rowMax);
  cute::fill(rowMax, -cutlass::platform::numeric_limits<SoftType>::infinity());
  cute::fill(rowSum, SoftType(0.0));

  // ------------ Pipelining begins -------------------------------

  // mbarrier.init
  typename MainloopPipeline::Params params;
  params.transaction_bytes[eK] = TmaTransactionBytesK;
  params.transaction_bytes[eV] = TmaTransactionBytesV;
  if (warp_group_idx == 0) {
    params.role = MainloopPipeline::ThreadCategory::Producer;
  } else {
    params.role = MainloopPipeline::ThreadCategory::Consumer;
  }
  params.is_leader = warp_group_thread_idx == 0;
  params.num_consumers = NumMmaThreads;

  MainloopPipeline pipeline(shared_storage.storage, params, cluster_shape_);
  // Change to this to use with CUTLASS 3.3 Pipeline API
  // MainloopPipeline pipeline(shared_storage.storage, params);

  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;

  FullBarrier  *send_mbar_ptr = reinterpret_cast<FullBarrier *>(&(shared_storage.noc_send[0]));
  EmptyBarrier *recv_mbar_ptr = reinterpret_cast<EmptyBarrier*>(&(shared_storage.noc_recv[0]));
  if (threadIdx.x == 0) {
    send_mbar_ptr[0].init(1);
    recv_mbar_ptr[0].init(1);
  }

  auto& shared_schedules = shared_storage.schedules;
  init_schedule(shared_schedules);

  uint32_t producer_phase = 1;
  uint32_t consumer_phase = 0;
  int sender_ready_phase = 0;
  int sender_dsmem_copy_finish_phase = 1;
  int receiver_dsmem_copy_finish_phase = 0;
  // int mma_wait_phase = 0;
  PhysicalPipelineState producer_physical_state = cutlass::make_producer_start_state<MainloopPipeline>();
  PhysicalPipelineState producer_noc_send_state = cutlass::make_producer_start_state<MainloopPipeline>();
  PhysicalPipelineState consumer_physical_state = PhysicalPipelineState(0,0,0);
  LogicalPipelineState  producer_logical_state  = LogicalPipelineState(0,1,0);
  LogicalPipelineState  consumer_logical_state  = LogicalPipelineState(0,0,0);
  cutlass::SeparatePipelineState receiver_ready_state_KV = cutlass::SeparatePipelineState<PatternLen>(0,1,0);

  // int blockIdxY = 0;
  int kIter = 0;

  __syncthreads();

  // Ensure All CTAs in Cluster have completed init before issuing commits
  cute::cluster_arrive_relaxed();
  cute::cluster_wait();

  // Producer warpgroup
  if (warp_group_idx == 0) {
    // method in cutlass/arch/reg_reconfig.h
    // calls setmaxnreg.dec.sync.aligned.u32
    cutlass::arch::warpgroup_reg_dealloc<80>();

    // int lane_predicate = cute::elect_one_sync();
    // if (warp_idx_in_warpgroup == 0 && lane_predicate) {
    if (threadIdx.x == 0) {

      int tma_k_prologue = min(stageCount, nTilesOfK);

      // For the DMA (prologue) - we start with an opposite phase - since we
      // skip all waits i.e., we know that the buffer is indeed empty
      // PipelineState smem_pipe_write =
      //     make_producer_start_state<MainloopPipeline>();
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tma_k_prologue; ++i) {
        block_iter_id src_id_KV = shared_schedules.srcKV[bid.x][kIter % PatternLen];
        BarrierType *tmaBarK = pipeline.producer_get_barrier(producer_logical_state, eK);
        BarrierType *tmaBarV = pipeline.producer_get_barrier(producer_logical_state, eV);
        int kTileIter = (kIter / PatternLen) * PatternLen + shared_schedules.tileOrder[bid.x][kIter % PatternLen];
        pipeline.wait_empty(producer_physical_state, eK);
        if (src_id_KV.x == -1) {
          pipeline.copy_prepare(producer_logical_state, eK, Align128Bytes(TmaTransactionBytesK * (1 + SIMULATE_MULTIPLE)));
          if (shared_schedules.dstKV[bid.x][((kIter - stageCount) % PatternLen + PatternLen) % PatternLen].x != -1) {
            pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eK, kIter % PatternLen);
          }
          fmhaForwardProducerK(sK(_, _, i), tmaLoadK, tileShapeK, gmemLayoutK,
                              sV(_, _, i), tmaLoadV, tileShapeV, gmemLayoutV,
                              gmemLayoutDummyKV, tmaLoadDummyK, tmaLoadDummyV,
                              kTileIter, tmaBarK, tmaBarV, ClusterShape());
        }
        pipeline.wait_empty(producer_physical_state, eV);
        if (src_id_KV.x == -1) {
          pipeline.copy_prepare(producer_logical_state, eV, Align128Bytes(TmaTransactionBytesV * (1 + SIMULATE_MULTIPLE)));
          if (shared_schedules.dstKV[bid.x][((kIter - stageCount) % PatternLen + PatternLen) % PatternLen].x != -1) {
            pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eV, kIter % PatternLen);
          }
          fmhaForwardProducerV(sK(_, _, i), tmaLoadK, tileShapeK, gmemLayoutK,
                              sV(_, _, i), tmaLoadV, tileShapeV, gmemLayoutV,
                              gmemLayoutDummyKV, tmaLoadDummyK, tmaLoadDummyV,
                              kTileIter, tmaBarK, tmaBarV, ClusterShape());
        }
        ++producer_physical_state;
        ++producer_logical_state;
        ++kIter;
        if (((kIter - stageCount) % PatternLen + PatternLen) % PatternLen == 0) {
          sender_dsmem_copy_finish_phase ^= 1;
        }
      }
      // int tma_k_iter = nTilesOfK - tma_k_prologue;
      int pattern_iters = (nTilesOfK / PatternLen) * PatternLen - tma_k_prologue;

      CUTE_NO_UNROLL
      for (int i = 0; i < pattern_iters; ++i) {
        block_iter_id src_id_KV = shared_schedules.srcKV[bid.x][kIter % PatternLen];
        BarrierType *tmaBarK = pipeline.producer_get_barrier(producer_logical_state, eK);
        BarrierType *tmaBarV = pipeline.producer_get_barrier(producer_logical_state, eV);
        auto stage = producer_physical_state.index();
        int kTileIter = (kIter / PatternLen) * PatternLen + shared_schedules.tileOrder[bid.x][kIter % PatternLen];
        pipeline.wait_empty(producer_physical_state, eK);
        if (src_id_KV.x == -1) {
          pipeline.copy_prepare(producer_logical_state, eK, Align128Bytes(TmaTransactionBytesK * (1 + SIMULATE_MULTIPLE)));
          if (shared_schedules.dstKV[bid.x][((kIter - stageCount) % PatternLen + PatternLen) % PatternLen].x != -1) {
            pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eK, kIter % PatternLen);
          }
          fmhaForwardProducerK(sK(_, _, stage), tmaLoadK, tileShapeK, gmemLayoutK,
                              sV(_, _, stage), tmaLoadV, tileShapeV, gmemLayoutV,
                              gmemLayoutDummyKV, tmaLoadDummyK, tmaLoadDummyV,
                              kTileIter, tmaBarK, tmaBarV, ClusterShape());
        }
        pipeline.wait_empty(producer_physical_state, eV);
        if (src_id_KV.x == -1) {
          pipeline.copy_prepare(producer_logical_state, eV, Align128Bytes(TmaTransactionBytesV * (1 + SIMULATE_MULTIPLE)));
          if (shared_schedules.dstKV[bid.x][((kIter - stageCount) % PatternLen + PatternLen) % PatternLen].x != -1) {
            pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eV, kIter % PatternLen);
          }
          fmhaForwardProducerV(sK(_, _, stage), tmaLoadK, tileShapeK, gmemLayoutK,
                              sV(_, _, stage), tmaLoadV, tileShapeV, gmemLayoutV,
                              gmemLayoutDummyKV, tmaLoadDummyK, tmaLoadDummyV,
                              kTileIter, tmaBarK, tmaBarV, ClusterShape());
        }
        ++producer_physical_state;
        ++producer_logical_state;
        ++kIter;
        if (((kIter - stageCount) % PatternLen + PatternLen) % PatternLen == 0) {
          sender_dsmem_copy_finish_phase ^= 1;
        }
      }

      int left_iters = nTilesOfK - tma_k_prologue - pattern_iters;
      CUTE_NO_UNROLL
      for (int i = 0; i < left_iters; ++i) {
        pipeline.wait_empty(producer_physical_state, eK);
        pipeline.wait_empty(producer_physical_state, eV);
        pipeline.copy_prepare(producer_logical_state, eK, Align128Bytes(TmaTransactionBytesK * (1 + SIMULATE_MULTIPLE)));
        pipeline.copy_prepare(producer_logical_state, eV, Align128Bytes(TmaTransactionBytesV * (1 + SIMULATE_MULTIPLE)));
        BarrierType *tmaBarK = pipeline.producer_get_barrier(producer_logical_state, eK);
        BarrierType *tmaBarV = pipeline.producer_get_barrier(producer_logical_state, eV);
        auto stage = producer_physical_state.index();
        int kTileIter = kIter;
        if (i < stageCount && 
            shared_schedules.dstKV[bid.x][((kIter - stageCount) % PatternLen + PatternLen) % PatternLen].x != -1) {
          pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eK, kIter % PatternLen);
          pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eV, kIter % PatternLen);
        }
        fmhaForwardProducerK(sK(_, _, stage), tmaLoadK, tileShapeK, gmemLayoutK,
                            sV(_, _, stage), tmaLoadV, tileShapeV, gmemLayoutV,
                            gmemLayoutDummyKV, tmaLoadDummyK, tmaLoadDummyV,
                            kTileIter, tmaBarK, tmaBarV, ClusterShape());
        fmhaForwardProducerV(sK(_, _, stage), tmaLoadK, tileShapeK, gmemLayoutK,
                            sV(_, _, stage), tmaLoadV, tileShapeV, gmemLayoutV,
                            gmemLayoutDummyKV, tmaLoadDummyK, tmaLoadDummyV,
                            kTileIter, tmaBarK, tmaBarV, ClusterShape());
        ++producer_physical_state;
        ++producer_logical_state;
        ++kIter;
        if (((kIter - stageCount) % PatternLen + PatternLen) % PatternLen == 0) {
          sender_dsmem_copy_finish_phase ^= 1;
        }
      }

      // Tail Loop
      // Handles the case where we never enter the mainloop
      PipelineState tail =
          tma_k_prologue == stageCount ? producer_physical_state : PipelineState{};
      for (int i = 0; i < tma_k_prologue; ++i) {
        pipeline.wait_empty(tail, eK);
        pipeline.wait_empty(tail, eV);
        ++tail;
      }
    }
    // Issue NoC copy
    else if (threadIdx.x == 32) {
      uint32_t src_int_addr;
      uint32_t smem_int_mbar;
      uint32_t remote_addr;
      
      auto blockTmaK = tmaLoadK.get_slice(0);
      auto blockTmaV = tmaLoadV.get_slice(0);

      int sep_stage = 0;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int i = 0; i < PatternLen; i++) {
        sep_stage = i;
        if (shared_schedules.dstKV[bid.x][i].iter >= stageCount) { break; }
      }
      receiver_ready_state_KV.set_sep_stage(sep_stage);

      // int tma_k_iter = nTilesOfK;
      int pattern_iters = (nTilesOfK / PatternLen) * PatternLen;
      CUTE_NO_UNROLL
      for (int i = 0; i < pattern_iters; ++i) {
        auto stage = producer_noc_send_state.index();
        Tensor tKsK = blockTmaK.partition_D(sK);
        Tensor tVsV = blockTmaV.partition_D(sV);

        block_iter_id dst_id_KV = shared_schedules.dstKV[bid.x][kIter % PatternLen];
        int dst_stage = (dst_id_KV.iter - (kIter % PatternLen) + stage) % stageCount;
        if (dst_id_KV.x != -1) {
          uint32_t block_id = dst_id_KV.x + bid.y * cluster_shape.x;
          BarrierType* tmaBarK = pipeline.producer_get_barrier_by_stage(dst_id_KV.iter % PatternLen, eK);
          BarrierType* tmaBarV = pipeline.producer_get_barrier_by_stage(dst_id_KV.iter % PatternLen, eV);

          pipeline.sender_wait_sender_ready(sender_ready_phase, eK, kIter % PatternLen);
          pipeline.sender_wait_receiver_ready(receiver_ready_state_KV, eK);
          if (shared_schedules.dstKV[dst_id_KV.x][((dst_id_KV.iter - stageCount) + PatternLen) % PatternLen].x != -1) {
            pipeline.sync_wait(sender_ready_phase, eK, kIter % PatternLen);
          }
          // copy to the block with the same bid.y
          pipeline.dsmem_copy_prepare(Align128Bytes(TmaTransactionBytesK / NOC_ACCL_MULTIPLE), block_id, eK, dst_id_KV.iter % PatternLen);
          
          src_int_addr = cast_smem_ptr_to_uint(tKsK(_,_,_,stage).data().get().get());
          smem_int_mbar = set_block_rank(cast_smem_ptr_to_uint(tmaBarK), block_id);
          remote_addr = set_block_rank(cast_smem_ptr_to_uint(tKsK(_,_,_,dst_stage).data().get().get()), block_id);
          noc::dsmem_copy_func(src_int_addr, remote_addr, smem_int_mbar, Align128Bytes(TmaTransactionBytesK / NOC_ACCL_MULTIPLE));
        
          pipeline.sender_wait_sender_ready(sender_ready_phase, eV, kIter % PatternLen);
          pipeline.sender_wait_receiver_ready(receiver_ready_state_KV, eV);
          if (shared_schedules.dstKV[dst_id_KV.x][((dst_id_KV.iter - stageCount) + PatternLen) % PatternLen].x != -1) {
            pipeline.sync_wait(sender_ready_phase, eV, kIter % PatternLen);
          }
          pipeline.dsmem_copy_prepare(Align128Bytes(TmaTransactionBytesV / NOC_ACCL_MULTIPLE), block_id, eV, dst_id_KV.iter % PatternLen);
          
          src_int_addr = cast_smem_ptr_to_uint(tVsV(_,_,_,stage).data().get().get());
          smem_int_mbar = set_block_rank(cast_smem_ptr_to_uint(tmaBarV), block_id);
          remote_addr = set_block_rank(cast_smem_ptr_to_uint(tVsV(_,_,_,dst_stage).data().get().get()), block_id);
          noc::dsmem_copy_func(src_int_addr, remote_addr, smem_int_mbar, Align128Bytes(TmaTransactionBytesV / NOC_ACCL_MULTIPLE));
        } 
        ++kIter;
        ++receiver_ready_state_KV;
        ++producer_noc_send_state;
        if (kIter % PatternLen == 0) {
          sender_ready_phase ^= 1;
        }
      }

    }
    // Monitor if the NoC copy is done, if done, notify the src block.
    else if (threadIdx.x == 64) {
      // int tma_k_iter = nTilesOfK;
      int pattern_iters = (nTilesOfK / PatternLen) * PatternLen;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int i = 0; i < pattern_iters; ++i) {
        block_iter_id src_id_KV = shared_schedules.srcKV[bid.x][kIter % PatternLen];
        // Copy on this iteration is from dsmem
        if (src_id_KV.x != -1) {
          uint32_t block_id = src_id_KV.x + bid.y * cluster_shape.x;
          pipeline.receiver_wait_dsmem_copy_finish(receiver_dsmem_copy_finish_phase, eK, kIter % PatternLen);
          pipeline.receiver_arrive_dsmem_copy_finish(block_id, eK, (src_id_KV.iter + stageCount) % PatternLen);
          pipeline.receiver_wait_dsmem_copy_finish(receiver_dsmem_copy_finish_phase, eV, kIter % PatternLen);
          pipeline.receiver_arrive_dsmem_copy_finish(block_id, eV, (src_id_KV.iter + stageCount) % PatternLen);
        }
        ++kIter;
        if (kIter % PatternLen == 0) {
          receiver_dsmem_copy_finish_phase ^= 1;
        }
      }
    }
    // If the previous iteration on this buffer issues a NoC copy, check if the NoC copy is done,
    // if done, notify the src block of this iteration
    else if (threadIdx.x == 96) {
      // int tma_k_iter = nTilesOfK;
      int pattern_iters = (nTilesOfK / PatternLen) * PatternLen;
      CUTLASS_PRAGMA_NO_UNROLL
      for (int i = 0; i < pattern_iters; ++i) {
        if (shared_schedules.dstKV[bid.x][((kIter - stageCount) % PatternLen + PatternLen) % PatternLen].x != -1) {
          block_iter_id src_id_KV = shared_schedules.srcKV[bid.x][kIter % PatternLen];
          pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eK, kIter % PatternLen);
          if (src_id_KV.x != -1) {
            uint32_t block_id = src_id_KV.x + bid.y * cluster_shape.x;
            pipeline.sync_arrive(block_id, eK, src_id_KV.iter);
          }
          pipeline.sender_wait_dsmem_copy_finish(sender_dsmem_copy_finish_phase, eV, kIter % PatternLen);
          if (src_id_KV.x != -1) {
            uint32_t block_id = src_id_KV.x + bid.y * cluster_shape.x;
            pipeline.sync_arrive(block_id, eV, src_id_KV.iter);
          }
        }
        ++kIter;
        if (((kIter - stageCount) % PatternLen + PatternLen) % PatternLen == 0) {
          sender_dsmem_copy_finish_phase ^= 1;
        }
      }
    }
  }
  // Consumer warpgroup(s)
  else if (warp_group_idx == 1 || warp_group_idx == 2) {
    // method in cutlass/arch/reg_reconfig.h
    // calls setmaxnreg.inc.sync.aligned.u32
    cutlass::arch::warpgroup_reg_alloc<192>();

    // PipelineState smem_pipe_read;
    // PipelineState smem_pipe_release;
    PhysicalPipelineState consumer_release_state = consumer_physical_state;

    // Init Shared Memory read stages & PhaseBit
    static constexpr uint32_t K_PIPE_MMAS = 1;
    static_assert(K_PIPE_MMAS < stageCount, "ERROR : Too many MMAs in flight");

    // Total number of gemm iterations
    auto gemm_k_iterations = nTilesOfK;

    int mma_k_prologue = min(K_PIPE_MMAS, gemm_k_iterations);
    int pattern_iters = (nTilesOfK / PatternLen) * PatternLen;

    CUTLASS_PRAGMA_UNROLL
    for (int iter = 0; iter < mma_k_prologue; ++iter) {
      pipeline.consumer_wait(consumer_logical_state, eK);
      pipeline.consumer_wait(consumer_logical_state, eV);
      warpgroup_arrive();

      int stage = consumer_physical_state.index();
      // NOTICE: This requires bM*bN*4 <= bN * headdim / SplitNum
      // Tensor sNocS =
      //     make_tensor(make_smem_ptr(reinterpret_cast<cutlass::half_t*>(shared_storage.kv.smem_k.data()) 
      //                               + size(smemLayoutK(_,_,0)) * stage + 0          ), smemLayoutPS);
      // Tensor sNocR =
      //     make_tensor(make_smem_ptr(reinterpret_cast<cutlass::half_t*>(shared_storage.kv.smem_k.data())
      //                               + size(smemLayoutK(_,_,0)) * stage + size(sNocS)), smemLayoutPS);
      fmhaForwardConsumerK(Q, K, V, S, tSrQ, tSrK(_, _, _, stage), tSrS,
                          tOrV(_, _, _, stage), tOrO, tOrPLayout, reg2reg, rowMax,
                          rowSum, tileShapeS, gmemLayoutS, scale, kIter++,
                          tiledMma0, tiledMma1, tiledMmaCvt0,
                          send_mbar_ptr, recv_mbar_ptr,
                          sNocS, sNocR, producer_phase, consumer_phase, AccumType(0), SoftType(0));

      fmhaForwardConsumerV(Q, K, V, S, tSrQ, tSrK(_, _, _, stage), tSrS,
                          tOrV(_, _, _, stage), tOrO, tOrPLayout, reg2reg, rowMax,
                          rowSum, tileShapeS, gmemLayoutS, scale, kIter++,
                          tiledMma0, tiledMma1, tiledMmaCvt0,
                          send_mbar_ptr, recv_mbar_ptr, 
                          sNocS, sNocR, producer_phase, consumer_phase, AccumType(0), SoftType(0));
      ++consumer_logical_state;
      ++consumer_physical_state;
    }
    gemm_k_iterations -= mma_k_prologue;

    int kReleaseIter = 0;
    CUTLASS_PRAGMA_NO_UNROLL
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      /// Wait on the smem_pipe_read stage / phase
      pipeline.consumer_wait(consumer_logical_state, eK);
      warpgroup_arrive();

      int stage = consumer_physical_state.index();
      // NOTICE: This requires bM*bN*4 <= bN * headdim / SplitNum
      // Tensor sNocS =
      //     make_tensor(make_smem_ptr(reinterpret_cast<cutlass::half_t*>(shared_storage.kv.smem_k.data()) 
      //                               + size(smemLayoutK(_,_,0)) * stage + 0          ), smemLayoutPS);
      // Tensor sNocR =
      //     make_tensor(make_smem_ptr(reinterpret_cast<cutlass::half_t*>(shared_storage.kv.smem_k.data())
      //                               + size(smemLayoutK(_,_,0)) * stage + size(sNocS)), smemLayoutPS);
      fmhaForwardConsumerK(Q, K, V, S, tSrQ, tSrK(_, _, _, stage), tSrS,
                          tOrV(_, _, _, stage), tOrO, tOrPLayout, reg2reg, rowMax,
                          rowSum, tileShapeS, gmemLayoutS, scale, kIter++,
                          tiledMma0, tiledMma1, tiledMmaCvt0,
                          send_mbar_ptr, recv_mbar_ptr,
                          sNocS, sNocR, producer_phase, consumer_phase, AccumType(0), SoftType(0));

      warpgroup_wait<2 * K_PIPE_MMAS>();
      if (threadIdx.x % 128 == 0) {
        block_iter_id src_id_KV = shared_schedules.srcKV[bid.x][(kReleaseIter + stageCount) % PatternLen];
        if (src_id_KV.x != -1 && kReleaseIter + stageCount < pattern_iters) {
          uint32_t block_id = src_id_KV.x + bid.y * cluster_shape.x;
          pipeline.receiver_arrive_sender(block_id, eK, src_id_KV.iter);
        }
      }
      pipeline.consumer_release_self(consumer_release_state, eK);
      
      pipeline.consumer_wait(consumer_logical_state, eV);
      fmhaForwardConsumerV(Q, K, V, S, tSrQ, tSrK(_, _, _, stage), tSrS,
                          tOrV(_, _, _, stage), tOrO, tOrPLayout, reg2reg, rowMax,
                          rowSum, tileShapeS, gmemLayoutS, scale, kIter++,
                          tiledMma0, tiledMma1, tiledMmaCvt0,
                          send_mbar_ptr, recv_mbar_ptr,
                          sNocS, sNocR, producer_phase, consumer_phase, AccumType(0), SoftType(0));
      // warpgroup_wait<2 * K_PIPE_MMAS>();
      //    warpgroup_fence_operand(tSrS);
      //    warpgroup_fence_operand(tOrO);

      if (threadIdx.x % 128 == 0) {
        block_iter_id src_id_KV = shared_schedules.srcKV[bid.x][(kReleaseIter + stageCount) % PatternLen];
        if (src_id_KV.x != -1 && kReleaseIter + stageCount < pattern_iters) {
          uint32_t block_id = src_id_KV.x + bid.y * cluster_shape.x;
          pipeline.receiver_arrive_sender(block_id, eV, src_id_KV.iter);
        }
      }
      pipeline.consumer_release_self(consumer_release_state, eV);

      // Advance stages
      ++consumer_logical_state;
      ++consumer_physical_state;
      ++consumer_release_state;
      ++kReleaseIter;
    }

    // warpgroup_wait<0>();
    // Tail Loop
    for (int i = 0; i < K_PIPE_MMAS; ++i) {
      pipeline.consumer_release_self(consumer_release_state, eK);
      pipeline.consumer_release_self(consumer_release_state, eV);
      ++consumer_release_state;
    }

    // TMA Store epilogue
    bool leaderWarp = warp_group_idx == 1 && warp_idx_in_warpgroup == 0;
    fmhaForwardWriteOutTMA(tOrO, rowMax, rowSum, O, tileShapeO, gmemLayoutO,
                           tiledMma1, sO, tmaStoreO, leaderWarp, SoftType(0.0));

// Write out rowMax and rowSum to GMEM.
// Required for verification ONLY.
#ifdef COPYOUTMI
    fmhaForwardWriteOutSoftMax(rowMax, rowSum, mi_ptr, sPrimePtr, gmemLayoutMi,
                               tiledMma0, tileShapeO);
#endif
  }
}
