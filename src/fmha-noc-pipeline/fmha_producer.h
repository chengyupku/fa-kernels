#pragma once

#include "noc_config.h"

// FMHA Producer does K and V copy
template <class TensorEngineK, class SmemLayoutK, class TiledCopyK,
          class TileShapeK, class GmemLayoutK, class TensorEngineV,
          class SmemLayoutV, class TiledCopyV, class TileShapeV,
          class TiledCopyDummyK, class TiledCopyDummyV,
          class GmemLayoutV, class GmemLayoutDummyKV, class BarrierType, class ClusterShape>
__device__ static void //__launch_bounds__(128, 2)
fmhaForwardProducerK(Tensor<TensorEngineK, SmemLayoutK> &&sK,
                    TiledCopyK const &tmaLoadK, TileShapeK tileShapeK,
                    GmemLayoutK gmemLayoutK,
                    Tensor<TensorEngineV, SmemLayoutV> &&sV,
                    TiledCopyV const &tmaLoadV, TileShapeV tileShapeV,
                    GmemLayoutV gmemLayoutV, 
                    GmemLayoutDummyKV gmemLayoutDummyKV,
                    TiledCopyDummyK const &tmaLoadDummyK, TiledCopyDummyV const &tmaLoadDummyV,
                    int blockIdxY, 
                    BarrierType *tmaBarK, BarrierType *tmaBarV, 
                    const ClusterShape &) {

  using namespace cute;

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  dim3 cluster_shape = cluster.dim_blocks();
  uint32_t clusterBlockRank = cluster.block_rank();

  auto blockIdxH = uint64_t(blockIdx.z % H);
  auto blockIdxB = uint64_t(blockIdx.z / H);

  // Get the full un-partitioned tensors.
  // TMA tensors are special tensors.
  Tensor mK = tmaLoadK.get_tma_tensor(shape(gmemLayoutK));

  Tensor mDummyK = tmaLoadDummyK.get_tma_tensor(shape(gmemLayoutDummyKV));

  // NOTICE: block_rank_in_cluster is hard coded to 0
  // uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
  uint32_t block_rank_in_cluster = 0;
  constexpr uint32_t cluster_shape_x = get<0>(ClusterShape{});
  uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                  block_rank_in_cluster / cluster_shape_x};
  auto blockTmaK = tmaLoadK.get_slice(cluster_local_block_id.x);

  auto blockTmaDummyK = tmaLoadDummyK.get_slice(cluster_local_block_id.x);

  //
  // Partition the copying of dest tiles for K and V among threads.
  //
  Tensor tKsKX = blockTmaK.partition_D(sK);
  Tensor tKsK = group_modes<1, rank(tKsKX)>(tKsKX);
  static_assert(size<1>(tKsK) == 1);

  //
  // Get the GMEM tensors for K and V
  //
  auto blkCoordK = make_coord(blockIdxY, 0, blockIdxH, blockIdxB);
  Tensor gK = local_tile(mK(_,_,clusterBlockRank / cluster_shape.x,_,_), tileShapeK, blkCoordK);

  Tensor tKgKX = blockTmaK.partition_S(gK);
  Tensor tKgK = group_modes<1, rank(tKgKX)>(tKgKX); // (TMA,REST)
  assert(size<1>(tKgK) == size<2>(gK));
  assert(size<1>(tKgK) == kTiles);
  static_assert(size<1>(tKsK) == 1);

#ifdef GEMM2FP8
  auto blkCoordV = make_coord(0, blockIdxY, blockIdxH, blockIdxB);
#else
  auto blkCoordV = make_coord(blockIdxY, 0, blockIdxH, blockIdxB);
#endif

  uint16_t mcast_mask_a = 0;
  auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
  for (int n = 0; n < size(block_layout); ++n) {
    mcast_mask_a |= (uint16_t(1) << block_layout(n, 0, Int<0>{}));
  }

  // Copy current tiles of V and K from GMEM to SMEM.
  // Uses TMA multicast for CLUSTERN>1
  copy(tmaLoadK.with(*tmaBarK, mcast_mask_a), tKgK(_, 0), tKsK(_, 0));
#if SIMULATE_MULTIPLE > 0
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < SIMULATE_MULTIPLE; i++) {
    Tensor gDummyK = local_tile(mDummyK(_,_,clusterBlockRank / cluster_shape.x, i,_,_), tileShapeK, blkCoordK);
    Tensor tDummyKgKX = blockTmaDummyK.partition_S(gDummyK);
    Tensor tDummyKgK = group_modes<1, rank(tDummyKgKX)>(tDummyKgKX); // (TMA,REST)
    copy(tmaLoadDummyK.with(*tmaBarK, mcast_mask_a), tDummyKgK(_, 0), tKsK(_, 0));
  }
#endif
}

// FMHA Producer does K and V copy
template <class TensorEngineK, class SmemLayoutK, class TiledCopyK,
          class TileShapeK, class GmemLayoutK, class TensorEngineV,
          class SmemLayoutV, class TiledCopyV, class TileShapeV,
          class TiledCopyDummyK, class TiledCopyDummyV,
          class GmemLayoutV, class GmemLayoutDummyKV, class BarrierType, class ClusterShape>
__device__ static void //__launch_bounds__(128, 2)
fmhaForwardProducerV(Tensor<TensorEngineK, SmemLayoutK> &&sK,
                    TiledCopyK const &tmaLoadK, TileShapeK tileShapeK,
                    GmemLayoutK gmemLayoutK,
                    Tensor<TensorEngineV, SmemLayoutV> &&sV,
                    TiledCopyV const &tmaLoadV, TileShapeV tileShapeV,
                    GmemLayoutV gmemLayoutV, 
                    GmemLayoutDummyKV gmemLayoutDummyKV,
                    TiledCopyDummyK const &tmaLoadDummyK, TiledCopyDummyV const &tmaLoadDummyV,
                    int blockIdxY, 
                    BarrierType *tmaBarK, BarrierType *tmaBarV, 
                    const ClusterShape &) {

  using namespace cute;

  namespace cg = cooperative_groups;
  cg::cluster_group cluster = cg::this_cluster();
  dim3 cluster_shape = cluster.dim_blocks();
  uint32_t clusterBlockRank = cluster.block_rank();

  auto blockIdxH = uint64_t(blockIdx.z % H);
  auto blockIdxB = uint64_t(blockIdx.z / H);

  // Get the full un-partitioned tensors.
  // TMA tensors are special tensors.
  Tensor mV = tmaLoadV.get_tma_tensor(shape(gmemLayoutV));

  Tensor mDummyV = tmaLoadDummyV.get_tma_tensor(shape(gmemLayoutDummyKV));

  // NOTICE: block_rank_in_cluster is hard coded to 0
  // uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();
  uint32_t block_rank_in_cluster = 0;
  constexpr uint32_t cluster_shape_x = get<0>(ClusterShape{});
  uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                  block_rank_in_cluster / cluster_shape_x};
  auto blockTmaV = tmaLoadV.get_slice(cluster_local_block_id.x);

  auto blockTmaDummyV = tmaLoadDummyV.get_slice(cluster_local_block_id.x);

  //
  // Partition the copying of dest tiles for K and V among threads.
  //
  Tensor tVsVX = blockTmaV.partition_D(sV);
  Tensor tVsV = group_modes<1, rank(tVsVX)>(tVsVX);
  static_assert(size<1>(tVsV) == 1);

  //
  // Get the GMEM tensors for K and V
  //
#ifdef GEMM2FP8
  auto blkCoordV = make_coord(0, blockIdxY, blockIdxH, blockIdxB);
#else
  auto blkCoordV = make_coord(blockIdxY, 0, blockIdxH, blockIdxB);
#endif

  Tensor gV = local_tile(mV(_,_,clusterBlockRank / cluster_shape.x,_,_), tileShapeV, blkCoordV);

  Tensor tVgVX = blockTmaV.partition_S(gV);
  Tensor tVgV = group_modes<1, rank(tVgVX)>(tVgVX); // (TMA,REST)
  assert(size<1>(tVgV) == size<2>(gV));
  assert(size<1>(tVgV) == 1);

  uint16_t mcast_mask_a = 0;
  auto block_layout = Layout<ClusterShape>{}; // (m,n) -> block_id
  for (int n = 0; n < size(block_layout); ++n) {
    mcast_mask_a |= (uint16_t(1) << block_layout(n, 0, Int<0>{}));
  }

  // Copy current tiles of V and K from GMEM to SMEM.
  // Uses TMA multicast for CLUSTERN>1
  copy(tmaLoadV.with(*tmaBarV, mcast_mask_a), tVgV(_, 0), tVsV(_, 0));
#if SIMULATE_MULTIPLE > 0
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < SIMULATE_MULTIPLE; i++) {
    Tensor gDummyV = local_tile(mDummyV(_,_,clusterBlockRank / cluster_shape.x, i,_,_), tileShapeV, blkCoordV);
    Tensor tDummyVgVX = blockTmaDummyV.partition_S(gDummyV);
    Tensor tDummyVgV = group_modes<1, rank(tDummyVgVX)>(tDummyVgVX); // (TMA,REST)
    copy(tmaLoadDummyV.with(*tmaBarV, mcast_mask_a), tDummyVgV(_, 0), tVsV(_, 0));
  }
#endif
}
