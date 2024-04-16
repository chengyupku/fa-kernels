#pragma once

// Default kQueriesPerBlock is 64.
#ifdef QBLKSIZE
constexpr int kQueriesPerBlock = QBLKSIZE;
#else
constexpr int kQueriesPerBlock = 64;
#endif

// Default kKeysPerBlock is 64.
#ifdef KBLKSIZE
constexpr int kKeysPerBlock = KBLKSIZE;
#else
constexpr int kKeysPerBlock = 64;
#endif

// Default number of stages is 3.
#ifdef STAGECOUNT
constexpr int stageCount = STAGECOUNT;
#else
constexpr int stageCount = 3;
#endif

constexpr int NumCopyThreads = 128;

#ifdef CTA256
constexpr int NumMmaWarpGroups = 2;
constexpr int NumMmaThreads = 256;
#else
constexpr int NumMmaWarpGroups = 1;
constexpr int NumMmaThreads = 128;
#endif

// #include "cutlass/pipeline/pipeline.hpp"
#include "async_pipeline.hpp"
#include "utils.h"

// Shared Storage with Aligned addresses.
template <class Gemm1Type, class Gemm2Type, class SmemLayoutK,
          class SmemLayoutS, class SmemLayoutV>
struct SharedStorageKV {
  cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutK>> smem_k;
  cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutV>> smem_v;
#ifdef SINSMEM
  cute::array_aligned<Gemm2Type, cute::cosize_v<SmemLayoutS>> smem_s;
#endif
};

// Shared Storage with Aligned addresses.
#if defined(QINRMEM) && EXECMODE != 2
template <class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutS, class SmemLayoutV,
          class SmemLayoutO,
          typename ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>>
struct SharedStorage {
  union {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    SharedStorageKV<Gemm1Type, Gemm2Type, SmemLayoutK, SmemLayoutS, SmemLayoutV>
        kv;
  };
  struct {
    cute::uint64_t tma_load_mbar[8]; // 8 TMA barrier pre-allcoated for usage.
    typename cutlass::PipelineTmaAsync<stageCount>::SharedStorage storage;
  };
};
#else
template <class Gemm1Type, class Gemm2Type, class OutputType, class SmemLayoutQ,
          class SmemLayoutK, class SmemLayoutS, class SmemLayoutPS, class SmemLayoutV,
          class SmemLayoutO,
          typename ClusterShape = cutlass::gemm::GemmShape<1, 1, 1>>
struct SharedStorage {
  struct {
    cute::array_aligned<Gemm1Type, cute::cosize_v<SmemLayoutQ>> smem_q;
    union {
      SharedStorageKV<Gemm1Type, Gemm2Type, SmemLayoutK, SmemLayoutS,
                      SmemLayoutV>
          kv;
      cute::array_aligned<OutputType, cute::cosize_v<SmemLayoutO>> smem_o;
    };
    cute::array_aligned<cutlass::half_t, cute::cosize_v<SmemLayoutPS>> smem_s;
    cute::array_aligned<cutlass::half_t, cute::cosize_v<SmemLayoutPS>> smem_r;
  };
  struct {
    cute::uint64_t tma_load_mbar[8]; // 8 TMA barriers pre-allocated for usage.
    cute::uint64_t noc_send[1];
    cute::uint64_t noc_recv[1];
    typename cutlass::PipelineTmaNoCAsync<stageCount, PatternLen>::SharedStorage storage;
  };
  struct ScheduleStorage : cute::aligned_struct<128> {
    int8_t tileOrder[IntraSplitNum][PatternLen];
    block_iter_id srcKV[IntraSplitNum][PatternLen];
    block_iter_id dstKV[IntraSplitNum][PatternLen];
    // block_iter_id srcK[IntraSplitNum][PatternLen];
    // block_iter_id dstK[IntraSplitNum][PatternLen];
    // block_iter_id srcV[IntraSplitNum][PatternLen];
    // block_iter_id dstV[IntraSplitNum][PatternLen];
  } schedules;
};
#endif
