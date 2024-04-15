#pragma once
#include "cutlass/pipeline/pipeline.hpp"

#define eK 0
#define eV 1
namespace cutlass {

using namespace cute;

template <int Stages_>
class PipelineTmaNoCAsync {
public :
  using FullBarrier = cutlass::arch::ClusterTransactionBarrier;
  using EmptyBarrier = cutlass::arch::ClusterBarrier;
  using ProducerBarrierType = FullBarrier::ValueType;
  using ConsumerBarrierType = EmptyBarrier::ValueType;
  static constexpr uint32_t Stages = Stages_;
  using PipelineState = cutlass::PipelineState<Stages>;

  struct SharedStorage {
    FullBarrier full_barrier_[2][Stages];
    EmptyBarrier empty_barrier_[2][Stages];
  };

  enum class ThreadCategory {
    NonParticipant,
    Producer,
    Consumer,
    ProducerConsumer
  };

  struct Params {
    uint32_t transaction_bytes[2] = {0};
    ThreadCategory role = ThreadCategory::NonParticipant;
    uint32_t is_leader = 0;
    uint32_t num_consumers = 0;
  };

  // Constructor
  template<typename ClusterShape>
  CUTLASS_DEVICE
  PipelineTmaNoCAsync(SharedStorage& storage, Params params, ClusterShape cluster_shape)
      : params_(params)
      , full_barrier_ptr_(storage.full_barrier_)
      , empty_barrier_ptr_(storage.empty_barrier_) {

    int warp_idx = canonical_warp_idx();
    int lane_predicate = cute::elect_one_sync();

    if (warp_idx == 0 && lane_predicate == 1) {
      // Barrier FULL init
      for (int i = 0; i < Stages; ++i) {
        full_barrier_ptr_[eK][i].init(1);
        full_barrier_ptr_[eV][i].init(1);
      }
      uint32_t const num_consumer_warpgroups_per_cluster = params_.num_consumers / NumThreadsPerWarpGroup;
      uint32_t const multicast_consumer_arrival_count = (cute::size<0>(cluster_shape) + cute::size<1>(cluster_shape) - 1) *
          num_consumer_warpgroups_per_cluster;
      // Barrier EMPTY init
      for (int i = 0; i < Stages; ++i) {
        empty_barrier_ptr_[eK][i].init(multicast_consumer_arrival_count);
        empty_barrier_ptr_[eV][i].init(multicast_consumer_arrival_count);
      }
    }
    cutlass::arch::fence_barrier_init();

    // Logic to optimally schedule Empty Arrives
    // Goal : To divide SYNCS Empty Arrival duty equally amongst the Warp-Group (128 threads)
    dim3 block_id = cute::block_id_in_cluster();
    auto cluster_size = cute::size(cluster_shape);
    static constexpr int MaxClusterSize = 16;
    static_assert(cluster_size <= MaxClusterSize, "ERROR : Cluster size too large !" );

    // STEP 1 : Use Cute Layout function to generate an optimal dst block-id (0-15)
    if (params_.num_consumers % NumThreadsPerWarpGroup == 0) {
      int thread_idx = threadIdx.x % NumThreadsPerWarpGroup;
      is_signalling_thread_ = (thread_idx % (NumThreadsPerWarpGroup / MaxClusterSize)) == 0;
      auto layout = cute::composition(Swizzle<2,0,-2>{},
                                      Layout<Shape<_4,_4>,Stride<_4,_1>>{});
      uint32_t thread_row = warp_idx % 4;
      uint32_t thread_col = (thread_idx / 8) % 4;
      dst_blockid_ = layout(thread_row, thread_col);
    }
    else if (params_.num_consumers == 32) {
      int thread_idx = threadIdx.x % 32;
      is_signalling_thread_ = (thread_idx % (32 / MaxClusterSize)) == 0;
      auto layout = Layout<Shape<_4,_4>,Stride<_4, _1>>{};
      uint32_t thread_row = thread_idx / 8;
      uint32_t thread_col = (thread_idx % 8) / 2;
      dst_blockid_ = layout(thread_row, thread_col);
    }
    else {
      is_signalling_thread_ = 0;
      #ifndef NDEBUG
        asm volatile ("brkpt;\n" ::);
      #endif
    }

    // STEP 2: Find if this dst block-id needs an arrival for this problem
    is_signalling_thread_ &= dst_blockid_ < cluster_size;
    is_signalling_thread_ &= is_same_row_or_col(dst_blockid_, block_id, cluster_shape);
  }
  
  template <typename ClusterShape>
  CUTLASS_DEVICE
  bool is_same_row_or_col(int dst_block_id, dim3 block_id, ClusterShape cluster_shape) {
    return (((dst_block_id % cute::size<0>(cluster_shape)) == block_id.x) ||
            (
              ((dst_block_id / cute::size<0>(cluster_shape)) == block_id.y)
            ));
  }

  ////////////////////
  // Producer APIs
  ////////////////////
  // Four member functions are always used in pairs:
  //
  // * producer_try_acquire and producer_acquire, and
  // * consumer_try_wait and consumer_wait. 
  //
  // The two functions with "try" in their names are called "try" functions,
  // and the other two are conceptually "finalize" functions.
  // The "try" function in each pair starts the process of waiting on the barrier to flip.
  // It opportunistically waits for an implementation-dependent timeout.
  // Whether or not the barrier has flipped yet, the try function will return a token.
  // If the token indicates that the barrier has not flipped,
  // then the token must be passed into the corresponding "finalize" function.
  // The finalize function will then block until the barrier has flipped.
  // If the token indicates that the barrier _has_ flipped,
  // then it is still correct to pass it into the finalize function.
  // The finalize function will return immediately in that case.

  // CUTLASS_DEVICE
  // ProducerToken producer_try_acquire(PipelineState state, uint32_t skip_wait = false) {
  //   return producer_try_acquire(state.index(), state.phase(), skip_wait);
  // }

  CUTLASS_DEVICE
  void producer_acquire(PipelineState state, uint32_t var, ProducerToken barrier_token = {BarrierStatus::WaitAgain}) {
    producer_acquire(state.index(), state.phase(), barrier_token, var);
  }

  CUTLASS_DEVICE
  void producer_commit(PipelineState state, uint32_t bytes) {
    producer_commit(state.index(), bytes);
  }

  // Prevents early exit of producer blocks in Cluster.
  // This should be called once before kernel exits.
  CUTLASS_DEVICE
  void producer_tail(PipelineState state) {
    for (int count = 0; count < Stages; ++count) {
      producer_acquire(state, eK, {BarrierStatus::WaitOnly});  
      producer_acquire(state, eV, {BarrierStatus::WaitOnly});  
      ++state;
    }
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(PipelineState state, uint32_t var) {
    return producer_get_barrier(state.index(), var);
  }

  ////////////////////
  // Consumer APIs
  ////////////////////
  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(PipelineState state, uint32_t var, uint32_t skip_wait = false) {
    return consumer_try_wait(state.index(), state.phase(), skip_wait, var);
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(PipelineState state, uint32_t var, uint32_t skip_wait = false) {
    return consumer_test_wait(state.index(), state.phase(), skip_wait, var);
  }
  
  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, uint32_t var) {
    consumer_wait(state.index(), state.phase(), var);
  }

  CUTLASS_DEVICE
  void consumer_wait(PipelineState state, ConsumerToken barrier_token, uint32_t var) {
    consumer_wait(state.index(), state.phase(), barrier_token, var);
  }

  CUTLASS_DEVICE
  void consumer_release(PipelineState state, uint32_t var) {
    consumer_release(state.index(), var);
  }

  CUTLASS_DEVICE
  void consumer_release_self(PipelineState state, uint32_t var) {
    consumer_release_self(state.index(), var);
  }

private :
  uint32_t dst_blockid_ = 0;
  uint32_t is_signalling_thread_ = 0;
  FullBarrier (*full_barrier_ptr_)[Stages] = nullptr;
  EmptyBarrier (*empty_barrier_ptr_)[Stages] = nullptr;
  Params params_;

  // CUTLASS_DEVICE
  // ProducerToken producer_try_acquire(uint32_t stage, uint32_t phase, uint32_t skip_wait) {
  //   if (skip_wait) {
  //     return {BarrierStatus::WaitDone};
  //   }
  //   uint32_t barrier_status = empty_barrier_ptr_[stage].try_wait(phase);
  //   return {static_cast<BarrierStatus>(barrier_status)};
  // }

  CUTLASS_DEVICE
  void producer_acquire(uint32_t stage, uint32_t phase, ProducerToken barrier_token, uint32_t var) {
    if (barrier_token != BarrierStatus::WaitDone) {
      empty_barrier_ptr_[var][stage].wait(phase);
    }
    if (barrier_token == BarrierStatus::WaitOnly) {
      return;
    }

    if (params_.is_leader) {
      full_barrier_ptr_[var][stage].arrive_and_expect_tx(params_.transaction_bytes[var]);
    }
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Consumer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }

    // Most likely you have elected more than one leader
    if (params_.is_leader && (threadIdx.x % 32 != 0)) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }

  // NOP for TMA based mainloop
  CUTLASS_DEVICE
  void producer_commit(uint32_t stage, uint32_t bytes) {
    // Below code is used only for unit-testing (in the absence of TMA commit)
    #if CUTLASS_UNIT_TEST_PIPELINE
      if (params_.is_leader) {
        // STEP 1 : Commit to self
        full_barrier_ptr_[stage].complete_transaction(bytes);

        // STEP 2 : Commit to other blocks in our cluster
        auto cluster_shape = cute::cluster_shape();
        Layout block_layout_in_cluster = make_layout(cluster_shape);
        dim3 local_block_id = cute::block_id_in_cluster();

        CUTLASS_PRAGMA_UNROLL
        for(int n = 0; n < size<1>(block_layout_in_cluster); ++n) {
          uint32_t dst_block_id = block_layout_in_cluster(local_block_id.x,n,Int<0>{});
          full_barrier_ptr_[stage].complete_transaction(dst_block_id, bytes, n!=local_block_id.y);
        }

        CUTLASS_PRAGMA_UNROLL
        for(int m = 0; m < size<0>(block_layout_in_cluster); ++m) {
          uint32_t dst_block_id = block_layout_in_cluster(m,local_block_id.y,Int<0>{});
          full_barrier_ptr_[stage].complete_transaction(dst_block_id, bytes, m!=local_block_id.x);
        }
      }
    #endif
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_try_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait, uint32_t var) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status = full_barrier_ptr_[var][stage].try_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }

  CUTLASS_DEVICE
  ConsumerToken consumer_test_wait(uint32_t stage, uint32_t phase, uint32_t skip_wait, uint32_t var) {
    if (skip_wait) {
      return {BarrierStatus::WaitDone};
    }
    uint32_t barrier_status = full_barrier_ptr_[var][stage].test_wait(phase);
    return {static_cast<BarrierStatus>(barrier_status)};
  }
  
  // Wait for producer to commit transactions (done by TMA)
  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, uint32_t var) {
    full_barrier_ptr_[var][stage].wait(phase);
  }

  // Wait for producer to commit transactions (done by TMA)
  CUTLASS_DEVICE
  void consumer_wait(uint32_t stage, uint32_t phase, ConsumerToken barrier_token, uint32_t var) {
    if (barrier_token == BarrierStatus::WaitAgain) {
      full_barrier_ptr_[var][stage].wait(phase);
    }
  }

  // Consumer signalling Producer of completion
  // Ensures all blocks in the Same Row and Column get notifed.
  CUTLASS_DEVICE
  void consumer_release(uint32_t stage, uint32_t var, uint32_t skip = false) {
    empty_barrier_ptr_[var][stage].arrive(dst_blockid_, is_signalling_thread_ & (!skip));
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Producer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }

  // Only arrive self-block, not support multicast
  CUTLASS_DEVICE
  void consumer_release_self(uint32_t stage, uint32_t var, uint32_t skip = false) {
    if ((!skip) && (threadIdx.x % NumThreadsPerWarpGroup == 0)) {
      empty_barrier_ptr_[var][stage].arrive();
    }
    #ifndef NDEBUG
    if (params_.role == ThreadCategory::Producer || params_.role == ThreadCategory::NonParticipant) {
      asm volatile ("brkpt;\n" ::);
    }
    #endif
  }

  CUTLASS_DEVICE
  ProducerBarrierType* producer_get_barrier(uint32_t stage, uint32_t var) {
    return reinterpret_cast<ProducerBarrierType*>(&full_barrier_ptr_[var][stage]);
  }
};

} // namespace cutlass