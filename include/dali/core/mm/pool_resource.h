// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_CORE_MM_POOL_RESOURCE_H_
#define DALI_CORE_MM_POOL_RESOURCE_H_

#include <mutex>
#include "dali/core/mm/memory_resource.h"
#include "dali/core/mm/detail/free_list.h"
#include "dali/core/small_vector.h"
#include "dali/core/util.h"

namespace dali {
namespace mm {

struct pool_options {
  /**
   * @brief Maximum block size
   *
   * Growth stops at this point; larger blocks are allocated only when allocate is called with
   * a larger memory requirements.
   */
  size_t max_block_size = static_cast<size_t>(-1);  // no limit
  /// Minimum size of blocks requested from upstream
  size_t min_block_size = (1 << 12);
  /// The factor by which the allocation size grows until it reaches max_block_size
  float growth_factor = 2;
  /**
   * @brief Whether to try to allocate smaller blocks from upstream if default upcoming
   *        block is unavailable.
   */
  bool try_smaller_on_failure = true;
  /**
   * @brief Whether to try to return completely free blocks to the upstream when an allocation
   *        from upstream failed. This may effectively flush the pool.
   *
   * @remarks This option is ignored when `try_smaller_on_failure` is set to `false`.
   */
  bool return_to_upstream_on_failure = true;
  size_t upstream_alignment = 256;
};

constexpr pool_options default_host_pool_opts() noexcept {
  return { (1 << 28), (1 << 12), 2.0f, true, true };
}

constexpr pool_options default_device_pool_opts() noexcept {
  return { (static_cast<size_t>(1) << 32), (1 << 20), 2.0f, true, true };
}

template <memory_kind kind, typename Context, class FreeList, class LockType>
class pool_resource_base : public memory_resource<kind, Context> {
 public:
  explicit pool_resource_base(memory_resource<kind, Context> *upstream = nullptr,
                              const pool_options opt = {})
  : upstream_(upstream), options_(opt) {
     next_block_size_ = opt.min_block_size;
  }

  pool_resource_base(const pool_resource_base &) = delete;
  pool_resource_base(pool_resource_base &&) = delete;

  ~pool_resource_base() {
    free_all();
  }

  void free_all() {
    for (auto &block : blocks_) {
      upstream_->deallocate(block.ptr, block.bytes, block.alignment);
    }
    blocks_.clear();
    free_list_.clear();
  }

  /**
   * @brief Tries to obtain a block from the internal free list.
   *
   * Allocates `bytes` memory from the free list. If a block that satisifies
   * the size or alignment requirements is not found, the function returns
   * nullptr withoug allocating from upstream.
   */
  void *try_allocate_from_free(size_t bytes, size_t alignment) {
    if (!bytes)
      return nullptr;

    {
      lock_guard guard(lock_);
      return free_list_.get(bytes, alignment);
    }
  }

 protected:
  void *do_allocate(size_t bytes, size_t alignment) override {
    if (!bytes)
      return nullptr;

    {
      lock_guard guard(lock_);
      void *ptr = free_list_.get(bytes, alignment);
      if (ptr)
        return ptr;
    }
    alignment = std::max(alignment, options_.upstream_alignment);
    size_t blk_size = bytes;
    void *new_block = get_upstream_block(blk_size, bytes, alignment);
    assert(new_block);
    try {
      lock_guard guard(lock_);
      blocks_.push_back({ new_block, blk_size, alignment });
      if (blk_size == bytes) {
        // we've allocated a block exactly of the required size - there's little
        // chance that it will be merged with anything in the pool, so we'll return it as-is
        return new_block;
      } else {
        // we've allocated an oversized block - put the remainder in the free list
        lock_guard guard(lock_);
        free_list_.put(static_cast<char *>(new_block) + bytes, blk_size - bytes);
        return new_block;
      }
    } catch (...) {
      upstream_->deallocate(new_block, blk_size, alignment);
      throw;
    }
  }

  void do_deallocate(void *ptr, size_t bytes, size_t alignment) override {
    lock_guard guard(lock_);
    free_list_.put(ptr, bytes);
  }

  void *get_upstream_block(size_t &blk_size, size_t min_bytes, size_t alignment) {
    blk_size = next_block_size(min_bytes);
    bool tried_return_to_upstream = false;
    for (;;) {
      try {
        return upstream_->allocate(blk_size, alignment);
      } catch (const std::bad_alloc &) {
        if (!options_.try_smaller_on_failure)
          throw;
        if (blk_size == min_bytes) {
          // We've reached the minimum size and still got no memory from upstream
          // - try to free something.
          if (tried_return_to_upstream || !options_.return_to_upstream_on_failure)
            throw;
          if (blocks_.empty())  // nothing to free -> fail
            throw;
          // If there are some upstream blocks which are completely free
          // (the free list covers them completely), we can try to return them
          // to the upstream, with the hope that it will reorganize and succeed in
          // the subsequent allocation attempt.
          int blocks_freed = 0;
          for (int i = blocks_.size() - 1; i >= 0; i--) {
            UpstreamBlock blk = blocks_[i];
            // If we can remove the block from the free list, it
            // means that there are no suballocations from this block
            // - we can safely free it to the upstream.
            if (free_list_.remove_if_in_list(blk.ptr, blk.bytes)) {
              upstream_->deallocate(blk.ptr, blk.bytes, blk.alignment);
              blocks_.erase_at(i);
              blocks_freed++;
            }
          }
          if (!blocks_freed)
            throw;  // we freed nothing, so there's no point in retrying to allocate
          // mark that we've tried, so we can fail fast the next time
          tried_return_to_upstream = true;
        }
        blk_size = std::max(min_bytes, blk_size >> 1);

        // Shrink the next_block_size_, so that we don't try to allocate a big block
        // next time, because it would likely fail anyway.
        next_block_size_ = blk_size;
      }
    }
  }

  virtual Context do_get_context() const noexcept {
    return upstream_->get_context();
  }

  size_t next_block_size(size_t upcoming_allocation_size) {
    size_t actual_block_size = std::max<size_t>(upcoming_allocation_size,
                                                next_block_size_ * options_.growth_factor);
    // Align the upstream block to reduce fragmentation.
    // The upstream resource (e.g. OS routine) may return blocks that have
    // coarse size granularity. This may result in fragmentation - the next
    // large block will be overaligned and we'll never see the padding.
    // Even though we might have received contiguous memory, we're not aware of that.
    // To reduce the probability of this happening, we align the size to 1/1024th
    // of the allocation size or 4kB (typical page size), whichever is larger.
    // This makes (at least sometimes) the large blocks to be seen as adjacent
    // and therefore enables coalescing in the free list.
    size_t alignment = 1uL << std::max((ilog2(actual_block_size) - 10), 12);
    actual_block_size = align_up(actual_block_size, alignment);
    next_block_size_ = std::min<size_t>(actual_block_size, options_.max_block_size);
    return actual_block_size;
  }

  memory_resource<kind, Context> *upstream_;
  FreeList free_list_;
  LockType lock_;
  pool_options options_;
  size_t next_block_size_ = 0;

  struct UpstreamBlock {
    void *ptr;
    size_t bytes, alignment;
  };

  SmallVector<UpstreamBlock, 16> blocks_;
  using lock_guard = std::lock_guard<LockType>;
  using unique_lock = std::unique_lock<LockType>;
};

}  // namespace mm
}  // namespace dali

#endif  // DALI_CORE_MM_POOL_RESOURCE_H_
