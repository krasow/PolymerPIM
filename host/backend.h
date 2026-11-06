#pragma once
// Device runtime under the event queue: places vectors (one shard per DPU)
// and runs operations in issue order.  Only sync() blocks.

#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common.h"
#include "detail/vector_desc.h"

class DpuOOMException : public std::runtime_error {
 public:
  DpuOOMException() : std::runtime_error("DPU OOM") {}
  explicit DpuOOMException(const std::string& msg) : std::runtime_error(msg) {}
};

class DpuBackend {
 public:
  // One DPU's view of the vectors in a launch.
  class Shard {
   public:
    virtual ~Shard() = default;
    virtual uint32_t dpu() const = 0;
    // MRAM offset and bytes (incl. reserved) of `vec` on this DPU.
    virtual uint32_t offset(const detail::VectorDesc& vec) const = 0;
    virtual uint32_t size_bytes(const detail::VectorDesc& vec) const = 0;
    // Global index of this DPU's first element of `vec`.
    virtual uint32_t first_element(const detail::VectorDesc& vec) const = 0;
  };

  // Fills one DPU's (uninitialised) arguments.
  using FillArgs = std::function<void(DPU_LAUNCH_ARGS&, const Shard&)>;
  // Sees the filled arguments; for logging.
  using InspectArgs = std::function<void(const DPU_LAUNCH_ARGS*, uint32_t)>;

  virtual ~DpuBackend() = default;

  // Allocates up to `num_dpus` DPUs with `binary`; returns how many.
  virtual uint32_t open(uint32_t num_dpus, const std::string& binary) = 0;
  virtual void close() = 0;
  virtual uint32_t num_dpus() const = 0;

  // Later launches run `binary`.  May load it everywhere at once, so nothing
  // may be in flight.
  virtual void load(const std::string& binary) = 0;

  // `n` elements sharded over the DPUs; lazy defers storage.  Throws
  // DpuOOMException.
  virtual detail::VectorDescRef allocate(size_t n, size_t reserved,
                                         size_t element_size, bool lazy) = 0;
  // `n_per_dpu` elements per DPU, read back in DPU order (e.g. partials).
  virtual detail::VectorDescRef allocate_per_dpu(size_t n_per_dpu,
                                                 size_t reserved,
                                                 size_t element_size,
                                                 bool lazy) = 0;
  // `n` elements held in full by every DPU.
  virtual detail::VectorDescRef allocate_local(size_t n,
                                               size_t element_size) = 0;
  // Gives a lazy vector storage.
  virtual void realize(const detail::VectorDescRef& vec) = 0;
  virtual void release(detail::VectorDesc& vec) = 0;

  // Copies `host` onto the shards; it must outlive the copy.
  virtual void write(const detail::VectorDescRef& vec, const char* host) = 0;
  // Where read() puts each shard in host memory.
  virtual detail::ShardLayout read_layout(const detail::VectorDesc& vec) = 0;
  // False until the producer is launched, for backends that place lazily.
  virtual bool layout_ready(const detail::VectorDesc&) { return true; }
  // Copies shard i to host + i * stride.
  virtual void read(const detail::VectorDescRef& vec, char* host,
                    size_t stride) = 0;

  // Runs a kernel that reads `reads` and fully overwrites `writes` (both:
  // in place).  Callbacks may run after return, so must own their captures.
  virtual void launch(const std::vector<detail::VectorDescRef>& reads,
                      const std::vector<detail::VectorDescRef>& writes,
                      const FillArgs& fill, const InspectArgs& inspect) = 0;

  // Runs `done`, possibly on another thread, once prior work completes.
  virtual void notify(std::function<void()> done) = 0;

  // Blocks until everything issued has completed.
  virtual void sync() = 0;
  // Lets pending notifications fire while the queue drains.
  virtual void progress() { sync(); }
  // Whether one kernel can use all of `vecs` without moving any.
  virtual bool can_share_kernel(const std::vector<detail::VectorDescRef>&) {
    return true;
  }

  virtual void read_logs(FILE* out) = 0;
};

// The UPMEM SDK driving one dpu_set as a single stream.
std::unique_ptr<DpuBackend> make_native_backend();
// upmem-legion; built only with RUNTIME=legion.
std::unique_ptr<DpuBackend> make_legion_backend();
