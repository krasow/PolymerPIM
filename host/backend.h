#pragma once
// The device runtime underneath DpuRuntime.  The event queue decides what runs
// and in which order; a backend owns where vectors live and carries each
// operation out.
//
// A vector is split into one contiguous shard per DPU.  Which DPUs, and where
// in MRAM, is the backend's decision: the front end only names vectors and
// reads per-DPU placement through a Shard while it fills launch arguments.
//
// Every operation is issued onto one in-order stream: it starts after
// everything issued before it has completed.  Nothing here blocks the caller
// except sync().

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
    // MRAM offset and bytes (reserved bytes included) of `vec` on this DPU.
    virtual uint32_t offset(const detail::VectorDesc& vec) const = 0;
    virtual uint32_t size_bytes(const detail::VectorDesc& vec) const = 0;
    // Global index of this DPU's first element of `vec`.
    virtual uint32_t first_element(const detail::VectorDesc& vec) const = 0;
  };

  // Writes one DPU's arguments into an uninitialised struct.
  using FillArgs = std::function<void(DPU_LAUNCH_ARGS&, const Shard&)>;
  // Sees the filled arguments before they are sent; for logging.
  using InspectArgs = std::function<void(const DPU_LAUNCH_ARGS*, uint32_t)>;

  virtual ~DpuBackend() = default;

  // Allocates up to `num_dpus` DPUs and loads `binary` onto them.  Returns the
  // number actually allocated.
  virtual uint32_t open(uint32_t num_dpus, const std::string& binary) = 0;
  virtual void close() = 0;
  virtual uint32_t num_dpus() const = 0;

  // Launches from now on run `binary`.  A backend may load it on every DPU
  // at once, so the caller must ensure nothing is in flight, or only on the
  // DPUs each launch runs on.
  virtual void load(const std::string& binary) = 0;

  // A vector of `n` elements sharded across every DPU, each shard preceded by
  // `reserved` bytes.  A lazy vector gets its storage on first use.  Throws
  // DpuOOMException when MRAM is exhausted.
  virtual detail::VectorDescRef allocate(size_t n, size_t reserved,
                                         size_t element_size, bool lazy) = 0;
  // `n_per_dpu` elements on every DPU, each slot preceded by `reserved`
  // bytes; read back in DPU order.  For per-DPU results such as reduction
  // partials.
  virtual detail::VectorDescRef allocate_per_dpu(size_t n_per_dpu,
                                                 size_t reserved,
                                                 size_t element_size,
                                                 bool lazy) = 0;
  // A vector of `n` elements held in full by every DPU.
  virtual detail::VectorDescRef allocate_local(size_t n,
                                               size_t element_size) = 0;
  // Gives a lazy vector its storage; a no-op once it has some.
  virtual void realize(const detail::VectorDescRef& vec) = 0;
  virtual void release(detail::VectorDesc& vec) = 0;

  // Copies `host`, the vector's elements in order, onto its shards.  `host`
  // must stay valid until the copy completes.
  virtual void write(const detail::VectorDescRef& vec, const char* host) = 0;
  // How read() lays the vector out in host memory: one block per shard, at a
  // uniform stride, in element order.  The vector must have storage.
  virtual detail::ShardLayout read_layout(const detail::VectorDesc& vec) = 0;
  // False until read_layout(vec) is known, which some backends settle only
  // once the vector's producer has been launched.
  virtual bool layout_ready(const detail::VectorDesc&) { return true; }
  // Copies every shard into `host`, shard i's at i * stride, where stride is
  // read_layout(vec).stride.
  virtual void read(const detail::VectorDescRef& vec, char* host,
                    size_t stride) = 0;

  // Starts a kernel on every DPU.  It reads `reads` and overwrites `writes`
  // in full; a vector in both is updated in place (null entries are skipped).
  // `fill` writes each DPU's arguments.  Both callbacks may run after launch()
  // returns, so they must own what they capture.
  virtual void launch(const std::vector<detail::VectorDescRef>& reads,
                      const std::vector<detail::VectorDescRef>& writes,
                      const FillArgs& fill, const InspectArgs& inspect) = 0;

  // Runs `done` once everything issued before it has completed.  `done` may
  // run on another thread.
  virtual void notify(std::function<void()> done) = 0;

  // Blocks until everything issued so far has completed.
  virtual void sync() = 0;

  virtual void read_logs(FILE* out) = 0;
};

// The UPMEM SDK driving one dpu_set as a single stream.
std::unique_ptr<DpuBackend> make_native_backend();
// upmem-legion: vectors as regions, kernels as index launches over ranks.
// Built only with RUNTIME=legion.
std::unique_ptr<DpuBackend> make_legion_backend();
