// Self-registering test framework for the vectordpu runtime.  See README.md.
//
// Declare with TEST(suite, name); fail by tripping a CHECK_* macro; SKIP if the
// build configuration makes the test inapplicable.  tf::measure also reports
// how many DPU kernel passes a region produced, which is what makes fusion
// testable.
#pragma once

#include <detail/vector.h>
#include <runtime.h>
#include <stats.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace tf {

// ---------------------------------------------------------------------------
// Outcome plumbing
// ---------------------------------------------------------------------------

struct Outcome {
  enum Kind { Pass, Fail, Skip };
  Kind kind = Pass;
  std::string message;
};

// One test runs at a time, so a single slot keeps the CHECK macros
// plumbing-free.
Outcome& current_outcome();
void fail(std::string message);
void skip(std::string message);

using TestFn = void (*)();

// What the current runtime is expected to do with this test.
enum class Expect {
  Pass,
  Fail,   // test is right, runtime is not: XFAIL, and XPASS once it is fixed
  Fatal,  // deadlocks or crashes; needs --run-known-fatal or --isolate
};

struct TestCase {
  const char* suite;
  const char* name;
  TestFn fn;
  const char* file;
  int line;
  Expect expect = Expect::Pass;
  const char* note = "";
};

void register_test(TestCase test);

struct Registrar {
  Registrar(const char* suite, const char* name, TestFn fn, const char* file,
            int line, Expect expect = Expect::Pass, const char* note = "") {
    register_test({suite, name, fn, file, line, expect, note});
  }
};

// ---------------------------------------------------------------------------
// Runtime configuration shared by all suites (overridable from the CLI)
// ---------------------------------------------------------------------------

size_t elements();
uint32_t dpus();
bool verbose();

// Reseeded before every test, so a failure is reproducible.
void reseed(uint64_t seed);
std::mt19937& rng();

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

template <typename T>
std::string str(const T& value) {
  std::ostringstream out;
  out << value;
  return out.str();
}

// ---------------------------------------------------------------------------
// Kernel-count measurement
// ---------------------------------------------------------------------------

// Drains every queued event so counters describe a closed region.
void drain();

// Counter baseline at construction, delta at close().  Use when the region
// cannot be expressed as a lambda.
class KernelProbe {
 public:
  KernelProbe() { reset(); }

  void reset() {
    drain();
    base_ = RuntimeStats::get().snapshot();
    closed_ = false;
  }

  // Fences, then returns the delta since construction.  Idempotent.
  const StatsSnapshot& close() {
    if (!closed_) {
      drain();
      delta_ = RuntimeStats::get().snapshot() - base_;
      closed_ = true;
    }
    return delta_;
  }

 private:
  StatsSnapshot base_;
  StatsSnapshot delta_;
  bool closed_ = false;
};

// Runs `body` as a closed region and returns the counters it produced.
template <typename F>
StatsSnapshot measure(F&& body) {
  KernelProbe probe;
  body();
  return probe.close();
}

// ---------------------------------------------------------------------------
// Data helpers
// ---------------------------------------------------------------------------

// Small by default so chained adds/muls cannot overflow int32 and disguise a
// fusion bug as a wrap-around.
template <typename T>
T random_value(T lo = -100, T hi = 100) {
  if constexpr (std::is_integral<T>::value) {
    std::uniform_int_distribution<int64_t> dist(lo, hi);
    return static_cast<T>(dist(rng()));
  } else {
    std::uniform_real_distribution<double> dist(lo, hi);
    return static_cast<T>(dist(rng()));
  }
}

template <typename T>
std::vector<T> random_vector(size_t n, T lo = -100, T hi = 100) {
  std::vector<T> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = random_value<T>(lo, hi);
  return out;
}

template <typename T>
std::vector<T> iota_vector(size_t n, T start = 0, T step = 1) {
  std::vector<T> out(n);
  T v = start;
  for (size_t i = 0; i < n; ++i, v = static_cast<T>(v + step)) out[i] = v;
  return out;
}

template <typename T>
std::vector<T> constant_vector(size_t n, T value) {
  return std::vector<T>(n, value);
}

// Reports the first few mismatches plus a total, so a single corrupted lane is
// easy to localise.
template <typename T>
bool vec_equal(const std::vector<T>& actual, const std::vector<T>& expected,
               std::string& message, double tolerance = 0.0) {
  if (actual.size() != expected.size()) {
    message = "size " + str(actual.size()) + " != " + str(expected.size());
    return false;
  }
  size_t mismatches = 0;
  std::ostringstream detail;
  for (size_t i = 0; i < actual.size(); ++i) {
    bool ok = tolerance > 0.0
                  ? std::fabs(static_cast<double>(actual[i]) -
                              static_cast<double>(expected[i])) <= tolerance
                  : actual[i] == expected[i];
    if (ok) continue;
    if (mismatches < 4) {
      if (mismatches) detail << ", ";
      detail << "[" << i << "] got " << actual[i] << " want " << expected[i];
    }
    mismatches++;
  }
  if (mismatches == 0) return true;
  message = str(mismatches) + "/" + str(actual.size()) +
            " lanes differ: " + detail.str();
  return false;
}

// ---------------------------------------------------------------------------
// Build-configuration limits, so expectations survive a parameter sweep
// ---------------------------------------------------------------------------

constexpr bool pipeline_enabled() { return PIPELINE != 0; }
constexpr bool jit_enabled() { return JIT != 0; }

// Reduction chains that can share one kernel pass.
constexpr size_t max_reduction_chains() {
  return MAX_HFUSE_CHAINS < MAX_SAFE_HFUSED_REDUCTION_CHAINS
             ? (size_t)MAX_HFUSE_CHAINS
             : (size_t)MAX_SAFE_HFUSED_REDUCTION_CHAINS;
}

// Chains per pass: the primary plus MAX_HFUSE_CHAINS-1 extras.
constexpr size_t max_hfuse_chains() { return (size_t)MAX_HFUSE_CHAINS; }
constexpr size_t max_combined_inputs() { return (size_t)MAX_COMBINED_INPUTS; }
constexpr size_t max_vfuse_ops() { return (size_t)MAX_VFUSE_OPS; }
constexpr size_t queue_absorb_limit() { return (size_t)QUEUE_ABSORB_LIMIT; }

// Turns a limit into an expected kernel count.
constexpr size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

template <typename T>
std::vector<typename dpu_vector<T>::reduction_result_t> get_all(
    std::vector<dpu_future<T>>& futures) {
  std::vector<typename dpu_vector<T>::reduction_result_t> results;
  results.reserve(futures.size());
  if (!futures.empty()) dpu_fence();
  for (auto& future : futures) results.push_back(future.get());
  return results;
}

}  // namespace tf

// ---------------------------------------------------------------------------
// Declaration + assertion macros
// ---------------------------------------------------------------------------

#define TF_DECLARE_TEST(suite_name, test_name, expect, note)                  \
  static void tf_test_##suite_name##_##test_name();                           \
  static ::tf::Registrar tf_reg_##suite_name##_##test_name(                   \
      #suite_name, #test_name, &tf_test_##suite_name##_##test_name, __FILE__, \
      __LINE__, (expect), (note));                                            \
  static void tf_test_##suite_name##_##test_name()

#define TEST(suite_name, test_name) \
  TF_DECLARE_TEST(suite_name, test_name, ::tf::Expect::Pass, "")

// Reported as XFAIL, and as a failure if it starts passing, so markers cannot
// rot.
#define TEST_XFAIL(suite_name, test_name, reason) \
  TF_DECLARE_TEST(suite_name, test_name, ::tf::Expect::Fail, reason)

// The test hangs or crashes the process; skipped unless --run-known-fatal.
#define TEST_KNOWN_FATAL(suite_name, test_name, reason) \
  TF_DECLARE_TEST(suite_name, test_name, ::tf::Expect::Fatal, reason)

// For bugs that only exist when fusion is compiled in: with PIPELINE=0 the same
// test passes, and a stale marker would report XPASS and fail the run.
#if PIPELINE
#define TEST_XFAIL_IF_FUSED(suite_name, test_name, reason) \
  TEST_XFAIL(suite_name, test_name, reason)
#define TEST_KNOWN_FATAL_IF_FUSED(suite_name, test_name, reason) \
  TEST_KNOWN_FATAL(suite_name, test_name, reason)
#else
#define TEST_XFAIL_IF_FUSED(suite_name, test_name, reason) \
  TEST(suite_name, test_name)
#define TEST_KNOWN_FATAL_IF_FUSED(suite_name, test_name, reason) \
  TEST(suite_name, test_name)
#endif

#define TF_FAIL_AT(msg)                                                 \
  do {                                                                  \
    ::tf::fail(std::string(__FILE__ ":") + ::tf::str(__LINE__) + ": " + \
               (msg));                                                  \
    return;                                                             \
  } while (0)

#define SKIP(reason)      \
  do {                    \
    ::tf::skip((reason)); \
    return;               \
  } while (0)

#define CHECK(cond)                                     \
  do {                                                  \
    if (!(cond)) TF_FAIL_AT("CHECK(" #cond ") failed"); \
  } while (0)

#define TF_CHECK_BINOP(a, b, op, opname)                           \
  do {                                                             \
    auto tf_a = (a);                                               \
    auto tf_b = (b);                                               \
    if (!(tf_a op tf_b))                                           \
      TF_FAIL_AT(std::string(opname "(" #a ", " #b ") failed: ") + \
                 ::tf::str(tf_a) + " vs " + ::tf::str(tf_b));      \
  } while (0)

#define CHECK_EQ(a, b) TF_CHECK_BINOP(a, b, ==, "CHECK_EQ")
#define CHECK_NE(a, b) TF_CHECK_BINOP(a, b, !=, "CHECK_NE")
#define CHECK_LT(a, b) TF_CHECK_BINOP(a, b, <, "CHECK_LT")
#define CHECK_LE(a, b) TF_CHECK_BINOP(a, b, <=, "CHECK_LE")
#define CHECK_GT(a, b) TF_CHECK_BINOP(a, b, >, "CHECK_GT")
#define CHECK_GE(a, b) TF_CHECK_BINOP(a, b, >=, "CHECK_GE")

#define CHECK_NEAR(a, b, tol)                                           \
  do {                                                                  \
    double tf_a = (double)(a);                                          \
    double tf_b = (double)(b);                                          \
    if (!(std::fabs(tf_a - tf_b) <= (double)(tol)))                     \
      TF_FAIL_AT(std::string("CHECK_NEAR(" #a ", " #b ") failed: ") +   \
                 ::tf::str(tf_a) + " vs " + ::tf::str(tf_b) + " tol " + \
                 ::tf::str((double)(tol)));                             \
  } while (0)

#define CHECK_VEC_EQ(actual, expected)                                       \
  do {                                                                       \
    std::string tf_msg;                                                      \
    if (!::tf::vec_equal((actual), (expected), tf_msg))                      \
      TF_FAIL_AT(std::string("CHECK_VEC_EQ(" #actual ", " #expected "): ") + \
                 tf_msg);                                                    \
  } while (0)

#define CHECK_VEC_NEAR(actual, expected, tol)                                  \
  do {                                                                         \
    std::string tf_msg;                                                        \
    if (!::tf::vec_equal((actual), (expected), tf_msg, (tol)))                 \
      TF_FAIL_AT(std::string("CHECK_VEC_NEAR(" #actual ", " #expected "): ") + \
                 tf_msg);                                                      \
  } while (0)

// Kernel-count assertions on a StatsSnapshot.  The failure message carries the
// whole counter set, which usually shows *why* fusion did or did not happen.
// With PIPELINE=0 they become no-ops so the value checks around them still run.
#if PIPELINE

#define CHECK_KERNELS_EQ(snap, expected)                                       \
  do {                                                                         \
    const StatsSnapshot& tf_s = (snap);                                        \
    if (tf_s.compute_launches != (size_t)(expected))                           \
      TF_FAIL_AT(std::string("expected ") + ::tf::str((size_t)(expected)) +    \
                 " kernel pass(es), got " + ::tf::str(tf_s.compute_launches) + \
                 "  [" + tf_s.to_string() + "]");                              \
  } while (0)

#define CHECK_KERNELS_LE(snap, limit)                                          \
  do {                                                                         \
    const StatsSnapshot& tf_s = (snap);                                        \
    if (tf_s.compute_launches > (size_t)(limit))                               \
      TF_FAIL_AT(std::string("expected at most ") +                            \
                 ::tf::str((size_t)(limit)) + " kernel pass(es), got " +       \
                 ::tf::str(tf_s.compute_launches) + "  [" + tf_s.to_string() + \
                 "]");                                                         \
  } while (0)

#define CHECK_KERNELS_GE(snap, limit)                                          \
  do {                                                                         \
    const StatsSnapshot& tf_s = (snap);                                        \
    if (tf_s.compute_launches < (size_t)(limit))                               \
      TF_FAIL_AT(std::string("expected at least ") +                           \
                 ::tf::str((size_t)(limit)) + " kernel pass(es), got " +       \
                 ::tf::str(tf_s.compute_launches) + "  [" + tf_s.to_string() + \
                 "]");                                                         \
  } while (0)

#define CHECK_KERNELS_LT(snap, limit)                                          \
  do {                                                                         \
    const StatsSnapshot& tf_s = (snap);                                        \
    if (!(tf_s.compute_launches < (size_t)(limit)))                            \
      TF_FAIL_AT(std::string("expected fewer than ") +                         \
                 ::tf::str((size_t)(limit)) + " kernel pass(es), got " +       \
                 ::tf::str(tf_s.compute_launches) + "  [" + tf_s.to_string() + \
                 "]");                                                         \
  } while (0)

// Minimum merges the fusion passes must have performed.
#define CHECK_FUSIONS_GE(snap, expected)                                  \
  do {                                                                    \
    const StatsSnapshot& tf_s = (snap);                                   \
    size_t tf_merges = tf_s.fused_away();                                 \
    if (tf_merges < (size_t)(expected))                                   \
      TF_FAIL_AT(std::string("expected at least ") +                      \
                 ::tf::str((size_t)(expected)) + " fusion merges, got " + \
                 ::tf::str(tf_merges) + "  [" + tf_s.to_string() + "]");  \
  } while (0)

#else  // !PIPELINE

#define CHECK_KERNELS_EQ(snap, expected) ((void)(snap))
#define CHECK_KERNELS_LE(snap, limit) ((void)(snap))
#define CHECK_KERNELS_GE(snap, limit) ((void)(snap))
#define CHECK_KERNELS_LT(snap, limit) ((void)(snap))
#define CHECK_FUSIONS_GE(snap, expected) ((void)(snap))

#endif  // PIPELINE
