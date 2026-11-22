// Every reduction kind, the lazy-future API, and the edge cases a tree
// reduction across DPUs and tasklets gets wrong first: single element,
// all-negative, identity values, ragged sizes.

#include "framework.h"

namespace {

using T = int32_t;

}  // namespace

TEST(reductions, sum_matches_host) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -100, 100);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  int64_t actual = (int64_t)sum(da).get();

  int64_t expected = 0;
  for (T x : a) expected += x;
  CHECK_EQ(actual, expected);
}

// Totals far past int32: per-tasklet, per-DPU and cross-DPU sums must all be
// carried in int64.
TEST(reductions, sum_promotes_past_int32) {
#if !ENABLE_PROMOTION_REDUCTIONS
  SKIP("requires ENABLE_PROMOTION_REDUCTIONS=1");
#else
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, 1000000000, 2000000000);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  int64_t actual = sum(da).get();

  int64_t expected = 0;
  for (T x : a) expected += x;
  CHECK_EQ(actual, expected);
#endif
}

// The same through a fused pipeline kernel, whose accumulator is int64 too.
TEST(reductions, fused_sum_promotes_past_int32) {
#if !ENABLE_PROMOTION_REDUCTIONS
  SKIP("requires ENABLE_PROMOTION_REDUCTIONS=1");
#else
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, 100000000, 1000000000);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  int64_t actual = sum(da * 2).get();

  int64_t expected = 0;
  for (T x : a) expected += 2 * (int64_t)x;
  CHECK_EQ(actual, expected);
#endif
}

// Scalars reach the DPU as uint32; on a promoted (int64) stack a negative one
// must still be negative.
TEST(reductions, fused_sum_with_negative_scalar) {
#if !ENABLE_PROMOTION_REDUCTIONS
  SKIP("requires ENABLE_PROMOTION_REDUCTIONS=1");
#else
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, 100000000, 700000000);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  int64_t actual = sum(da * -3).get();

  int64_t expected = 0;
  for (T x : a) expected += -3 * (int64_t)x;
  CHECK_EQ(actual, expected);
#endif
}

TEST(reductions, min_matches_host) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -1000, 1000);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  int64_t actual = (int64_t)min(da).get();

  T expected = a[0];
  for (T x : a)
    if (x < expected) expected = x;
  CHECK_EQ(actual, (int64_t)expected);
}

TEST(reductions, max_matches_host) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -1000, 1000);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  int64_t actual = (int64_t)max(da).get();

  T expected = a[0];
  for (T x : a)
    if (x > expected) expected = x;
  CHECK_EQ(actual, (int64_t)expected);
}

// Mostly 1s with a few small factors: checks identity handling, not range.
TEST(reductions, product_matches_host) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::constant_vector<T>(n, 1);
  a[0] = 3;
  a[n / 2] = -5;
  a[n - 1] = 7;
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  int64_t actual = (int64_t)product(da).get();

  int64_t expected = 1;
  for (T x : a) expected *= x;
  CHECK_EQ(actual, expected);
}

// Must return the element, not the identity.
TEST(reductions, single_element) {
  std::vector<T> a = {42};
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  CHECK_EQ((int64_t)sum(da).get(), (int64_t)42);
}

// min/max over a single element used to fold in to_cpu's zero padding.
TEST(reductions, single_element_min_max) {
  std::vector<T> a = {-7};
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(a);

  CHECK_EQ((int64_t)min(da).get(), (int64_t)-7);
  CHECK_EQ((int64_t)max(db).get(), (int64_t)-7);
}

// Exposes an accumulator seeded with 0 rather than the first element.
TEST(reductions, all_negative_min_max) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -500, -1);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(a);

  T want_min = a[0], want_max = a[0];
  for (T x : a) {
    if (x < want_min) want_min = x;
    if (x > want_max) want_max = x;
  }
  CHECK_EQ((int64_t)min(da).get(), (int64_t)want_min);
  CHECK_EQ((int64_t)max(db).get(), (int64_t)want_max);
}

TEST(reductions, all_positive_min_max) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, 1, 500);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(a);

  T want_min = a[0], want_max = a[0];
  for (T x : a) {
    if (x < want_min) want_min = x;
    if (x > want_max) want_max = x;
  }
  CHECK_EQ((int64_t)min(da).get(), (int64_t)want_min);
  CHECK_EQ((int64_t)max(db).get(), (int64_t)want_max);
}

TEST(reductions, sum_of_zeros) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::constant_vector<T>(n, 0);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  CHECK_EQ((int64_t)sum(da).get(), (int64_t)0);
}

// Sum of ones == element count, at every size the readback path handles.
// Catches a reduction that loses or double-counts a block or shard tail.
TEST(reductions, sum_at_all_sizes) {
  for (size_t n :
       {(size_t)1, (size_t)BLOCK_SIZE - 1, (size_t)BLOCK_SIZE,
        (size_t)BLOCK_SIZE * 2, (size_t)1000, (size_t)9973, (size_t)65537}) {
    std::vector<T> a = tf::constant_vector<T>(n, 1);
    dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
    int64_t actual = (int64_t)sum(da).get();
    if (actual != (int64_t)n) {
      tf::fail("n=" + tf::str(n) + ": sum of ones is " + tf::str(actual) +
               ", expected " + tf::str(n));
      return;
    }
  }
}

// Must see the fused values, not the intermediate's never-written MRAM.
TEST(reductions, sum_over_fused_expression) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -20, 20);
  std::vector<T> b = tf::random_vector<T>(n, -20, 20);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);

  int64_t actual = (int64_t)sum((da + db) * (T)2).get();

  int64_t expected = 0;
  for (size_t i = 0; i < n; ++i) expected += (int64_t)(a[i] + b[i]) * 2;
  CHECK_EQ(actual, expected);
}

// The value must not depend on when get() runs relative to other queued work.
TEST(reductions, lazy_future_get_after_other_work) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::random_vector<T>(n, -20, 20);
  std::vector<T> b = tf::random_vector<T>(n, -20, 20);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);
  dpu_vector<T> db = dpu_vector<T>::from_cpu(b);

  auto future = sum(da);

  // Unrelated work submitted between the reduction and its read.
  std::vector<T> other = (da + db).to_cpu();

  int64_t expected = 0;
  for (T x : a) expected += x;
  CHECK_EQ((int64_t)future.get(), expected);

  std::vector<T> expected_other(n);
  for (size_t i = 0; i < n; ++i) expected_other[i] = a[i] + b[i];
  CHECK_VEC_EQ(other, expected_other);
}

// Collecting futures together keeps reductions queued long enough to fuse.
TEST(reductions, future_vector_matches_eager_reads) {
  const size_t count = 6;
  const size_t n = 1024;
  std::vector<std::vector<T>> host;
  std::vector<dpu_vector<T>> vecs;
  for (size_t i = 0; i < count; ++i) {
    host.push_back(tf::random_vector<T>(n, -50, 50));
    vecs.push_back(dpu_vector<T>::from_cpu(host.back()));
  }
  tf::drain();

  std::vector<dpu_future<T>> futures;
  for (size_t i = 0; i < count; ++i) futures.push_back(sum(vecs[i]));
  std::vector<int64_t> lazy;
  for (auto value : tf::get_all(futures)) lazy.push_back((int64_t)value);

  CHECK_EQ(lazy.size(), count);
  for (size_t i = 0; i < count && i < lazy.size(); ++i) {
    int64_t expected = 0;
    for (T x : host[i]) expected += x;
    if (lazy[i] != expected)
      tf::fail("reduction " + tf::str(i) + ": got " + tf::str(lazy[i]) +
               " want " + tf::str(expected));
  }
}

// Must observe pre-overwrite values by dependency tracking, not by luck.
TEST(reductions, reduction_ordering_against_later_write) {
  const size_t n = tf::elements();
  std::vector<T> a = tf::constant_vector<T>(n, 5);
  dpu_vector<T> da = dpu_vector<T>::from_cpu(a);

  auto before = sum(da);
  int64_t before_value = (int64_t)before.get();
  CHECK_EQ(before_value, (int64_t)(5 * (int64_t)n));

  dpu_vector<T> doubled = da * (T)2;
  auto after = sum(doubled);
  CHECK_EQ((int64_t)after.get(), (int64_t)(10 * (int64_t)n));
}
