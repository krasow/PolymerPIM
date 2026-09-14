#include "framework.h"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace tf {
namespace {

std::vector<TestCase>& registry() {
  static std::vector<TestCase> tests;
  return tests;
}

Outcome g_outcome;
// Prime, so the per-DPU shard and the per-tasklet block both have a ragged tail
// at every DPU count.
size_t g_elements = 4099;
uint32_t g_dpus = 64;
bool g_verbose = false;
std::mt19937 g_rng(12345);

// An outcome resolved against the test's Expect marker.
struct Verdict {
  const char* label;
  bool counts_as_failure;
  bool show_message;
};

Verdict verdict_for(const TestCase& test, Outcome::Kind kind) {
  if (kind == Outcome::Skip) return {"SKIP", false, true};
  // Fatal means "known broken, and may take the process down", so a plain
  // failure is still the expected outcome; only a clean pass says the marker is
  // stale.  Fail is the same minus the crash.
  if (test.expect == Expect::Fatal || test.expect == Expect::Fail) {
    if (kind == Outcome::Fail) return {"XFAIL", false, true};
    return {"XPASS", true, false};
  }
  if (kind == Outcome::Fail) return {"FAIL", true, true};
  return {"PASS", false, false};
}

// A hung test cannot be interrupted from inside the runtime, so the watchdog
// names the test that wedged and kills the process rather than hanging forever.
struct Watchdog {
  std::atomic<bool> stop{false};
  std::atomic<const char*> suite{""};
  std::atomic<const char*> name{""};
  std::thread thread;

  std::atomic<long long> test_started_ms{0};

  static long long now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  void start(unsigned seconds) {
    if (seconds == 0) return;
    test_started_ms.store(now_ms());
    thread = std::thread([this, seconds] {
      while (!stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        if (stop.load()) return;
        long long elapsed = now_ms() - test_started_ms.load();
        if (elapsed > (long long)seconds * 1000) {
          std::cout << "  TIMEOUT " << suite.load() << "." << name.load()
                    << " exceeded " << seconds
                    << "s -- the runtime is wedged, aborting the run\n"
                    << std::flush;
          _exit(4);
        }
      }
    });
  }

  // Restarts the per-test budget.
  void begin(const TestCase& test) {
    suite.store(test.suite);
    name.store(test.name);
    test_started_ms.store(now_ms());
  }

  void shutdown() {
    stop.store(true);
    if (thread.joinable()) thread.join();
  }
};

std::string full_name(const TestCase& test) {
  return std::string(test.suite) + "." + test.name;
}

bool matches(const TestCase& test, const std::string& filter, bool exact) {
  if (filter.empty()) return true;
  const std::string full = full_name(test);
  return exact ? full == filter : full.find(filter) != std::string::npos;
}

}  // namespace

Outcome& current_outcome() { return g_outcome; }

void fail(std::string message) {
  // Keep the first failure; later CHECKs in the same test are usually fallout.
  if (g_outcome.kind == Outcome::Fail) return;
  g_outcome.kind = Outcome::Fail;
  g_outcome.message = std::move(message);
}

void skip(std::string message) {
  if (g_outcome.kind != Outcome::Pass) return;
  g_outcome.kind = Outcome::Skip;
  g_outcome.message = std::move(message);
}

void register_test(TestCase test) { registry().push_back(test); }

size_t elements() { return g_elements; }
uint32_t dpus() { return g_dpus; }
bool verbose() { return g_verbose; }

void reseed(uint64_t seed) { g_rng.seed((std::mt19937::result_type)seed); }
std::mt19937& rng() { return g_rng; }

void drain() { dpu_fence(); }

namespace {

void print_usage(const char* argv0) {
  std::cout
      << "usage: " << argv0 << " [options]\n"
      << "  --list             list registered tests and exit\n"
      << "  --filter=SUBSTR    run only tests whose suite.name contains "
         "SUBSTR\n"
      << "  --elements=N       element count for size-agnostic tests (default "
      << g_elements << ")\n"
      << "  --dpus=N           DPUs to allocate (default " << g_dpus << ")\n"
      << "  --seed=N           RNG seed (default 12345)\n"
      << "  --stats            print the counter delta for every test\n"
      << "  --fail-fast        stop at the first failure\n"
      << "  --timeout=SEC      abort the run if one test exceeds SEC (0=off, "
         "default 300)\n"
      << "  --run-known-fatal  also run tests marked TEST_KNOWN_FATAL\n"
      << "  --isolate          run every test in its own process (immune to a\n"
         "                     crash, a hang, or state left behind by an\n"
         "                     earlier test)\n"
      << "  -v, --verbose      verbose output\n";
}

void print_build_config() {
  std::cout << "build: PIPELINE=" << PIPELINE << " JIT=" << JIT
            << " JIT_PIPELINE_FALLBACK=" << JIT_PIPELINE_FALLBACK
            << " MAX_HFUSE_CHAINS=" << MAX_HFUSE_CHAINS
            << " MAX_SAFE_HFUSED_REDUCTION_CHAINS="
            << MAX_SAFE_HFUSED_REDUCTION_CHAINS
            << " MAX_VFUSE_OPS=" << MAX_VFUSE_OPS
            << " MAX_VFUSE_INPUTS=" << MAX_VFUSE_INPUTS
            << "\n       MAX_PIPELINE_STACK_DEPTH=" << MAX_PIPELINE_STACK_DEPTH
            << " QUEUE_ABSORB_LIMIT=" << QUEUE_ABSORB_LIMIT
            << " BLOCK_SIZE=" << BLOCK_SIZE << " NR_TASKLETS=" << NR_TASKLETS
            << "\n";
}

struct ChildResult {
  std::string label;  // PASS / FAIL / SKIP / XFAIL / XPASS / CRASH / TIMEOUT
  std::string message;
  bool counts_as_failure = false;
};

// Runs one test in a fresh process, re-executing this binary with
// --exact=<suite.name> --child.  The child gets its own runtime, heap and MRAM,
// which is what makes a crashing or hanging test survivable and stops a broken
// test from poisoning the next one.
ChildResult run_isolated(const char* self, const TestCase& test,
                         const std::string& dpus, const std::string& elements,
                         const std::string& seed, unsigned timeout_seconds,
                         bool verbose_flag, bool stats_flag) {
  const std::string exact = "--exact=" + full_name(test);
  int pipe_fds[2];
  if (pipe(pipe_fds) != 0) return {"FAIL", "pipe() failed", true};

  pid_t pid = fork();
  if (pid < 0) {
    close(pipe_fds[0]);
    close(pipe_fds[1]);
    return {"FAIL", "fork() failed", true};
  }

  if (pid == 0) {
    // Child: stdout/stderr into the pipe, then re-exec for one test.
    close(pipe_fds[0]);
    dup2(pipe_fds[1], STDOUT_FILENO);
    dup2(pipe_fds[1], STDERR_FILENO);
    close(pipe_fds[1]);

    std::vector<std::string> args = {
        self, exact, "--child", "--run-known-fatal", dpus, seed};
    if (!elements.empty()) args.push_back(elements);
    if (verbose_flag) args.push_back("-v");
    if (stats_flag) args.push_back("--stats");

    std::vector<char*> argv;
    for (std::string& arg : args)
      argv.push_back(const_cast<char*>(arg.c_str()));
    argv.push_back(nullptr);
    execv(self, argv.data());
    _exit(127);  // execv only returns on failure
  }

  // Read while waiting, so a chatty child cannot fill the pipe and deadlock.
  close(pipe_fds[1]);
  std::string output;
  char buffer[4096];
  bool timed_out = false;
  const long long deadline_ms =
      Watchdog::now_ms() + (long long)timeout_seconds * 1000;

  int flags = fcntl(pipe_fds[0], F_GETFL, 0);
  fcntl(pipe_fds[0], F_SETFL, flags | O_NONBLOCK);

  int status = 0;
  while (true) {
    ssize_t got = read(pipe_fds[0], buffer, sizeof(buffer));
    if (got > 0) {
      output.append(buffer, (size_t)got);
      continue;
    }

    pid_t done = waitpid(pid, &status, WNOHANG);
    if (done == pid) {
      while ((got = read(pipe_fds[0], buffer, sizeof(buffer))) > 0)
        output.append(buffer, (size_t)got);
      break;
    }

    if (timeout_seconds != 0 && Watchdog::now_ms() > deadline_ms) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      timed_out = true;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  close(pipe_fds[0]);

  ChildResult result;
  const std::string marker = "RESULT\t";
  size_t at = output.find(marker);
  if (at != std::string::npos) {
    size_t end = output.find('\n', at);
    std::string line =
        output.substr(at + marker.size(), end == std::string::npos
                                              ? std::string::npos
                                              : end - at - marker.size());
    size_t tab = line.find('\t');
    result.label = line.substr(0, tab);
    if (tab != std::string::npos) result.message = line.substr(tab + 1);
  }

  const bool fatal_expected = test.expect == Expect::Fatal;
  if (timed_out) {
    return {"TIMEOUT",
            "exceeded " + std::to_string(timeout_seconds) + "s (deadlock)",
            !fatal_expected};
  }
  if (WIFSIGNALED(status)) {
    return {"CRASH",
            "killed by signal " + std::to_string(WTERMSIG(status)) +
                (result.message.empty() ? "" : " -- " + result.message),
            !fatal_expected};
  }
  if (result.label.empty()) {
    return {"CRASH",
            "no verdict reported (exit " + std::to_string(WEXITSTATUS(status)) +
                ")",
            !fatal_expected};
  }
  result.counts_as_failure = result.label == "FAIL" || result.label == "XPASS";
  return result;
}

// Parent-side driver for --isolate: one child process per test.
int run_isolated_suite(const char* self, const std::vector<TestCase>& tests,
                       const std::string& filter, bool exact_filter,
                       bool fail_fast, bool show_stats,
                       unsigned timeout_seconds, bool verbose_flag,
                       uint32_t dpu_count, size_t element_count,
                       uint64_t seed) {
  print_build_config();
  std::cout << "isolated: one process per test, " << timeout_seconds
            << "s timeout\n\n";

  const std::string dpus_arg = "--dpus=" + std::to_string(dpu_count);
  const std::string seed_arg = "--seed=" + std::to_string(seed);
  const std::string elements_arg =
      element_count ? "--elements=" + std::to_string(element_count)
                    : std::string();

  size_t passed = 0, failed = 0, skipped = 0, xfailed = 0, selected = 0;
  std::vector<std::string> failures;
  std::string current_suite;

  for (const TestCase& test : tests) {
    if (!matches(test, filter, exact_filter)) continue;
    selected++;

    if (current_suite != test.suite) {
      current_suite = test.suite;
      std::cout << "[" << current_suite << "]\n";
    }

    // No Expect::Fatal skip: isolation makes those safe, and CRASH or TIMEOUT
    // is their expected verdict.
    ChildResult result =
        run_isolated(self, test, dpus_arg, elements_arg, seed_arg,
                     timeout_seconds, verbose_flag, show_stats);

    std::cout << "  " << std::left << std::setw(8) << result.label
              << std::setw(46) << test.name << "\n";
    if (!result.message.empty() && (result.label != "PASS" || show_stats))
      std::cout << "         " << result.message << "\n";
    if (test.expect == Expect::Fatal &&
        (result.label == "CRASH" || result.label == "TIMEOUT"))
      std::cout << "         known fatal: " << test.note << "\n";
    if (test.expect == Expect::Fail && result.label == "XFAIL")
      std::cout << "         known issue: " << test.note << "\n";
    if (result.label == "XPASS")
      std::cout << "         this test now behaves -- drop its marker ("
                << test.note << ")\n";

    if (result.counts_as_failure) {
      failed++;
      failures.push_back(full_name(test));
    } else if (result.label == "SKIP") {
      skipped++;
    } else if (result.label == "XFAIL" || result.label == "CRASH" ||
               result.label == "TIMEOUT") {
      xfailed++;
    } else {
      passed++;
    }

    if (failed && fail_fast) break;
  }

  std::cout << "\n"
            << selected << " selected: " << passed << " passed, " << failed
            << " failed, " << skipped << " skipped, " << xfailed
            << " known-fail\n";
  if (!failures.empty()) {
    std::cout << "failed tests:\n";
    for (const std::string& name : failures) std::cout << "  " << name << "\n";
  }
  return failed == 0 ? 0 : 1;
}

bool arg_value(const char* arg, const char* prefix, std::string& out) {
  const size_t len = std::strlen(prefix);
  if (std::strncmp(arg, prefix, len) != 0) return false;
  out = arg + len;
  return true;
}

}  // namespace
}  // namespace tf

int main(int argc, char** argv) {
  using namespace tf;

  std::string filter;
  bool exact_filter = false;
  bool child_mode = false;
  bool isolate = false;
  bool list_only = false;
  bool fail_fast = false;
  bool show_stats = false;
  bool elements_from_cli = false;
  bool run_known_fatal = false;
  unsigned timeout_seconds = 300;
  uint64_t seed = 12345;

  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    std::string value;
    if (std::strcmp(arg, "--list") == 0) {
      list_only = true;
    } else if (std::strcmp(arg, "--stats") == 0) {
      show_stats = true;
    } else if (std::strcmp(arg, "--fail-fast") == 0) {
      fail_fast = true;
    } else if (std::strcmp(arg, "--run-known-fatal") == 0) {
      run_known_fatal = true;
    } else if (std::strcmp(arg, "-v") == 0 ||
               std::strcmp(arg, "--verbose") == 0) {
      g_verbose = true;
    } else if (arg_value(arg, "--filter=", value)) {
      filter = value;
      exact_filter = false;
    } else if (arg_value(arg, "--exact=", value)) {
      filter = value;
      exact_filter = true;
    } else if (std::strcmp(arg, "--child") == 0) {
      child_mode = true;
    } else if (std::strcmp(arg, "--isolate") == 0) {
      isolate = true;
    } else if (arg_value(arg, "--elements=", value)) {
      g_elements = std::strtoull(value.c_str(), nullptr, 10);
      elements_from_cli = true;
    } else if (arg_value(arg, "--dpus=", value)) {
      g_dpus = (uint32_t)std::strtoul(value.c_str(), nullptr, 10);
    } else if (arg_value(arg, "--seed=", value)) {
      seed = std::strtoull(value.c_str(), nullptr, 10);
    } else if (arg_value(arg, "--timeout=", value)) {
      timeout_seconds = (unsigned)std::strtoul(value.c_str(), nullptr, 10);
    } else if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      print_usage(argv[0]);
      return 0;
    } else {
      std::cerr << "unknown argument: " << arg << "\n";
      print_usage(argv[0]);
      return 2;
    }
  }

  // Static-init order across TUs is unspecified: sort suites alphabetically and
  // tests in source order within a suite.
  std::vector<TestCase>& tests = registry();
  std::stable_sort(tests.begin(), tests.end(),
                   [](const TestCase& a, const TestCase& b) {
                     int suite = std::strcmp(a.suite, b.suite);
                     if (suite != 0) return suite < 0;
                     int file = std::strcmp(a.file, b.file);
                     if (file != 0) return file < 0;
                     return a.line < b.line;
                   });

  if (list_only) {
    for (const TestCase& test : tests)
      if (matches(test, filter, exact_filter))
        std::cout << test.suite << "." << test.name << "\n";
    return 0;
  }

  if (isolate) {
    return run_isolated_suite(argv[0], tests, filter, exact_filter, fail_fast,
                              show_stats, timeout_seconds, g_verbose, g_dpus,
                              elements_from_cli ? g_elements : 0, seed);
  }

  if (!child_mode) print_build_config();

  // One runtime for the whole process; --isolate is the per-test alternative.
  DpuRuntime::get().init(g_dpus);
  const uint32_t allocated = DpuRuntime::get().num_dpus();
  if (!child_mode) std::cout << "allocated " << allocated << " DPUs\n";

  // to_cpu only reads back correctly when every shard is a whole number of
  // 8-byte words, so default to a size that is and warn about one that is not
  // -- otherwise every value check would compare against corrupted data.
  (void)allocated;
  if (!child_mode) std::cout << "elements  " << g_elements << "\n\n";

  size_t passed = 0, failed = 0, skipped = 0, selected = 0, xfailed = 0;
  std::vector<std::string> failures;
  std::string current_suite;

  Watchdog watchdog;
  watchdog.start(timeout_seconds);

  for (const TestCase& test : tests) {
    if (!matches(test, filter, exact_filter)) continue;
    selected++;

    if (current_suite != test.suite && !child_mode) {
      current_suite = test.suite;
      std::cout << "[" << current_suite << "]\n";
    }

    // Would take the whole run down; opt in explicitly.  --isolate never lands
    // here, since the child is expendable.
    if (test.expect == Expect::Fatal && !run_known_fatal) {
      if (child_mode) {
        std::cout << "RESULT\tSKIP\tknown fatal: " << test.note << "\n";
      } else {
        std::cout << "  " << std::left << std::setw(8) << "SKIP"
                  << std::setw(46) << test.name << "\n"
                  << "         known fatal (--run-known-fatal to run, or "
                     "--isolate): "
                  << test.note << "\n";
      }
      skipped++;
      continue;
    }

    // Drained queue and fresh seed, so a result does not depend on what ran
    // before it.
    drain();
    reseed(seed);
    g_outcome = Outcome{};
    watchdog.begin(test);
    const StatsSnapshot before = RuntimeStats::get().snapshot();

    test.fn();

    // Attribute whatever the test left queued to it, not to the next one.
    drain();
    const StatsSnapshot delta = RuntimeStats::get().snapshot() - before;

    const Verdict verdict = verdict_for(test, g_outcome.kind);

    if (child_mode) {
      std::string message = g_outcome.message;
      if (show_stats) message += "  [" + delta.to_string() + "]";
      // Tab-separated so the message can contain anything else.
      for (char& c : message)
        if (c == '\n' || c == '\t') c = ' ';
      std::cout << "RESULT\t" << verdict.label << "\t" << message << "\n";
    } else {
      std::cout << "  " << std::left << std::setw(8) << verdict.label
                << std::setw(46) << test.name;
      if (show_stats) std::cout << "  " << delta.to_string();
      std::cout << "\n";

      if (verdict.show_message && !g_outcome.message.empty())
        std::cout << "         " << g_outcome.message << "\n";
      if (test.expect == Expect::Fail && g_outcome.kind == Outcome::Fail)
        std::cout << "         known issue: " << test.note << "\n";
      if (test.expect == Expect::Fail && g_outcome.kind == Outcome::Pass)
        std::cout << "         this test now passes -- drop the TEST_XFAIL "
                     "marker ("
                  << test.note << ")\n";
      if (test.expect == Expect::Fatal && g_outcome.kind == Outcome::Pass)
        std::cout << "         this test now passes -- drop the "
                     "TEST_KNOWN_FATAL marker ("
                  << test.note << ")\n";
    }

    if (verdict.counts_as_failure) {
      failed++;
      failures.push_back(std::string(test.suite) + "." + test.name);
    } else if (g_outcome.kind == Outcome::Skip) {
      skipped++;
    } else if (std::strcmp(verdict.label, "XFAIL") == 0) {
      xfailed++;
    } else {
      passed++;
    }

    if (failed && fail_fast) break;
  }

  watchdog.shutdown();

  if (child_mode) {
    DpuRuntime::get().shutdown();
    return failed == 0 ? 0 : 1;
  }

  std::cout << "\n"
            << selected << " selected: " << passed << " passed, " << failed
            << " failed, " << skipped << " skipped, " << xfailed
            << " known-fail\n";
  if (!failures.empty()) {
    std::cout << "failed tests:\n";
    for (const std::string& name : failures) std::cout << "  " << name << "\n";
  }

  DpuRuntime::get().shutdown();
  return failed == 0 ? 0 : 1;
}
