// Baseline: the hand-written loops this kernel replaces.
//
// Transcribed from the recurring scratch-program family under the fleet's
// work trees (f5_components.cpp, p7_sample3.cpp, henon_components.cpp, ...):
// a powers table, encode/decode between mixed-radix indices and digit
// tuples, the negation-pair representative, and projective (scalar-class)
// labeling by marking every scalar multiple at each class seed.
//
// Usage: base_p_hand_written <prime> <width> <repeats>
// Prints one JSON line: {"prime":..,"width":..,"repeats":..,
//                        "decode_seconds":..,"negation_seconds":..,
//                        "classes_seconds":..,"class_count":..}

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point started) {
  return std::chrono::duration<double>(Clock::now() - started).count();
}

int main(int argc, char** argv) {
  const int prime = argc > 1 ? std::atoi(argv[1]) : 5;
  const int width = argc > 2 ? std::atoi(argv[2]) : 6;
  const int repeats = argc > 3 ? std::atoi(argv[3]) : 5;

  long long space = 1;
  std::vector<int> powers(width, 1);
  for (int dimension = 1; dimension < width; ++dimension) {
    powers[dimension] = powers[dimension - 1] * prime;
  }
  for (int dimension = 0; dimension < width; ++dimension) {
    space *= prime;
  }

  double decode_seconds = 0.0;
  double negation_seconds = 0.0;
  double classes_seconds = 0.0;
  std::size_t class_count = 0;

  std::vector<int> digits(static_cast<std::size_t>(space) * width);
  std::vector<long long> negated(static_cast<std::size_t>(space));
  std::vector<int> class_ids(static_cast<std::size_t>(space));
  std::vector<long long> representatives;

  for (int repeat = 0; repeat < repeats; ++repeat) {
    // Decode: digit tuples for every index, least significant digit first.
    auto started = Clock::now();
    for (long long index = 0; index < space; ++index) {
      long long remaining = index;
      for (int dimension = 0; dimension < width; ++dimension) {
        digits[index * width + dimension] =
            static_cast<int>(remaining % prime);
        remaining /= prime;
      }
    }
    decode_seconds += seconds_since(started);

    // Negation representative: min(index, encoding of -digits).
    started = Clock::now();
    for (long long index = 0; index < space; ++index) {
      long long negation_encoding = 0;
      for (int dimension = 0; dimension < width; ++dimension) {
        const int digit = digits[index * width + dimension];
        if (digit != 0) {
          negation_encoding +=
              static_cast<long long>(prime - digit) * powers[dimension];
        }
      }
      negated[index] = std::min(negation_encoding, index);
    }
    negation_seconds += seconds_since(started);

    // Scalar-class labeling: at each unseen seed take the canonical minimum
    // over nonzero multiples, then mark every multiple with that id.
    started = Clock::now();
    std::fill(class_ids.begin(), class_ids.end(), -1);
    class_count = 0;
    representatives.clear();
    for (long long index = 1; index < space; ++index) {
      if (class_ids[index] >= 0) {
        continue;
      }
      long long best = -1;
      for (int factor = 1; factor < prime; ++factor) {
        long long candidate = 0;
        for (int dimension = width - 1; dimension >= 0; --dimension) {
          const int scaled =
              digits[index * width + dimension] * factor % prime;
          candidate = candidate * prime + scaled;
        }
        if (best < 0 || candidate < best) {
          best = candidate;
        }
      }
      representatives.push_back(best);
      const int id = static_cast<int>(class_count);
      for (int factor = 1; factor < prime; ++factor) {
        long long multiple = 0;
        long long scale = 1;
        for (int dimension = 0; dimension < width; ++dimension) {
          multiple += static_cast<long long>(
              digits[index * width + dimension] * factor % prime) * scale;
          scale *= prime;
        }
        class_ids[multiple] = id;
      }
      ++class_count;
    }
    class_count += 1;  // the zero vector forms its own class
    classes_seconds += seconds_since(started);
    representatives.clear();
  }

  std::printf(
      "{\"prime\":%d,\"width\":%d,\"repeats\":%d,"
      "\"decode_seconds\":%.9f,\"negation_seconds\":%.9f,"
      "\"classes_seconds\":%.9f,\"class_count\":%zu}\n",
      prime,
      width,
      repeats,
      decode_seconds / repeats,
      negation_seconds / repeats,
      classes_seconds / repeats,
      class_count);
  return 0;
}
