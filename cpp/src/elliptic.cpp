#include "fast_math.h"

#include "parallel.hpp"

#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

void set_error(
    char* destination,
    std::size_t destination_size,
    const char* message) {
  if (destination == nullptr || destination_size == 0) {
    return;
  }
  std::strncpy(destination, message, destination_size - 1);
  destination[destination_size - 1] = '\0';
}

// Lemire fast modular reduction for moduli below 2^16, so that a product of
// two residues stays inside 32 bits and one reduction costs two multiplies.
class Modulus {
 public:
  explicit Modulus(std::uint32_t modulus)
      : modulus_(modulus),
        magic_(~std::uint64_t{0} / modulus + 1) {}

  std::uint32_t value() const { return modulus_; }

  std::uint32_t reduce(std::uint32_t input) const {
    const auto low = magic_ * input;
    return static_cast<std::uint32_t>(
        (static_cast<__uint128_t>(low) * modulus_) >> 64);
  }

  std::uint32_t multiply(std::uint32_t left, std::uint32_t right) const {
    return reduce(left * right);
  }

  std::uint32_t add(std::uint32_t left, std::uint32_t right) const {
    const auto sum = left + right;
    return sum >= modulus_ ? sum - modulus_ : sum;
  }

  std::uint32_t subtract(std::uint32_t left, std::uint32_t right) const {
    return left >= right ? left - right : left + modulus_ - right;
  }

  std::uint32_t power(std::uint32_t base, std::uint32_t exponent) const {
    std::uint32_t result = 1 % modulus_;
    std::uint32_t factor = reduce(base);
    while (exponent != 0) {
      if ((exponent & 1u) != 0) {
        result = multiply(result, factor);
      }
      factor = multiply(factor, factor);
      exponent >>= 1;
    }
    return result;
  }

  std::uint32_t reduce_signed(std::int64_t input) const {
    const auto folded = input % static_cast<std::int64_t>(modulus_);
    return static_cast<std::uint32_t>(
        folded < 0 ? folded + static_cast<std::int64_t>(modulus_) : folded);
  }

 private:
  std::uint32_t modulus_;
  std::uint64_t magic_;
};

std::vector<std::int8_t> quadratic_characters(const Modulus& modulus) {
  const auto prime = modulus.value();
  std::vector<std::int8_t> characters(prime, -1);
  characters[0] = 0;
  for (std::uint32_t root = 1; root < prime; ++root) {
    characters[modulus.multiply(root, root)] = 1;
  }
  return characters;
}

constexpr std::uint32_t kSextupleSize = 6;
constexpr std::uint32_t kSexticSize = 7;   // p6 has seven coefficients
constexpr std::uint32_t kProductSize = 13; // p6(x-T) p6(x+T) has thirteen

struct MestreFibre {
  std::uint32_t quartic[5];
};

// r(x, t) for one fibre: p6(x - t) p6(x + t) = g(x)^2 - r(x), g monic of
// degree six.  Requires the sextuple to lie on the Mestre locus, which is what
// makes the x^5 coefficient of r vanish; the kernel does not re-check it.
MestreFibre fibre_quartic(
    const Modulus& modulus,
    const std::uint32_t sextic[kSexticSize],
    const std::uint32_t binomials[kSexticSize][kSexticSize],
    std::uint32_t parameter) {
  const auto prime = modulus.value();

  std::uint32_t powers[kSexticSize];
  powers[0] = 1 % prime;
  for (std::uint32_t index = 1; index < kSexticSize; ++index) {
    powers[index] = modulus.multiply(powers[index - 1], parameter);
  }

  std::uint32_t left[kSexticSize] = {};
  std::uint32_t right[kSexticSize] = {};
  for (std::uint32_t degree = 0; degree < kSexticSize; ++degree) {
    if (sextic[degree] == 0) {
      continue;
    }
    for (std::uint32_t taken = 0; taken <= degree; ++taken) {
      const auto weight = modulus.multiply(
          sextic[degree],
          modulus.multiply(binomials[degree][taken], powers[taken]));
      const auto target = degree - taken;
      right[target] = modulus.add(right[target], weight);
      left[target] = (taken & 1u) == 0
          ? modulus.add(left[target], weight)
          : modulus.subtract(left[target], weight);
    }
  }

  std::uint32_t product[kProductSize] = {};
  for (std::uint32_t i = 0; i < kSexticSize; ++i) {
    if (left[i] == 0) {
      continue;
    }
    for (std::uint32_t j = 0; j < kSexticSize; ++j) {
      product[i + j] =
          modulus.add(product[i + j], modulus.multiply(left[i], right[j]));
    }
  }

  const auto half = modulus.power(2, prime - 2);
  std::uint32_t root[kSexticSize] = {};
  root[6] = 1 % prime;
  for (std::int32_t degree = 5; degree >= 0; --degree) {
    std::uint32_t accumulated = 0;
    for (std::uint32_t i = static_cast<std::uint32_t>(degree) + 1;
         i < kSexticSize;
         ++i) {
      const auto j = static_cast<std::uint32_t>(degree) + 6 - i;
      if (j < kSexticSize) {
        accumulated =
            modulus.add(accumulated, modulus.multiply(root[i], root[j]));
      }
    }
    root[degree] = modulus.multiply(
        modulus.subtract(product[degree + 6], accumulated), half);
  }

  MestreFibre fibre{};
  for (std::uint32_t degree = 0; degree < 5; ++degree) {
    std::uint32_t accumulated = 0;
    for (std::uint32_t i = 0; i <= degree; ++i) {
      accumulated =
          modulus.add(accumulated, modulus.multiply(root[i], root[degree - i]));
    }
    fibre.quartic[degree] = modulus.subtract(accumulated, product[degree]);
  }
  return fibre;
}

// a_p of the fibre's Jacobian.  The quartic and its Jacobian share a_p at good
// primes, but the quartic model degenerates modulo p far more often than the
// curve does, and a degenerate quartic returns a character sum of size p rather
// than a trace bounded by 2 sqrt(p).  Scoring the Jacobian cubic
// Y^2 = X^3 - 27 I X - 27 J instead keeps every table entry an honest trace and
// leaves only the genuinely singular fibres, which report zero.
std::int32_t fibre_trace(
    const Modulus& modulus,
    const std::int8_t* characters,
    const std::uint32_t quartic[5]) {
  const auto prime = modulus.value();
  const auto r0 = quartic[0];
  const auto r1 = quartic[1];
  const auto r2 = quartic[2];
  const auto r3 = quartic[3];
  const auto r4 = quartic[4];

  const auto invariant_i = modulus.add(
      modulus.subtract(
          modulus.multiply(modulus.reduce(12), modulus.multiply(r4, r0)),
          modulus.multiply(modulus.reduce(3), modulus.multiply(r3, r1))),
      modulus.multiply(r2, r2));
  std::uint32_t invariant_j = modulus.multiply(
      modulus.reduce(72), modulus.multiply(r4, modulus.multiply(r2, r0)));
  invariant_j = modulus.add(
      invariant_j,
      modulus.multiply(
          modulus.reduce(9), modulus.multiply(r3, modulus.multiply(r2, r1))));
  invariant_j = modulus.subtract(
      invariant_j,
      modulus.multiply(
          modulus.reduce(27), modulus.multiply(r4, modulus.multiply(r1, r1))));
  invariant_j = modulus.subtract(
      invariant_j,
      modulus.multiply(
          modulus.reduce(27), modulus.multiply(r3, modulus.multiply(r3, r0))));
  invariant_j = modulus.subtract(
      invariant_j,
      modulus.multiply(
          modulus.reduce(2), modulus.multiply(r2, modulus.multiply(r2, r2))));

  // The cubic's discriminant is proportional to 4 I^3 - J^2; a zero means the
  // model is singular modulo p, so the fibre contributes nothing.
  const auto cube = modulus.multiply(
      invariant_i, modulus.multiply(invariant_i, invariant_i));
  const auto discriminant = modulus.subtract(
      modulus.multiply(modulus.reduce(4), cube),
      modulus.multiply(invariant_j, invariant_j));
  if (discriminant == 0) {
    return 0;
  }

  const auto linear =
      modulus.subtract(0, modulus.multiply(modulus.reduce(27), invariant_i));
  const auto constant =
      modulus.subtract(0, modulus.multiply(modulus.reduce(27), invariant_j));
  std::int32_t total = 0;
  for (std::uint32_t point = 0; point < prime; ++point) {
    auto value = modulus.multiply(point, modulus.multiply(point, point));
    value = modulus.add(value, modulus.multiply(linear, point));
    value = modulus.add(value, constant);
    total += characters[value];
  }
  return -total;
}

}  // namespace

extern "C" {

int fast_math_elliptic_mestre_ap_tables_i32(
    const std::int64_t* sextuple,
    const std::uint32_t* primes,
    std::size_t prime_count,
    const std::uint64_t* table_offsets,
    std::uint32_t thread_count,
    std::int32_t* tables,
    fast_math_elliptic_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    if (prime_count != 0 &&
        (sextuple == nullptr || primes == nullptr ||
         table_offsets == nullptr || tables == nullptr)) {
      throw std::invalid_argument("sextuple, prime, or table pointer is null");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    stats->prime_count = prime_count;
    const auto started = Clock::now();

    for (std::size_t index = 0; index < prime_count; ++index) {
      const auto prime = primes[index];
      if (prime < 3 || prime >= (1u << 16)) {
        throw std::invalid_argument(
            "primes must be odd and below 65536");
      }
      if (table_offsets[index + 1] - table_offsets[index] != prime) {
        throw std::invalid_argument(
            "table_offsets must be the prefix sums of the primes");
      }
      stats->parameter_count += prime;
    }

    std::uint32_t binomials[kSexticSize][kSexticSize] = {};
    for (std::uint32_t upper = 0; upper < kSexticSize; ++upper) {
      binomials[upper][0] = 1;
      for (std::uint32_t lower = 1; lower <= upper; ++lower) {
        binomials[upper][lower] = binomials[upper - 1][lower - 1] +
            (lower <= upper - 1 ? binomials[upper - 1][lower] : 0);
      }
    }

    fast_math_internal::parallel_for_dynamic(
        prime_count,
        thread_count,
        [&](std::size_t index) {
          const Modulus modulus(primes[index]);
          const auto prime = modulus.value();
          const auto characters = quadratic_characters(modulus);

          std::uint32_t reduced_binomials[kSexticSize][kSexticSize] = {};
          for (std::uint32_t upper = 0; upper < kSexticSize; ++upper) {
            for (std::uint32_t lower = 0; lower <= upper; ++lower) {
              reduced_binomials[upper][lower] =
                  modulus.reduce(binomials[upper][lower]);
            }
          }

          std::uint32_t sextic[kSexticSize] = {};
          sextic[0] = 1 % prime;
          std::uint32_t degree = 0;
          for (std::uint32_t root = 0; root < kSextupleSize; ++root) {
            const auto negated =
                modulus.subtract(0, modulus.reduce_signed(sextuple[root]));
            ++degree;
            sextic[degree] = sextic[degree - 1];
            for (std::uint32_t back = degree - 1; back > 0; --back) {
              sextic[back] = modulus.add(
                  sextic[back - 1], modulus.multiply(sextic[back], negated));
            }
            sextic[0] = modulus.multiply(sextic[0], negated);
          }

          auto* output = tables + table_offsets[index];
          for (std::uint32_t parameter = 0; parameter < prime; ++parameter) {
            const auto fibre =
                fibre_quartic(modulus, sextic, reduced_binomials, parameter);
            output[parameter] =
                fibre_trace(modulus, characters.data(), fibre.quartic);
          }
        });

    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown native error");
    return 2;
  }
}

int fast_math_elliptic_nagao_scores_f64(
    const std::int32_t* tables,
    const std::uint32_t* primes,
    const double* weights,
    std::size_t prime_count,
    const std::uint64_t* table_offsets,
    const std::int64_t* numerators,
    const std::int64_t* denominators,
    std::size_t parameter_count,
    std::uint32_t thread_count,
    double* scores,
    fast_math_elliptic_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    if (parameter_count != 0 &&
        (numerators == nullptr || denominators == nullptr ||
         scores == nullptr)) {
      throw std::invalid_argument("parameter or score pointer is null");
    }
    if (prime_count != 0 &&
        (tables == nullptr || primes == nullptr || weights == nullptr ||
         table_offsets == nullptr)) {
      throw std::invalid_argument("table or prime pointer is null");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    stats->prime_count = prime_count;
    stats->parameter_count = parameter_count;
    const auto started = Clock::now();

    std::vector<std::uint32_t> inverses(
        prime_count == 0 ? 0 : table_offsets[prime_count]);
    for (std::size_t index = 0; index < prime_count; ++index) {
      const auto prime = primes[index];
      if (prime < 3 || prime >= (1u << 16)) {
        throw std::invalid_argument("primes must be odd and below 65536");
      }
      const Modulus modulus(prime);
      auto* row = inverses.data() + table_offsets[index];
      row[0] = 0;
      for (std::uint32_t residue = 1; residue < prime; ++residue) {
        row[residue] = modulus.power(residue, prime - 2);
      }
    }

    constexpr std::size_t kBlock = 4096;
    const auto block_count = (parameter_count + kBlock - 1) / kBlock;
    fast_math_internal::parallel_for_static(
        block_count,
        thread_count,
        [&](std::size_t block) {
          const auto begin = block * kBlock;
          const auto end =
              std::min(parameter_count, begin + kBlock);
          for (auto index = begin; index < end; ++index) {
            scores[index] = 0.0;
          }
          for (std::size_t prime_index = 0; prime_index < prime_count;
               ++prime_index) {
            const Modulus modulus(primes[prime_index]);
            const auto* table = tables + table_offsets[prime_index];
            const auto* inverse = inverses.data() + table_offsets[prime_index];
            const auto weight = weights[prime_index];
            for (auto index = begin; index < end; ++index) {
              const auto denominator =
                  modulus.reduce_signed(denominators[index]);
              if (denominator == 0) {
                continue;
              }
              const auto numerator = modulus.reduce_signed(numerators[index]);
              const auto parameter =
                  modulus.multiply(numerator, inverse[denominator]);
              scores[index] -= weight * table[parameter];
            }
          }
        });

    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown native error");
    return 2;
  }
}

int fast_math_elliptic_quartic_sieve_i64(
    const std::uint32_t* coefficient_residues,
    const std::uint32_t* primes,
    std::size_t prime_count,
    std::int64_t numerator_low,
    std::int64_t numerator_high,
    std::int64_t denominator_low,
    std::int64_t denominator_high,
    std::uint32_t thread_count,
    std::int64_t* candidates,
    std::size_t candidate_capacity,
    fast_math_elliptic_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    if (prime_count == 0) {
      throw std::invalid_argument("at least one sieve prime is required");
    }
    if (coefficient_residues == nullptr || primes == nullptr) {
      throw std::invalid_argument("coefficient or prime pointer is null");
    }
    if (numerator_high < numerator_low || denominator_high < denominator_low) {
      throw std::invalid_argument("search ranges must be nondecreasing");
    }
    if (denominator_low < 1) {
      throw std::invalid_argument("denominators must be positive");
    }
    set_error(error_message, error_message_size, "");
    *stats = {};
    stats->prime_count = prime_count;
    const auto started = Clock::now();

    const auto width = static_cast<std::uint64_t>(
        numerator_high - numerator_low + 1);
    const auto words = (width + 63) / 64;
    const auto denominator_count = static_cast<std::size_t>(
        denominator_high - denominator_low + 1);
    stats->parameter_count = denominator_count * width;

    // patterns[prime][w mod p] is the acceptance bitmap of the numerator range
    // for that denominator class, laid out as p words: the pattern repeats with
    // period p bits, and p odd words cover every phase 64j mod p can take.
    std::vector<std::vector<std::uint64_t>> patterns(prime_count);
    std::vector<Modulus> moduli;
    moduli.reserve(prime_count);
    for (std::size_t index = 0; index < prime_count; ++index) {
      const auto prime = primes[index];
      if (prime < 3 || prime > 4096) {
        throw std::invalid_argument(
            "sieve primes must be odd and at most 4096");
      }
      const Modulus modulus(prime);
      moduli.push_back(modulus);
      const auto characters = quadratic_characters(modulus);
      const auto* coefficients = coefficient_residues + 5 * index;

      std::vector<std::uint8_t> accepted(
          static_cast<std::size_t>(prime) * prime, 0);
      for (std::uint32_t denominator = 0; denominator < prime; ++denominator) {
        std::uint32_t denominator_powers[5];
        denominator_powers[0] = 1 % prime;
        for (std::uint32_t power = 1; power < 5; ++power) {
          denominator_powers[power] =
              modulus.multiply(denominator_powers[power - 1], denominator);
        }
        auto* row =
            accepted.data() + static_cast<std::size_t>(denominator) * prime;
        for (std::uint32_t numerator = 0; numerator < prime; ++numerator) {
          std::uint32_t value = 0;
          std::uint32_t numerator_power = 1 % prime;
          for (std::uint32_t degree = 0; degree < 5; ++degree) {
            const auto term = modulus.multiply(
                modulus.reduce(coefficients[degree]),
                modulus.multiply(
                    numerator_power, denominator_powers[4 - degree]));
            value = modulus.add(value, term);
            numerator_power = modulus.multiply(numerator_power, numerator);
          }
          row[numerator] = characters[value] >= 0 ? 1 : 0;
        }
      }

      auto& pattern = patterns[index];
      pattern.assign(static_cast<std::size_t>(prime) * prime, 0);
      for (std::uint32_t denominator = 0; denominator < prime; ++denominator) {
        const auto* row =
            accepted.data() + static_cast<std::size_t>(denominator) * prime;
        auto* words_out =
            pattern.data() + static_cast<std::size_t>(denominator) * prime;
        for (std::uint32_t word = 0; word < prime; ++word) {
          std::uint64_t bits = 0;
          const auto base =
              numerator_low + static_cast<std::int64_t>(word) * 64;
          auto residue = modulus.reduce_signed(base);
          for (std::uint32_t bit = 0; bit < 64; ++bit) {
            if (row[residue] != 0) {
              bits |= std::uint64_t{1} << bit;
            }
            ++residue;
            if (residue == prime) {
              residue = 0;
            }
          }
          words_out[word] = bits;
        }
      }
    }

    std::vector<std::vector<std::int64_t>> found(denominator_count);
    fast_math_internal::parallel_for_dynamic(
        denominator_count,
        thread_count,
        [&](std::size_t index) {
          const auto denominator =
              denominator_low + static_cast<std::int64_t>(index);
          std::vector<std::uint64_t> sieve(words, ~std::uint64_t{0});
          for (std::size_t prime_index = 0; prime_index < prime_count;
               ++prime_index) {
            const auto& modulus = moduli[prime_index];
            const auto prime = modulus.value();
            const auto* pattern = patterns[prime_index].data() +
                static_cast<std::size_t>(
                    modulus.reduce_signed(denominator)) *
                    prime;
            std::uint32_t phase = 0;
            for (std::uint64_t word = 0; word < words; ++word) {
              sieve[word] &= pattern[phase];
              ++phase;
              if (phase == prime) {
                phase = 0;
              }
            }
          }

          auto& local = found[index];
          for (std::uint64_t word = 0; word < words; ++word) {
            auto bits = sieve[word];
            while (bits != 0) {
              const auto bit = static_cast<std::uint32_t>(
                  __builtin_ctzll(bits));
              bits &= bits - 1;
              const auto offset = word * 64 + bit;
              if (offset >= width) {
                break;
              }
              local.push_back(
                  numerator_low + static_cast<std::int64_t>(offset));
              local.push_back(denominator);
            }
          }
        });

    std::size_t written = 0;
    for (const auto& local : found) {
      for (std::size_t index = 0; index + 1 < local.size(); index += 2) {
        if (written < candidate_capacity) {
          candidates[2 * written] = local[index];
          candidates[2 * written + 1] = local[index + 1];
        }
        ++written;
      }
    }
    stats->candidate_count = written;
    stats->truncated = written > candidate_capacity ? 1 : 0;
    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown native error");
    return 2;
  }
}

}
