// Shift-divisor gate scan: wheel-compressed segmented sieve over linear
// forms A*v - B followed by deterministic Miller-Rabin on the surviving
// candidates.  The contract (documented in python/fast_math/shift_gates.py)
// is exact: survivors are the v in range whose class is alive in every
// lookup table and whose every form has a modulus-coprime part that is
// prime or the square of a prime.  The sieve is a pure optimization; the
// caller asserts the size precondition that keeps it sound.

#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <vector>

namespace {

using u64 = std::uint64_t;
using u32 = std::uint32_t;
using u128 = unsigned __int128;

constexpr std::size_t kChunkBits = std::size_t(1) << 22;  // 512 KiB bitmap
constexpr u64 kMrBases[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};

void set_error(char* destination, std::size_t size, const char* message) {
  if (destination == nullptr || size == 0) return;
  std::strncpy(destination, message, size - 1);
  destination[size - 1] = '\0';
}

u64 mulmod_u64(u64 a, u64 b, u64 m) { return u64((u128(a) * b) % m); }

u64 powmod_u64(u64 base, u64 exponent, u64 m) {
  u64 result = 1 % m;
  base %= m;
  while (exponent != 0) {
    if (exponent & 1) result = mulmod_u64(result, base, m);
    base = mulmod_u64(base, base, m);
    exponent >>= 1;
  }
  return result;
}

u32 invmod_u32(u64 value, u32 prime) {  // prime modulus
  return u32(powmod_u64(value % prime, prime - 2, prime));
}

// --- Montgomery arithmetic, single 64-bit word -----------------------------

struct Mont64 {
  u64 n, ninv, r2;  // ninv = -n^{-1} mod 2^64, r2 = 2^128 mod n
};

Mont64 mont64_create(u64 n) {
  u64 inv = n;  // Newton: converges to n^{-1} mod 2^64
  for (int i = 0; i < 5; ++i) inv *= 2 - n * inv;
  u64 r = (~u64(0)) % n + 1;  // 2^64 mod n
  return Mont64{n, ~inv + 1, u64((u128(r) * r) % n)};
}

u64 mont64_mul(const Mont64& m, u64 a, u64 b) {
  u128 t = u128(a) * b;
  u64 low = u64(t) * m.ninv;
  u128 s = t + u128(low) * m.n;
  u64 result = u64(s >> 64);
  return result >= m.n ? result - m.n : result;
}

bool mr_prime_u64(u64 n) {
  if (n < 2) return false;
  for (u64 p : kMrBases) {
    if (n % p == 0) return n == p;
  }
  Mont64 m = mont64_create(n);
  u64 d = n - 1;
  int s = 0;
  while ((d & 1) == 0) {
    d >>= 1;
    ++s;
  }
  const u64 one = mont64_mul(m, 1, m.r2);
  const u64 minus_one = n - one;
  for (u64 base : kMrBases) {
    u64 x = mont64_mul(m, base % n, m.r2);
    u64 y = one;
    u64 e = d;
    while (e != 0) {  // x^d in Montgomery form
      if (e & 1) y = mont64_mul(m, y, x);
      x = mont64_mul(m, x, x);
      e >>= 1;
    }
    if (y == one || y == minus_one) continue;
    bool witness = true;
    for (int i = 1; i < s; ++i) {
      y = mont64_mul(m, y, y);
      if (y == minus_one) {
        witness = false;
        break;
      }
    }
    if (witness) return false;
  }
  return true;
}

// --- Montgomery arithmetic, two 64-bit words (values below 2^127) ----------

struct Mont128 {
  u128 n;
  u64 ninv;  // -n^{-1} mod 2^64
  u128 r2;   // 2^256 mod n
};

u128 mod_double(u128 x, u128 n) {  // (2x) mod n for x < n
  u128 y = x << 1;
  return (y >= n || y < x) ? y - n : y;
}

u128 pow2_mod(unsigned bits, u128 n) {  // 2^bits mod n
  u128 x = 1 % n;
  for (unsigned i = 0; i < bits; ++i) x = mod_double(x, n);
  return x;
}

Mont128 mont128_create(u128 n) {
  u64 n0 = u64(n);
  u64 inv = n0;
  for (int i = 0; i < 5; ++i) inv *= 2 - n0 * inv;
  return Mont128{n, ~inv + 1, pow2_mod(256, n)};
}

u128 mont128_mul(const Mont128& m, u128 a, u128 b) {
  // Operand-scanning CIOS with 64-bit limbs; a, b < n < 2^127.
  u64 a0 = u64(a), a1 = u64(a >> 64);
  u64 b0 = u64(b), b1 = u64(b >> 64);
  u64 n0 = u64(m.n), n1 = u64(m.n >> 64);
  u64 t0 = 0, t1 = 0, t2 = 0, t3 = 0;
  auto round = [&](u64 ai) {
    u128 c = u128(ai) * b0 + t0;
    u64 s0 = u64(c);
    c = u128(ai) * b1 + t1 + u64(c >> 64);
    u64 s1 = u64(c);
    u128 c2 = u128(t2) + u64(c >> 64);
    u64 s2 = u64(c2);
    u64 s3 = t3 + u64(c2 >> 64);
    u64 q = s0 * m.ninv;
    c = u128(q) * n0 + s0;
    c = u128(q) * n1 + s1 + u64(c >> 64);
    t0 = u64(c);
    c2 = u128(s2) + u64(c >> 64);
    t1 = u64(c2);
    t2 = s3 + u64(c2 >> 64);
    t3 = 0;
  };
  round(a0);
  round(a1);
  // Two more reduction-only rounds are unnecessary: after two rounds the
  // accumulator is t < 2n for n < 2^127.
  u128 t = (u128(t1) << 64) | t0;
  if (t2 != 0 || t >= m.n) t -= m.n;
  return t;
}

bool mr_prime_u128(u128 n) {
  if ((n >> 64) == 0) return mr_prime_u64(u64(n));
  for (u64 p : kMrBases) {
    if (n % p == 0) return false;  // n > 2^64 cannot equal a base
  }
  Mont128 m = mont128_create(n);
  u128 d = n - 1;
  int s = 0;
  while ((d & 1) == 0) {
    d >>= 1;
    ++s;
  }
  const u128 one = mont128_mul(m, 1, m.r2);
  const u128 minus_one = n - one;
  for (u64 base : kMrBases) {
    u128 x = mont128_mul(m, base, m.r2);
    u128 y = one;
    u128 e = d;
    while (e != 0) {
      if (e & 1) y = mont128_mul(m, y, x);
      x = mont128_mul(m, x, x);
      e >>= 1;
    }
    if (y == one || y == minus_one) continue;
    bool witness = true;
    for (int i = 1; i < s; ++i) {
      y = mont128_mul(m, y, y);
      if (y == minus_one) {
        witness = false;
        break;
      }
    }
    if (witness) return false;
  }
  return true;
}

u64 isqrt_u128(u128 n) {
  if (n == 0) return 0;
  double approx = double(u64(n >> 64)) * 18446744073709551616.0 + double(u64(n));
  u64 root = u64(__builtin_sqrt(approx));
  for (int i = 0; i < 4; ++i) {  // Newton, exact for n < 2^127
    u64 next = u64((n / root + root) / 2);
    if (next >= root - 1 && next <= root + 1 && u128(next) * next <= n &&
        (u128(next) + 1) * (u128(next) + 1) > n)
      return next;
    root = next;
  }
  while (u128(root) * root > n) --root;
  while ((u128(root) + 1) * (u128(root) + 1) <= n) ++root;
  return root;
}

bool prime_or_prime_square(u128 part) {
  if (part <= 1) return part == 1;
  if (mr_prime_u128(part)) return true;
  u64 root = isqrt_u128(part);
  return u128(root) * root == part && mr_prime_u64(root);
}

std::vector<u32> primes_between(u32 low, u32 high) {
  std::vector<bool> composite(high + 1, false);
  std::vector<u32> out;
  for (u32 p = 2; p <= high; ++p) {
    if (composite[p]) continue;
    if (p >= low) out.push_back(p);
    for (u64 q = u64(p) * p; q <= high; q += p) composite[q] = true;
  }
  return out;
}

struct GateSpec {
  const u64* form_a;
  const u64* form_b;
  std::size_t form_count;
  const u32* smooth_primes;
  std::size_t smooth_prime_count;
  const u64* lut_moduli;
  const u64* lut_offsets;
  std::size_t lut_count;
  const u64* lut_bits;
  u64 wheel;
  const u64* wheel_classes;
  std::size_t wheel_class_count;
  u64 v_start;
  u64 v_count;
};

bool lut_alive(const GateSpec& gate, u64 v) {
  for (std::size_t i = 0; i < gate.lut_count; ++i) {
    u64 index = gate.lut_offsets[i] + v % gate.lut_moduli[i];
    if (((gate.lut_bits[index >> 6] >> (index & 63)) & 1) == 0) return false;
  }
  return true;
}

bool forms_alive(const GateSpec& gate, u64 v) {
  for (std::size_t i = 0; i < gate.form_count; ++i) {
    u128 value = u128(gate.form_a[i]) * v - gate.form_b[i];
    for (std::size_t s = 0; s < gate.smooth_prime_count; ++s) {
      u32 p = gate.smooth_primes[s];
      while (value % p == 0) value /= p;
    }
    if (!prime_or_prime_square(value)) return false;
  }
  return true;
}

}  // namespace

extern "C" {

FAST_MATH_API int fast_math_shift_gate_scan_u64(
    const std::uint64_t* form_a,
    const std::uint64_t* form_b,
    std::size_t form_count,
    const std::uint32_t* smooth_primes,
    std::size_t smooth_prime_count,
    const std::uint64_t* lut_moduli,
    const std::uint64_t* lut_offsets,
    std::size_t lut_count,
    const std::uint64_t* lut_bits,
    std::uint64_t wheel,
    const std::uint64_t* wheel_classes,
    std::size_t wheel_class_count,
    std::uint32_t sieve_low,
    std::uint32_t sieve_bound,
    std::uint64_t v_start,
    std::uint64_t v_count,
    std::uint32_t thread_count,
    std::uint64_t* survivors,
    std::size_t survivor_capacity,
    std::size_t* survivor_count,
    std::uint64_t* stats,
    char* error_message,
    std::size_t error_capacity) {
  if (form_count == 0 || wheel_class_count == 0 || v_count == 0) {
    set_error(error_message, error_capacity, "empty gate or range");
    return 1;
  }
  const GateSpec gate{form_a,      form_b,        form_count,       smooth_primes,
                      smooth_prime_count, lut_moduli, lut_offsets,  lut_count,
                      lut_bits,    wheel,         wheel_classes,    wheel_class_count,
                      v_start,     v_count};

  const std::vector<u32> primes = primes_between(sieve_low, sieve_bound);
  // Per (prime, form): a mod q, and (a * wheel)^{-1} mod q; a form whose
  // coefficient q divides never vanishes mod q (offsets are coprime) and is
  // skipped via inverse 0.
  std::vector<u32> a_mod(primes.size() * form_count);
  std::vector<u32> inv_aw(primes.size() * form_count);
  for (std::size_t qi = 0; qi < primes.size(); ++qi) {
    const u32 q = primes[qi];
    const u64 w_mod = wheel % q;
    for (std::size_t fi = 0; fi < form_count; ++fi) {
      const u64 a = form_a[fi] % q;
      a_mod[qi * form_count + fi] = u32(a);
      inv_aw[qi * form_count + fi] =
          (a == 0 || w_mod == 0) ? 0 : invmod_u32(mulmod_u64(a, w_mod, q), q);
    }
  }

  const u64 chunk_span = u64(kChunkBits);
  // Rows: one per wheel class; each row covers t with v = class + wheel * t.
  const u64 t_high = (v_start + v_count - 1) / wheel + 1;
  const u64 t_low = v_start / wheel;
  const u64 chunks_per_row = (t_high - t_low + chunk_span - 1) / chunk_span;
  const std::size_t task_count = wheel_class_count * std::size_t(chunks_per_row);

  std::atomic<std::size_t> out_count{0};
  std::atomic<bool> overflow{false};
  std::atomic<u64> stat_sieve{0}, stat_lut{0}, stat_survivors{0};

  fast_math_internal::parallel_for_dynamic(
      task_count, thread_count, [&](std::size_t task) {
        const std::size_t class_index = task / chunks_per_row;
        const u64 chunk_index = u64(task % chunks_per_row);
        const u64 r = wheel_classes[class_index];
        const u64 t_base = t_low + chunk_index * chunk_span;
        if (t_base >= t_high) return;
        const u64 t_len = std::min(chunk_span, t_high - t_base);

        thread_local std::vector<u64> bitmap;
        bitmap.assign((kChunkBits + 63) / 64, ~u64(0));

        for (std::size_t qi = 0; qi < primes.size(); ++qi) {
          const u32 q = primes[qi];
          const u64 r_mod = r % q;
          const u64 t_base_mod = t_base % q;
          for (std::size_t fi = 0; fi < form_count; ++fi) {
            const u32 inv = inv_aw[qi * form_count + fi];
            if (inv == 0) continue;
            // Solve a*(r + wheel*t) = b (mod q):
            // t = (b - a*r) * inv_aw (mod q), then rebase to the chunk.
            const u64 rhs = (form_b[fi] % q + q - mulmod_u64(a_mod[qi * form_count + fi], r_mod, q)) % q;
            u64 t0 = mulmod_u64(rhs, inv, q);
            t0 = (t0 + q - t_base_mod) % q;
            for (u64 i = t0; i < t_len; i += q)
              bitmap[i >> 6] &= ~(u64(1) << (i & 63));
          }
        }

        u64 local_sieve = 0, local_lut = 0, local_surv = 0;
        const std::size_t word_count = (t_len + 63) / 64;
        for (std::size_t w = 0; w < word_count; ++w) {
          u64 word = bitmap[w];
          if (w == word_count - 1 && (t_len & 63) != 0)
            word &= (u64(1) << (t_len & 63)) - 1;
          while (word != 0) {
            const int bit = __builtin_ctzll(word);
            word &= word - 1;
            const u64 t = t_base + (u64(w) << 6) + u64(bit);
            const u64 v = r + wheel * t;
            if (v < v_start || v - v_start >= v_count) continue;
            ++local_sieve;
            if (!lut_alive(gate, v)) continue;
            ++local_lut;
            if (!forms_alive(gate, v)) continue;
            ++local_surv;
            const std::size_t slot = out_count.fetch_add(1);
            if (slot < survivor_capacity) {
              survivors[slot] = v;
            } else {
              overflow.store(true);
            }
          }
        }
        stat_sieve.fetch_add(local_sieve);
        stat_lut.fetch_add(local_lut);
        stat_survivors.fetch_add(local_surv);
      });

  if (overflow.load()) {
    set_error(error_message, error_capacity, "survivor capacity exceeded");
    return 2;
  }
  const std::size_t found = std::min(out_count.load(), survivor_capacity);
  std::sort(survivors, survivors + found);
  *survivor_count = found;
  if (stats != nullptr) {
    stats[0] = v_count;
    stats[1] = stat_sieve.load();
    stats[2] = stat_lut.load();
    stats[3] = stat_survivors.load();
  }
  return 0;
}

}  // extern "C"
