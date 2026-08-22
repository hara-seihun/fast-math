// HIP backend for the shift-divisor gate scan (contract documented in
// python/fast_math/shift_gates.py; CPU twin in shift_gates.cpp).
//
// Kernel 1: one workgroup per (wheel class, t-chunk).  The chunk's bitmap
// lives in LDS; threads share the (sieve prime, form) task list, striding a
// q-ascending order so every thread gets a mix of dense small primes and
// sparse large ones.  Extraction applies the packed alive tables and appends
// candidates to a global buffer.
//
// Kernel 2: one thread per candidate runs the exact per-form condition -
// strip modulus-smooth factors, then deterministic Miller-Rabin (Montgomery,
// one or two 64-bit words), accepting primes and squares of primes.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

constexpr unsigned kChunkBits = 1u << 16;  // 8 KiB LDS bitmap
constexpr unsigned kThreads = 256;
constexpr unsigned kMaxForms = 16;
constexpr unsigned kMaxLuts = 8;
constexpr unsigned kMaxSmooth = 8;

__constant__ u64 kMrBases[13] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};

struct GateConstants {
  u64 form_a[kMaxForms];
  u64 form_b[kMaxForms];
  u64 lut_moduli[kMaxLuts];
  u64 lut_offsets[kMaxLuts];
  u32 smooth[kMaxSmooth];
  u64 smooth_inv_lo[kMaxSmooth];  // p^{-1} mod 2^128, low word
  u64 smooth_inv_hi[kMaxSmooth];  // p^{-1} mod 2^128, high word
  u32 smooth_r64[kMaxSmooth];     // 2^64 mod p
  u32 form_count;
  u32 lut_count;
  u32 smooth_count;
  u64 wheel;
  u64 v_start;
  u64 v_count;
  u64 t_low;
  u64 t_high;
  u64 chunks_per_row;
};

// Dense sieve primes are applied by stamping precomputed periodic bit
// patterns instead of per-hit atomics.  For a fixed prime q the 13 forms
// kill t-residues c_i - r*W^{-1} (mod q): the row r shifts all forms by the
// same amount, so one packed pattern per prime group (adjacent primes
// paired via CRT, period q1*q2 words) serves every row; a block just reads
// the window at offset (r*W^{-1} + t_base) mod period with a funnel shift.
constexpr u32 kMaxPatGroups = 40;
constexpr u32 kPatternMaxPrime = 251;  // tables stay small enough for L2
struct PatternConstants {
  u32 count;
  u32 period[kMaxPatGroups];    // bit period = word period (odd)
  u32 word_off[kMaxPatGroups];  // offset into the pattern buffer, in words
  u32 invw[kMaxPatGroups];      // W^{-1} mod period
};
__constant__ PatternConstants kPat;

// Deterministic MR bases and 2^64 mod base, for split-word reduction.
__constant__ u32 kMrR64[13] = {0, 1, 1, 2, 5, 3, 1, 17, 6, 24, 16, 12, 16};
// Squares mod 256 and mod 63, as bitmasks: a value failing either test is
// not a perfect square, sparing the Newton isqrt on ~96% of composites.
__constant__ u64 kSq256[4] = {0x202021202030213ULL, 0x202021202020213ULL,
                              0x202021202030212ULL, 0x202021202020212ULL};
__constant__ u64 kSq63 = 0x402483012450293ULL;

__constant__ GateConstants kGate;

__device__ u64 mulmod_u64(u64 a, u64 b, u64 m) { return u64((u128(a) * b) % m); }

// x mod p via split words: avoids the soft 128-bit division.
__device__ u32 mod_u128_u32(u128 x, u32 p, u32 r64) {
  const u64 hi = u64(x >> 64) % p;
  const u64 lo = u64(x) % p;
  return u32((hi * r64 + lo) % p);
}

__device__ bool lut_alive(const u64* lut_bits, u64 v) {
  for (u32 i = 0; i < kGate.lut_count; ++i) {
    u64 index = kGate.lut_offsets[i] + v % kGate.lut_moduli[i];
    if (((lut_bits[index >> 6] >> (index & 63)) & 1) == 0) return false;
  }
  return true;
}

__global__ void sieve_kernel(
    const u64* wheel_classes,
    const u64* pat_words,
    const u32* primes,
    u32 prime_count,
    const u32* a_mod,    // [prime][form]
    const u32* inv_aw,   // [prime][form]
    const u64* lut_bits,
    u64* candidates,
    u64* candidate_count,
    u64 candidate_capacity,
    u64* stat_sieve,
    u64 task_base) {
  __shared__ u64 bitmap[kChunkBits / 64];
  const u64 task = task_base + blockIdx.x;
  const u64 class_index = task / kGate.chunks_per_row;
  const u64 chunk_index = task % kGate.chunks_per_row;
  const u64 r = wheel_classes[class_index];
  const u64 t_base = kGate.t_low + chunk_index * u64(kChunkBits);
  if (t_base >= kGate.t_high) return;
  const u64 t_len = min(u64(kChunkBits), kGate.t_high - t_base);

  const unsigned init_words = unsigned((t_len + 63) / 64);
  for (unsigned w = threadIdx.x; w < init_words; w += kThreads)
    bitmap[w] = ~u64(0);
  __syncthreads();

  // Pattern phase: non-atomic; each thread owns a strided set of words.
  for (u32 gi = 0; gi < kPat.count; ++gi) {
    const u32 P = kPat.period[gi];
    const u64* table = pat_words + kPat.word_off[gi];
    const u64 o = (u64(r % P) * kPat.invw[gi] + t_base) % P;
    const u32 b = u32(o) & 63u;
    const u32 step = kThreads % P;
    u32 y = (u32(o >> 6) + threadIdx.x) % P;
    for (unsigned w = threadIdx.x; w < init_words; w += kThreads) {
      const u32 y2 = (y + 1 == P) ? 0 : y + 1;
      const u64 mask = b ? (table[y] >> b) | (table[y2] << (64 - b)) : table[y];
      bitmap[w] &= mask;
      y += step;
      if (y >= P) y -= P;
    }
  }
  __syncthreads();

  const u32 form_count = kGate.form_count;
  const u32 task_count = prime_count * form_count;
  for (u32 work = threadIdx.x; work < task_count; work += kThreads) {
    const u32 qi = work / form_count;
    const u32 fi = work % form_count;
    const u32 q = primes[qi];
    const u32 inv = inv_aw[work];
    if (inv == 0) continue;
    const u64 rhs =
        (kGate.form_b[fi] % q + q - mulmod_u64(a_mod[work], r % q, q)) % q;
    u64 t0 = mulmod_u64(rhs, inv, q);
    t0 = (t0 + q - t_base % q) % q;
    for (u64 i = t0; i < t_len; i += q)
      atomicAnd(&bitmap[i >> 6], ~(u64(1) << (i & 63)));
  }
  __syncthreads();

  u64 local = 0;
  const unsigned word_count = unsigned((t_len + 63) / 64);
  for (unsigned w = threadIdx.x; w < word_count; w += kThreads) {
    u64 word = bitmap[w];
    if (w == word_count - 1 && (t_len & 63) != 0)
      word &= (u64(1) << (t_len & 63)) - 1;
    while (word != 0) {
      const int bit = __ffsll((long long)word) - 1;
      word &= word - 1;
      const u64 t = t_base + (u64(w) << 6) + u64(bit);
      const u64 v = r + kGate.wheel * t;
      if (v < kGate.v_start || v - kGate.v_start >= kGate.v_count) continue;
      ++local;
      if (!lut_alive(lut_bits, v)) continue;
      const u64 slot = atomicAdd(candidate_count, u64(1));
      if (slot < candidate_capacity) candidates[slot] = v;
    }
  }
  if (local != 0) atomicAdd(stat_sieve, local);
}

// --- Miller-Rabin ----------------------------------------------------------

struct Mont64 {
  u64 n, ninv, r2;
};

__device__ Mont64 mont64_create(u64 n) {
  u64 inv = n;
  for (int i = 0; i < 5; ++i) inv *= 2 - n * inv;
  u64 r = (~u64(0)) % n + 1;
  return Mont64{n, ~inv + 1, u64((u128(r) * r) % n)};
}

__device__ u64 mont64_mul(const Mont64& m, u64 a, u64 b) {
  u128 t = u128(a) * b;
  u128 u = u128(u64(t) * m.ninv) * m.n;
  // (t + u) is divisible by 2^64 and can span 129 bits when n > 2^64 * 0.618,
  // so accumulate the high halves explicitly instead of adding in u128.
  u128 sum = u128(u64(t >> 64)) + u64(u >> 64) + (u64(t) != 0 ? 1 : 0);
  return u64(sum >= m.n ? sum - m.n : sum);
}

__device__ bool mr_prime_u64(u64 n) {
  if (n < 2) return false;
  for (int i = 0; i < 13; ++i) {
    if (n % kMrBases[i] == 0) return n == kMrBases[i];
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
  for (int i = 0; i < 13; ++i) {
    u64 x = mont64_mul(m, kMrBases[i] % n, m.r2);
    u64 y = one;
    u64 e = d;
    while (e != 0) {
      if (e & 1) y = mont64_mul(m, y, x);
      x = mont64_mul(m, x, x);
      e >>= 1;
    }
    if (y == one || y == minus_one) continue;
    bool witness = true;
    for (int k = 1; k < s; ++k) {
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

struct Mont128 {
  u128 n;
  u64 ninv;
  u128 r2;
};

__device__ u128 mod_double(u128 x, u128 n) {
  u128 y = x << 1;
  return (y >= n || y < x) ? y - n : y;
}

__device__ u128 pow2_mod(unsigned bits, u128 n) {
  u128 x = 1 % n;
  for (unsigned i = 0; i < bits; ++i) x = mod_double(x, n);
  return x;
}

__device__ Mont128 mont128_create(u128 n) {
  u64 n0 = u64(n);
  u64 inv = n0;
  for (int i = 0; i < 5; ++i) inv *= 2 - n0 * inv;
  return Mont128{n, ~inv + 1, pow2_mod(256, n)};
}

__device__ u128 mont128_mul(const Mont128& m, u128 a, u128 b) {
  u64 a0 = u64(a), a1 = u64(a >> 64);
  u64 b0 = u64(b), b1 = u64(b >> 64);
  u64 n0 = u64(m.n), n1 = u64(m.n >> 64);
  u64 t0 = 0, t1 = 0, t2 = 0;
  for (int limb = 0; limb < 2; ++limb) {
    const u64 ai = limb == 0 ? a0 : a1;
    u128 c = u128(ai) * b0 + t0;
    u64 s0 = u64(c);
    c = u128(ai) * b1 + t1 + u64(c >> 64);
    u64 s1 = u64(c);
    u64 s2 = t2 + u64(c >> 64);
    u64 q = s0 * m.ninv;
    c = u128(q) * n0 + s0;
    c = u128(q) * n1 + s1 + u64(c >> 64);
    t0 = u64(c);
    u128 c2 = u128(s2) + u64(c >> 64);
    t1 = u64(c2);
    t2 = u64(c2 >> 64);
  }
  u128 t = (u128(t1) << 64) | t0;
  if (t2 != 0 || t >= m.n) t -= m.n;
  return t;
}

__device__ bool mr_prime_u128(u128 n) {
  if ((n >> 64) == 0) return mr_prime_u64(u64(n));
  for (int i = 0; i < 13; ++i) {
    if (mod_u128_u32(n, u32(kMrBases[i]), kMrR64[i]) == 0) return false;
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
  for (int i = 0; i < 13; ++i) {
    u128 x = mont128_mul(m, kMrBases[i], m.r2);
    u128 y = one;
    u128 e = d;
    while (e != 0) {
      if (e & 1) y = mont128_mul(m, y, x);
      x = mont128_mul(m, x, x);
      e >>= 1;
    }
    if (y == one || y == minus_one) continue;
    bool witness = true;
    for (int k = 1; k < s; ++k) {
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

__device__ u64 isqrt_u128(u128 n) {
  if (n == 0) return 0;
  double approx = double(u64(n >> 64)) * 18446744073709551616.0 + double(u64(n));
  u64 root = u64(sqrt(approx));
  if (root == 0) root = 1;
  for (int i = 0; i < 6; ++i) {
    u64 next = u64((n / root + root) / 2);
    if (next == root) break;
    root = next;
  }
  while (u128(root) * root > n) --root;
  while ((u128(root) + 1) * (u128(root) + 1) <= n) ++root;
  return root;
}

__device__ bool prime_or_prime_square(u128 part) {
  if (part <= 1) return part == 1;
  if (mr_prime_u128(part)) return true;
  // Composite: only a prime square still passes.  Cheap quadratic-residue
  // tests reject ~96% of composites before the Newton isqrt.
  const u32 r256 = u32(part) & 255u;
  if (((kSq256[r256 >> 6] >> (r256 & 63)) & 1) == 0) return false;
  const u32 r63 = mod_u128_u32(part, 63u, 22u);  // 2^64 mod 63 = 22
  if (((kSq63 >> r63) & 1) == 0) return false;
  u64 root = isqrt_u128(part);
  return u128(root) * root == part && mr_prime_u64(root);
}

__global__ void mr_kernel(
    const u64* candidates,
    u64 candidate_count,
    u64* survivors,
    u64* survivor_count,
    u64 survivor_capacity,
    u64* stat_lut) {
  const u64 index = u64(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= candidate_count) return;
  const u64 v = candidates[index];
  atomicAdd(stat_lut, u64(1));
  for (u32 fi = 0; fi < kGate.form_count; ++fi) {
    u128 value = u128(kGate.form_a[fi]) * v - kGate.form_b[fi];
    for (u32 s = 0; s < kGate.smooth_count; ++s) {
      const u32 p = kGate.smooth[s];
      if (p == 2) {
        while ((u64(value) & 1) == 0) value >>= 1;
        continue;
      }
      // Exact division by p^{-1} mod 2^128: two multiplies, no soft division.
      const u128 inv =
          (u128(kGate.smooth_inv_hi[s]) << 64) | kGate.smooth_inv_lo[s];
      while (mod_u128_u32(value, p, kGate.smooth_r64[s]) == 0) value *= inv;
    }
    if (!prime_or_prime_square(value)) return;
  }
  const u64 slot = atomicAdd(survivor_count, u64(1));
  if (slot < survivor_capacity) survivors[slot] = v;
}

int status(hipError_t error) { return error == hipSuccess ? 0 : 3; }

void set_error(char* destination, std::size_t size, const char* message) {
  if (destination == nullptr || size == 0) return;
  std::strncpy(destination, message, size - 1);
  destination[size - 1] = '\0';
}

// Modular inverse for composite moduli (pattern periods q1*q2).
u64 modinv_host(u64 a, u64 m) {
  long long t = 0, nt = 1;
  long long r = (long long)m, nr = (long long)(a % m);
  while (nr != 0) {
    const long long q = r / nr;
    const long long tmp_t = t - q * nt;
    t = nt; nt = tmp_t;
    const long long tmp_r = r - q * nr;
    r = nr; nr = tmp_r;
  }
  return u64(t < 0 ? t + (long long)m : t);
}

u64 powmod_host(u64 base, u64 exponent, u64 m) {
  u64 result = 1 % m;
  base %= m;
  while (exponent != 0) {
    if (exponent & 1) result = u64((u128(result) * base) % m);
    base = u64((u128(base) * base) % m);
    exponent >>= 1;
  }
  return result;
}

}  // namespace

extern "C" {

int fast_math_hip_shift_gate_scan_u64(
    const u64* form_a,
    const u64* form_b,
    std::size_t form_count,
    const u32* smooth_primes,
    std::size_t smooth_prime_count,
    const u64* lut_moduli,
    const u64* lut_offsets,
    std::size_t lut_count,
    const u64* lut_bits,
    std::size_t lut_word_count,
    u64 wheel,
    const u64* wheel_classes,
    std::size_t wheel_class_count,
    u32 sieve_low,
    u32 sieve_bound,
    u64 v_start,
    u64 v_count,
    u64* survivors,
    std::size_t survivor_capacity,
    std::size_t* survivor_count,
    u64* stats,
    char* error_message,
    std::size_t error_capacity) {
  if (form_count == 0 || form_count > kMaxForms || lut_count > kMaxLuts ||
      smooth_prime_count > kMaxSmooth || wheel_class_count == 0 || v_count == 0) {
    set_error(error_message, error_capacity, "unsupported gate shape");
    return 1;
  }

  // Host-side prime tables.
  std::vector<u32> primes;
  {
    std::vector<bool> composite(sieve_bound + 1, false);
    for (u64 p = 2; p <= sieve_bound; ++p) {
      if (composite[p]) continue;
      if (p >= sieve_low) primes.push_back(u32(p));
      for (u64 q = p * p; q <= sieve_bound; q += p) composite[q] = true;
    }
  }
  std::vector<u32> a_mod(primes.size() * form_count);
  std::vector<u32> inv_aw(primes.size() * form_count);
  for (std::size_t qi = 0; qi < primes.size(); ++qi) {
    const u32 q = primes[qi];
    const u64 w_mod = wheel % q;
    for (std::size_t fi = 0; fi < form_count; ++fi) {
      const u64 a = form_a[fi] % q;
      a_mod[qi * form_count + fi] = u32(a);
      inv_aw[qi * form_count + fi] =
          (a == 0 || w_mod == 0)
              ? 0
              : u32(powmod_host(u64((u128(a) * w_mod) % q), q - 2, q));
    }
  }

  // Pattern groups: adjacent primes <= kPatternMaxPrime paired via CRT.
  // Kill residue for form fi mod q is form_b*inv(a*W) = form_b*inv_aw mod q;
  // the row-dependent shift r*W^{-1} is applied uniformly in the kernel.
  PatternConstants pat{};
  std::vector<u64> pat_words;
  std::size_t n_pat = 0;
  while (n_pat < primes.size() && primes[n_pat] <= kPatternMaxPrime &&
         pat.count < kMaxPatGroups) {
    const std::size_t take = (n_pat + 1 < primes.size() &&
                              primes[n_pat + 1] <= kPatternMaxPrime &&
                              pat.count + 1 < kMaxPatGroups)
                                 ? 2
                                 : 1;
    u32 period = primes[n_pat];
    if (take == 2) period *= primes[n_pat + 1];
    const u32 gi = pat.count++;
    pat.period[gi] = period;
    pat.word_off[gi] = u32(pat_words.size());
    pat.invw[gi] = u32(modinv_host(wheel % period, period));
    pat_words.resize(pat_words.size() + period, ~u64(0));
    u64* table = pat_words.data() + pat.word_off[gi];
    for (std::size_t k = 0; k < take; ++k) {
      const std::size_t qi = n_pat + k;
      const u32 q = primes[qi];
      for (std::size_t fi = 0; fi < form_count; ++fi) {
        const u64 inv = inv_aw[qi * form_count + fi];
        if (inv == 0) continue;
        const u64 c = (form_b[fi] % q) * inv % q;
        for (u64 p = c; p < u64(period) * 64; p += q)
          table[p >> 6] &= ~(u64(1) << (p & 63));
      }
    }
    n_pat += take;
  }

  GateConstants gate{};
  for (std::size_t i = 0; i < form_count; ++i) {
    gate.form_a[i] = form_a[i];
    gate.form_b[i] = form_b[i];
  }
  for (std::size_t i = 0; i < lut_count; ++i) {
    gate.lut_moduli[i] = lut_moduli[i];
    gate.lut_offsets[i] = lut_offsets[i];
  }
  for (std::size_t i = 0; i < smooth_prime_count; ++i) {
    const u32 p = smooth_primes[i];
    gate.smooth[i] = p;
    if (p > 2) {
      u128 inv = p;  // Newton: p^{-1} mod 2^128
      for (int it = 0; it < 7; ++it) inv *= 2 - u128(p) * inv;
      gate.smooth_inv_lo[i] = u64(inv);
      gate.smooth_inv_hi[i] = u64(inv >> 64);
      gate.smooth_r64[i] = u32(((~u64(0)) % p + 1) % p);
    }
  }
  gate.form_count = u32(form_count);
  gate.lut_count = u32(lut_count);
  gate.smooth_count = u32(smooth_prime_count);
  gate.wheel = wheel;
  gate.v_start = v_start;
  gate.v_count = v_count;
  gate.t_low = v_start / wheel;
  gate.t_high = (v_start + v_count - 1) / wheel + 1;
  gate.chunks_per_row =
      (gate.t_high - gate.t_low + kChunkBits - 1) / kChunkBits;

  const u64 task_count = u64(wheel_class_count) * gate.chunks_per_row;
  // Candidate volume: wheel-alive density times sieve survival
  // exp(-forms * (lnln(bound) - lnln(low))), with 3x headroom.
  const double survival = std::exp(
      -double(form_count) * (std::log(std::log(double(sieve_bound))) -
                             std::log(std::log(double(sieve_low)))));
  const double alive =
      double(v_count) * double(wheel_class_count) / double(wheel);
  const u64 candidate_capacity =
      std::max<u64>(u64(1) << 20, u64(alive * survival) * 3 + (u64(1) << 18));

  u64 *d_classes = nullptr, *d_lut_bits = nullptr, *d_candidates = nullptr,
      *d_survivors = nullptr, *d_counters = nullptr, *d_pat = nullptr;
  u32 *d_primes = nullptr, *d_a_mod = nullptr, *d_inv_aw = nullptr;
  int rc = 0;
  do {
    if ((rc = status(hipMemcpyToSymbol(HIP_SYMBOL(kGate), &gate, sizeof gate)))) break;
    if ((rc = status(hipMemcpyToSymbol(HIP_SYMBOL(kPat), &pat, sizeof pat)))) break;
    if ((rc = status(hipMalloc(&d_pat, std::max<std::size_t>(pat_words.size(), 1) * sizeof(u64))))) break;
    if (!pat_words.empty() &&
        (rc = status(hipMemcpy(d_pat, pat_words.data(),
                               pat_words.size() * sizeof(u64),
                               hipMemcpyHostToDevice))))
      break;
    if ((rc = status(hipMalloc(&d_classes, wheel_class_count * sizeof(u64))))) break;
    if ((rc = status(hipMalloc(&d_lut_bits, lut_word_count * sizeof(u64))))) break;
    if ((rc = status(hipMalloc(&d_primes, primes.size() * sizeof(u32))))) break;
    if ((rc = status(hipMalloc(&d_a_mod, a_mod.size() * sizeof(u32))))) break;
    if ((rc = status(hipMalloc(&d_inv_aw, inv_aw.size() * sizeof(u32))))) break;
    if ((rc = status(hipMalloc(&d_candidates, candidate_capacity * sizeof(u64))))) break;
    if ((rc = status(hipMalloc(&d_survivors, survivor_capacity * sizeof(u64))))) break;
    if ((rc = status(hipMalloc(&d_counters, 4 * sizeof(u64))))) break;
    if ((rc = status(hipMemcpy(d_classes, wheel_classes,
                               wheel_class_count * sizeof(u64),
                               hipMemcpyHostToDevice)))) break;
    if ((rc = status(hipMemcpy(d_lut_bits, lut_bits, lut_word_count * sizeof(u64),
                               hipMemcpyHostToDevice)))) break;
    if ((rc = status(hipMemcpy(d_primes, primes.data(),
                               primes.size() * sizeof(u32),
                               hipMemcpyHostToDevice)))) break;
    if ((rc = status(hipMemcpy(d_a_mod, a_mod.data(), a_mod.size() * sizeof(u32),
                               hipMemcpyHostToDevice)))) break;
    if ((rc = status(hipMemcpy(d_inv_aw, inv_aw.data(),
                               inv_aw.size() * sizeof(u32),
                               hipMemcpyHostToDevice)))) break;
    if ((rc = status(hipMemset(d_counters, 0, 4 * sizeof(u64))))) break;

    // counters: [candidates, sieve_stat, survivors, lut_stat]
    const u64 max_blocks = u64(1) << 20;
    for (u64 base = 0; base < task_count && rc == 0; base += max_blocks) {
      const u64 blocks = std::min(max_blocks, task_count - base);
      hipLaunchKernelGGL(sieve_kernel, dim3(unsigned(blocks)), dim3(kThreads), 0, 0,
                         d_classes, d_pat, d_primes + n_pat,
                         u32(primes.size() - n_pat),
                         d_a_mod + n_pat * form_count,
                         d_inv_aw + n_pat * form_count, d_lut_bits,
                         d_candidates, d_counters, candidate_capacity,
                         d_counters + 1, base);
      rc = status(hipGetLastError());
    }
    if (rc) break;
    if ((rc = status(hipDeviceSynchronize()))) break;

    u64 counters[4] = {0, 0, 0, 0};
    if ((rc = status(hipMemcpy(counters, d_counters, sizeof counters,
                               hipMemcpyDeviceToHost)))) break;
    if (counters[0] > candidate_capacity) {
      char detail[128];
      std::snprintf(detail, sizeof detail,
                    "candidate capacity exceeded: %llu > %llu (sieve stat %llu)",
                    (unsigned long long)counters[0],
                    (unsigned long long)candidate_capacity,
                    (unsigned long long)counters[1]);
      set_error(error_message, error_capacity, detail);
      rc = 2;
      break;
    }
    const u64 candidate_total = counters[0];
    if (candidate_total != 0) {
      const unsigned blocks = unsigned((candidate_total + kThreads - 1) / kThreads);
      hipLaunchKernelGGL(mr_kernel, dim3(blocks), dim3(kThreads), 0, 0,
                         d_candidates, candidate_total, d_survivors,
                         d_counters + 2, survivor_capacity, d_counters + 3);
      if ((rc = status(hipGetLastError()))) break;
      if ((rc = status(hipDeviceSynchronize()))) break;
    }
    if ((rc = status(hipMemcpy(counters, d_counters, sizeof counters,
                               hipMemcpyDeviceToHost)))) break;
    if (counters[2] > survivor_capacity) {
      set_error(error_message, error_capacity, "survivor capacity exceeded");
      rc = 2;
      break;
    }
    const u64 found = counters[2];
    if (found != 0 &&
        (rc = status(hipMemcpy(survivors, d_survivors, found * sizeof(u64),
                               hipMemcpyDeviceToHost))))
      break;
    std::sort(survivors, survivors + found);
    *survivor_count = std::size_t(found);
    if (stats != nullptr) {
      stats[0] = v_count;
      stats[1] = counters[1];
      stats[2] = counters[3];
      stats[3] = found;
    }
  } while (false);

  if (rc == 3) set_error(error_message, error_capacity, "HIP runtime failure");
  hipFree(d_classes);
  hipFree(d_lut_bits);
  hipFree(d_pat);
  hipFree(d_primes);
  hipFree(d_a_mod);
  hipFree(d_inv_aw);
  hipFree(d_candidates);
  hipFree(d_survivors);
  hipFree(d_counters);
  return rc;
}

}  // extern "C"
