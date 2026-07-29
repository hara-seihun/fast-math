#include "fast_math.h"
#include "parallel.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

#if FAST_MATH_USE_COMMONCRYPTO
#include <CommonCrypto/CommonDigest.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<std::uint8_t, 21> kFormatTag{
    'f', 'a', 's', 't', '-', 'm', 'a', 't', 'h', '/',
    'u', '6', '4', '-', 'r', 'o', 'w', 's', '/', '1', '\0'};

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

std::array<std::uint8_t, 8> little_endian_u64(std::uint64_t value) {
  std::array<std::uint8_t, 8> bytes{};
  for (std::size_t index = 0; index < bytes.size(); ++index) {
    bytes[index] = static_cast<std::uint8_t>(value >> (8 * index));
  }
  return bytes;
}

#if FAST_MATH_USE_COMMONCRYPTO

class Sha256 {
 public:
  Sha256() {
    if (CC_SHA256_Init(&context_) != 1) {
      throw std::runtime_error("CC_SHA256_Init failed");
    }
  }

  void update(const void* data, std::size_t size) {
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    while (size != 0) {
      const auto chunk = static_cast<CC_LONG>(
          std::min<std::size_t>(
              size, std::numeric_limits<CC_LONG>::max()));
      if (CC_SHA256_Update(&context_, cursor, chunk) != 1) {
        throw std::runtime_error("CC_SHA256_Update failed");
      }
      cursor += chunk;
      size -= chunk;
    }
  }

  void finish(std::uint8_t* digest) {
    if (CC_SHA256_Final(digest, &context_) != 1) {
      throw std::runtime_error("CC_SHA256_Final failed");
    }
  }

 private:
  CC_SHA256_CTX context_{};
};

#else

constexpr std::array<std::uint32_t, 64> kRoundConstants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

class Sha256 {
 public:
  void update(const void* data, std::size_t size) {
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    if (size > (std::numeric_limits<std::uint64_t>::max() - bit_count_) / 8) {
      throw std::overflow_error("SHA-256 input is too large");
    }
    bit_count_ += static_cast<std::uint64_t>(size) * 8;

    while (size != 0) {
      const auto copied = std::min(size, buffer_.size() - buffer_size_);
      std::memcpy(buffer_.data() + buffer_size_, cursor, copied);
      buffer_size_ += copied;
      cursor += copied;
      size -= copied;
      if (buffer_size_ == buffer_.size()) {
        transform(buffer_.data());
        buffer_size_ = 0;
      }
    }
  }

  void finish(std::uint8_t* digest) {
    buffer_[buffer_size_++] = 0x80;
    if (buffer_size_ > 56) {
      std::fill(
          buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
          buffer_.end(),
          0);
      transform(buffer_.data());
      buffer_size_ = 0;
    }
    std::fill(
        buffer_.begin() + static_cast<std::ptrdiff_t>(buffer_size_),
        buffer_.begin() + 56,
        0);
    for (std::size_t index = 0; index < 8; ++index) {
      buffer_[63 - index] =
          static_cast<std::uint8_t>(bit_count_ >> (8 * index));
    }
    transform(buffer_.data());
    for (std::size_t word = 0; word < state_.size(); ++word) {
      for (std::size_t byte = 0; byte < 4; ++byte) {
        digest[4 * word + byte] = static_cast<std::uint8_t>(
            state_[word] >> (24 - 8 * byte));
      }
    }
  }

 private:
  void transform(const std::uint8_t* block) {
    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16; ++index) {
      words[index] =
          (static_cast<std::uint32_t>(block[4 * index]) << 24) |
          (static_cast<std::uint32_t>(block[4 * index + 1]) << 16) |
          (static_cast<std::uint32_t>(block[4 * index + 2]) << 8) |
          static_cast<std::uint32_t>(block[4 * index + 3]);
    }
    for (std::size_t index = 16; index < words.size(); ++index) {
      const auto previous15 = words[index - 15];
      const auto previous2 = words[index - 2];
      const auto sigma0 =
          std::rotr(previous15, 7) ^
          std::rotr(previous15, 18) ^
          (previous15 >> 3);
      const auto sigma1 =
          std::rotr(previous2, 17) ^
          std::rotr(previous2, 19) ^
          (previous2 >> 10);
      words[index] =
          words[index - 16] + sigma0 + words[index - 7] + sigma1;
    }

    auto a = state_[0];
    auto b = state_[1];
    auto c = state_[2];
    auto d = state_[3];
    auto e = state_[4];
    auto f = state_[5];
    auto g = state_[6];
    auto h = state_[7];
    for (std::size_t index = 0; index < words.size(); ++index) {
      const auto sum1 =
          std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
      const auto choose = (e & f) ^ (~e & g);
      const auto temporary1 =
          h + sum1 + choose + kRoundConstants[index] + words[index];
      const auto sum0 =
          std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
      const auto majority = (a & b) ^ (a & c) ^ (b & c);
      const auto temporary2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temporary1;
      d = c;
      c = b;
      b = a;
      a = temporary1 + temporary2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_{
      0x6a09e667U,
      0xbb67ae85U,
      0x3c6ef372U,
      0xa54ff53aU,
      0x510e527fU,
      0x9b05688cU,
      0x1f83d9abU,
      0x5be0cd19U};
  std::array<std::uint8_t, 64> buffer_{};
  std::size_t buffer_size_ = 0;
  std::uint64_t bit_count_ = 0;
};

#endif

void append_u64(Sha256& digest, std::uint64_t value) {
  const auto bytes = little_endian_u64(value);
  digest.update(bytes.data(), bytes.size());
}

void append_row(
    Sha256& digest,
    const std::uint64_t* row,
    std::size_t field_count) {
  if constexpr (std::endian::native == std::endian::little) {
    digest.update(row, field_count * sizeof(std::uint64_t));
  } else {
    for (std::size_t field = 0; field < field_count; ++field) {
      append_u64(digest, row[field]);
    }
  }
}

}  // namespace

extern "C" {

int fast_math_digest_u64_rows_sha256(
    const std::uint64_t* rows,
    std::size_t row_count,
    std::size_t field_count,
    const std::uint8_t* namespace_data,
    std::size_t namespace_size,
    std::uint32_t thread_count,
    std::uint8_t* digests,
    fast_math_digest_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (rows == nullptr || digests == nullptr || stats == nullptr) {
      throw std::invalid_argument("digest pointer is null");
    }
    if (row_count == 0 || field_count == 0) {
      throw std::invalid_argument(
          "digest input must contain at least one row and field");
    }
    if (namespace_size != 0 && namespace_data == nullptr) {
      throw std::invalid_argument("digest namespace pointer is null");
    }
    if (row_count >
        std::numeric_limits<std::size_t>::max() / field_count) {
      throw std::overflow_error("digest input is too large");
    }
    if (row_count >
        std::numeric_limits<std::size_t>::max() / 32) {
      throw std::overflow_error("digest output is too large");
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();

    Sha256 prefix;
    prefix.update(kFormatTag.data(), kFormatTag.size());
    append_u64(prefix, namespace_size);
    if (namespace_size != 0) {
      prefix.update(namespace_data, namespace_size);
    }
    append_u64(prefix, field_count);

    fast_math_internal::parallel_for_static(
        row_count,
        thread_count,
        [&](std::size_t row_index) {
          auto digest = prefix;
          append_row(
              digest,
              rows + row_index * field_count,
              field_count);
          digest.finish(digests + row_index * 32);
        });

    stats->row_count = row_count;
    stats->field_count = field_count;
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
