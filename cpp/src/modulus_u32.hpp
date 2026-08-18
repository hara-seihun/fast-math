#pragma once

#include <cstdint>
#include <stdexcept>

namespace fast_math_internal {

class PrimeModulusU32 {
 public:
  explicit PrimeModulusU32(std::uint32_t prime)
      : prime_(prime)
#if defined(__SIZEOF_INT128__)
        ,
        reciprocal_(static_cast<std::uint64_t>(
            (static_cast<unsigned __int128>(1) << 64) / prime))
#endif
  {}

  std::uint32_t prime() const {
    return prime_;
  }

  std::uint32_t reduce(std::uint32_t value) const {
    return value < prime_ ? value : value % prime_;
  }

  std::uint32_t multiply(
      std::uint32_t left,
      std::uint32_t right) const {
    const auto product = static_cast<std::uint64_t>(left) * right;
#if defined(__SIZEOF_INT128__)
    const auto quotient = static_cast<std::uint64_t>(
        (static_cast<unsigned __int128>(product) * reciprocal_) >> 64);
    auto remainder = product - quotient * prime_;
    if (remainder >= prime_) remainder -= prime_;
#else
    return static_cast<std::uint32_t>(product % prime_);
#endif
#if defined(FAST_MATH_VERIFY_MODULAR_ARITHMETIC)
    if (left >= prime_ || right >= prime_ ||
        remainder != product % prime_) {
      throw std::logic_error("invalid modular multiplication");
    }
#endif
    return static_cast<std::uint32_t>(remainder);
  }

  std::uint32_t subtract(
      std::uint32_t left,
      std::uint32_t right) const {
    const auto result = left >= right
        ? left - right
        : prime_ - (right - left);
#if defined(FAST_MATH_VERIFY_MODULAR_ARITHMETIC)
    if (left >= prime_ || right >= prime_ || result >= prime_) {
      throw std::logic_error("invalid modular subtraction");
    }
#endif
    return result;
  }

  std::uint32_t add(
      std::uint32_t left,
      std::uint32_t right) const {
    const auto sum = static_cast<std::uint64_t>(left) + right;
    const auto result = static_cast<std::uint32_t>(
        sum >= prime_ ? sum - prime_ : sum);
#if defined(FAST_MATH_VERIFY_MODULAR_ARITHMETIC)
    if (left >= prime_ || right >= prime_ || result != sum % prime_) {
      throw std::logic_error("invalid modular addition");
    }
#endif
    return result;
  }

  std::uint32_t negate_nonzero(std::uint32_t value) const {
    return prime_ - value;
  }

  std::uint32_t power(
      std::uint32_t base,
      std::uint32_t exponent) const {
    std::uint32_t result = 1 % prime_;
    while (exponent != 0) {
      if ((exponent & 1U) != 0) result = multiply(result, base);
      exponent >>= 1;
      if (exponent != 0) base = multiply(base, base);
    }
    return result;
  }

  std::uint32_t inverse(std::uint32_t value) const {
    if (value == 0) {
      throw std::invalid_argument("zero pivot is not invertible");
    }
    return power(value, prime_ - 2);
  }

 private:
  std::uint32_t prime_;
#if defined(__SIZEOF_INT128__)
  std::uint64_t reciprocal_;
#endif
};

}  // namespace fast_math_internal
