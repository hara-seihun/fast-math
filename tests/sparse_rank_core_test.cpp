#include "fast_math.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

constexpr std::array<std::uint64_t, 38> kRowOffsets{
    0, 3, 9, 14, 16, 24, 29, 37, 43, 49, 56, 64, 72,
    78, 82, 86, 90, 96, 103, 113, 118, 127, 133, 140,
    144, 150, 153, 159, 162, 168, 172, 178, 183, 188,
    192, 199, 200, 202,
};

constexpr std::array<std::uint32_t, 202> kColumns{
    6, 11, 12, 1, 9, 11, 20, 25, 26, 0, 2, 11, 23, 25,
    13, 22, 6, 7, 8, 11, 16, 17, 20, 26, 4, 10, 16, 18,
    24, 6, 8, 11, 21, 22, 23, 25, 26, 7, 11, 12, 16, 23,
    28, 1, 2, 9, 15, 17, 26, 7, 8, 13, 23, 24, 25, 26,
    2, 3, 11, 12, 16, 20, 23, 26, 1, 5, 9, 10, 14, 18,
    20, 23, 2, 6, 12, 22, 27, 28, 1, 16, 26, 28, 3, 7,
    11, 25, 19, 23, 25, 27, 7, 9, 11, 12, 15, 20, 0, 1,
    3, 6, 10, 12, 16, 6, 15, 16, 18, 19, 20, 21, 22, 23,
    28, 3, 12, 14, 15, 16, 0, 4, 6, 7, 8, 10, 16, 17,
    19, 5, 9, 14, 21, 23, 25, 3, 6, 8, 10, 12, 21, 24,
    2, 4, 5, 18, 1, 3, 10, 12, 17, 19, 5, 7, 10, 0, 11,
    14, 15, 16, 23, 14, 16, 17, 6, 8, 10, 13, 16, 28, 3,
    15, 22, 23, 5, 14, 17, 19, 21, 25, 9, 13, 19, 21, 25,
    5, 14, 22, 23, 27, 2, 5, 7, 14, 3, 5, 7, 13, 22, 24,
    26, 26, 3, 7,
};

constexpr std::array<std::uint32_t, 202> kValues{
    48, 75, 101, 86, 157, 5, 182, 157, 134, 173, 173, 31,
    111, 83, 5, 68, 144, 84, 120, 57, 133, 21, 192, 65,
    138, 34, 22, 195, 161, 162, 36, 165, 53, 142, 80, 62,
    55, 206, 90, 116, 68, 105, 175, 169, 59, 118, 118, 71,
    127, 206, 17, 14, 14, 159, 121, 139, 40, 161, 79, 29,
    182, 27, 71, 9, 111, 112, 196, 21, 192, 134, 38, 197,
    164, 6, 54, 28, 89, 86, 149, 47, 139, 149, 39, 123,
    61, 131, 165, 199, 199, 36, 107, 41, 86, 15, 97, 26,
    87, 106, 183, 178, 202, 198, 166, 5, 97, 18, 23, 191,
    192, 48, 187, 9, 139, 136, 73, 84, 203, 124, 73, 105,
    47, 1, 102, 13, 21, 35, 41, 202, 203, 86, 185, 186,
    88, 11, 158, 17, 197, 53, 8, 58, 152, 10, 176, 204,
    49, 89, 130, 155, 206, 122, 63, 52, 199, 168, 24, 70,
    45, 128, 133, 203, 184, 20, 74, 55, 148, 47, 109, 15,
    127, 200, 67, 181, 139, 135, 146, 198, 139, 73, 162,
    13, 68, 20, 106, 157, 60, 146, 0, 28, 169, 56, 57,
    51, 82, 182, 112, 189, 49, 185, 22, 72, 38, 79,
};

constexpr std::array<std::uint64_t, 29> kExpectedPivotRows{
    35, 3, 23, 15, 32, 36, 13, 5, 27, 12, 2, 25, 33, 29,
    26, 9, 34, 0, 31, 20, 17, 22, 1, 8, 19, 28, 7, 11, 14,
};

constexpr std::array<std::uint32_t, 29> kExpectedPivotColumns{
    26, 13, 4, 27, 19, 3, 28, 18, 17, 1, 0, 5, 2, 15, 22,
    24, 8, 6, 9, 10, 14, 21, 20, 12, 25, 7, 11, 23, 16,
};

}  // namespace

int main() {
  std::array<std::uint64_t, 29> pivot_rows{};
  std::array<std::uint32_t, 29> pivot_columns{};
  fast_math_sparse_rank_stats stats{};
  std::array<char, 512> error{};
  const auto status = fast_math_sparse_rank_mod_u32(
      kRowOffsets.data(),
      kColumns.data(),
      kValues.data(),
      37,
      29,
      kValues.size(),
      211,
      0,
      pivot_rows.data(),
      pivot_columns.data(),
      pivot_rows.size(),
      &stats,
      error.data(),
      error.size());
  if (status != 0) {
    std::cerr << "sparse rank failed: " << error.data() << '\n';
    return 1;
  }
  if (stats.rank != 29 ||
      stats.elimination_steps != 108 ||
      stats.basis_nonzeros != 146 ||
      !std::equal(
          pivot_rows.begin(),
          pivot_rows.end(),
          kExpectedPivotRows.begin()) ||
      !std::equal(
          pivot_columns.begin(),
          pivot_columns.end(),
          kExpectedPivotColumns.begin())) {
    std::cerr
        << "portable sparse-rank witness changed: rank="
        << stats.rank
        << " eliminations=" << stats.elimination_steps
        << " basis_nonzeros=" << stats.basis_nonzeros
        << '\n';
    return 1;
  }
  return 0;
}
