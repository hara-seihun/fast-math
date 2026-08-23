#include "fast_math.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kMaxPrime = 251;
constexpr std::uint32_t kMaxWidth = 16;
using Row = std::array<std::uint32_t, kMaxWidth>;

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

bool is_prime(std::uint32_t value) {
  if (value < 2) {
    return false;
  }
  if (value % 2 == 0) {
    return value == 2;
  }
  for (std::uint32_t divisor = 3;
       divisor <= value / divisor;
       divisor += 2) {
    if (value % divisor == 0) {
      return false;
    }
  }
  return true;
}

struct FieldSpec {
  std::uint32_t prime;
  std::uint32_t width;
  std::uint64_t space_size;
  std::array<std::uint32_t, kMaxPrime + 1> inverses{};

  FieldSpec(std::uint32_t checked_prime, std::uint32_t checked_width)
      : prime(checked_prime), width(checked_width), space_size(1) {
    if (!is_prime(prime) || prime > kMaxPrime) {
      throw std::invalid_argument("prime must be a prime at most 251");
    }
    if (width == 0 || width > kMaxWidth) {
      throw std::invalid_argument("width must be between 1 and 16");
    }
    for (std::uint32_t position = 0; position < width; ++position) {
      if (space_size >
          std::numeric_limits<std::uint64_t>::max() / prime) {
        throw std::invalid_argument(
            "prime^width does not fit in unsigned 64-bit codes");
      }
      space_size *= prime;
    }
    inverses[1] = 1;
    for (std::uint32_t value = 2; value < prime; ++value) {
      inverses[value] = static_cast<std::uint32_t>(
          prime -
          (static_cast<std::uint64_t>(prime / value) *
           inverses[prime % value]) %
              prime);
    }
  }
};

Row decode(std::uint64_t code, const FieldSpec& field) {
  Row row{};
  for (std::uint32_t position = 0; position < field.width; ++position) {
    row[position] = static_cast<std::uint32_t>(code % field.prime);
    code /= field.prime;
  }
  return row;
}

std::uint64_t encode(const Row& row, const FieldSpec& field) {
  std::uint64_t code = 0;
  for (std::uint32_t position = field.width; position > 0; --position) {
    code = code * field.prime + row[position - 1];
  }
  return code;
}

void subtract_scaled_from(
    Row& left,
    const Row& right,
    std::uint32_t scale,
    std::uint32_t first_column,
    const FieldSpec& field) {
  for (std::uint32_t column = first_column;
       column < field.width;
       ++column) {
    const auto product = static_cast<std::uint32_t>(
        static_cast<std::uint64_t>(scale) * right[column] % field.prime);
    left[column] = left[column] >= product
        ? left[column] - product
        : left[column] + field.prime - product;
  }
}

void subtract_scaled(
    Row& left,
    const Row& right,
    std::uint32_t scale,
    const FieldSpec& field) {
  subtract_scaled_from(left, right, scale, 0, field);
}

class SpanRank {
 public:
  explicit SpanRank(const FieldSpec& field) : field_(field) {}

  void add(std::uint64_t code) {
    Row row = decode(code, field_);
    for (std::uint32_t column = 0; column < field_.width; ++column) {
      if (row[column] == 0) {
        continue;
      }
      if (has_pivot_[column]) {
        subtract_scaled_from(
            row, rows_[column], row[column], column, field_);
        continue;
      }
      const auto inverse = field_.inverses[row[column]];
      for (std::uint32_t tail = column; tail < field_.width; ++tail) {
        row[tail] = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(row[tail]) * inverse % field_.prime);
      }
      rows_[column] = row;
      has_pivot_[column] = true;
      ++rank_;
      return;
    }
  }

  std::uint32_t rank() const { return rank_; }

 private:
  const FieldSpec& field_;
  std::array<Row, kMaxWidth> rows_{};
  std::array<bool, kMaxWidth> has_pivot_{};
  std::uint32_t rank_ = 0;
};

class SpanBasis {
 public:
  explicit SpanBasis(const FieldSpec& field) : field_(field) {}

  bool add(std::uint64_t code, std::uint64_t input_index) {
    Row row = decode(code, field_);
    for (std::uint32_t basis = 0; basis < rank_; ++basis) {
      const auto factor = row[pivot_columns_[basis]];
      if (factor != 0) {
        subtract_scaled(row, rows_[basis], factor, field_);
      }
    }

    std::uint32_t pivot = field_.width;
    for (std::uint32_t column = 0; column < field_.width; ++column) {
      if (row[column] != 0) {
        pivot = column;
        break;
      }
    }
    if (pivot == field_.width) {
      return false;
    }

    const auto inverse = field_.inverses[row[pivot]];
    for (std::uint32_t column = 0; column < field_.width; ++column) {
      row[column] = static_cast<std::uint32_t>(
          static_cast<std::uint64_t>(row[column]) * inverse % field_.prime);
    }
    for (std::uint32_t basis = 0; basis < rank_; ++basis) {
      const auto factor = rows_[basis][pivot];
      if (factor != 0) {
        subtract_scaled(rows_[basis], row, factor, field_);
      }
    }

    const auto position = static_cast<std::uint32_t>(std::lower_bound(
        pivot_columns_.begin(),
        pivot_columns_.begin() + rank_,
        pivot) - pivot_columns_.begin());
    for (std::uint32_t basis = rank_; basis > position; --basis) {
      rows_[basis] = rows_[basis - 1];
      pivot_columns_[basis] = pivot_columns_[basis - 1];
      pivot_indices_[basis] = pivot_indices_[basis - 1];
    }
    rows_[position] = row;
    pivot_columns_[position] = pivot;
    pivot_indices_[position] = input_index;
    ++rank_;
    return true;
  }

  std::uint64_t reduce(
      std::uint64_t code,
      std::uint32_t* coordinates) const {
    Row row = decode(code, field_);
    for (std::uint32_t basis = 0; basis < rank_; ++basis) {
      const auto factor = row[pivot_columns_[basis]];
      coordinates[basis] = factor;
      if (factor != 0) {
        subtract_scaled(row, rows_[basis], factor, field_);
      }
    }
    for (std::uint32_t basis = rank_; basis < field_.width; ++basis) {
      coordinates[basis] = 0;
    }
    return encode(row, field_);
  }

  std::uint32_t rank() const { return rank_; }
  const Row& row(std::uint32_t index) const { return rows_[index]; }
  std::uint32_t pivot_column(std::uint32_t index) const {
    return pivot_columns_[index];
  }
  std::uint64_t pivot_index(std::uint32_t index) const {
    return pivot_indices_[index];
  }

 private:
  const FieldSpec& field_;
  std::array<Row, kMaxWidth> rows_{};
  std::array<std::uint32_t, kMaxWidth> pivot_columns_{};
  std::array<std::uint64_t, kMaxWidth> pivot_indices_{};
  std::uint32_t rank_ = 0;
};

void validate_codes(
    const std::uint64_t* codes,
    std::size_t count,
    const FieldSpec& field,
    const char* kind) {
  for (std::size_t index = 0; index < count; ++index) {
    if (codes[index] >= field.space_size) {
      throw std::invalid_argument(kind);
    }
  }
}

void initialize_stats(
    fast_math_fp_span_stats* stats,
    std::uint64_t span_count,
    std::uint64_t point_count,
    std::uint64_t query_count) {
  *stats = {};
  stats->span_count = span_count;
  stats->point_count = point_count;
  stats->query_count = query_count;
}

}  // namespace

extern "C" {

int fast_math_fp_span_ranks_u64(
    const std::uint64_t* point_codes,
    std::size_t point_count,
    const std::uint64_t* span_offsets,
    std::size_t span_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint32_t* ranks,
    fast_math_fp_span_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    initialize_stats(stats, span_count, point_count, 0);
    if (span_offsets == nullptr) {
      throw std::invalid_argument("span offset pointer is null");
    }
    if (point_count != 0 && point_codes == nullptr) {
      throw std::invalid_argument("point pointer is null");
    }
    if (span_count != 0 && ranks == nullptr) {
      throw std::invalid_argument("rank pointer is null");
    }
    set_error(error_message, error_message_size, "");
    const FieldSpec field(prime, width);
    validate_codes(
        point_codes,
        point_count,
        field,
        "point code is outside the prime^width range");
    if (span_offsets[0] != 0) {
      throw std::invalid_argument("span offsets must start at zero");
    }
    for (std::size_t span = 0; span < span_count; ++span) {
      if (span_offsets[span] > span_offsets[span + 1] ||
          span_offsets[span + 1] > point_count) {
        throw std::invalid_argument(
            "span offsets must be nondecreasing and bounded by point_count");
      }
    }
    if (span_offsets[span_count] != point_count) {
      throw std::invalid_argument("final span offset must equal point_count");
    }

    const auto started = Clock::now();
    for (std::size_t span = 0; span < span_count; ++span) {
      SpanRank rank(field);
      const auto begin = static_cast<std::size_t>(span_offsets[span]);
      const auto end = static_cast<std::size_t>(span_offsets[span + 1]);
      for (std::size_t index = begin; index < end; ++index) {
        rank.add(point_codes[index]);
      }
      ranks[span] = rank.rank();
      stats->rank_sum += rank.rank();
    }
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

int fast_math_fp_point_span_u64(
    const std::uint64_t* point_codes,
    std::size_t point_count,
    const std::uint64_t* query_codes,
    std::size_t query_count,
    std::uint32_t prime,
    std::uint32_t width,
    std::uint64_t* pivot_indices,
    std::uint32_t* pivot_columns,
    std::uint64_t* reduced_basis_codes,
    std::uint8_t* independent_points,
    std::uint8_t* query_members,
    std::uint32_t* query_coordinates,
    std::uint64_t* query_quotient_codes,
    fast_math_fp_span_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (stats == nullptr) {
      throw std::invalid_argument("stats pointer is null");
    }
    initialize_stats(stats, 1, point_count, query_count);
    if (pivot_indices == nullptr || pivot_columns == nullptr ||
        reduced_basis_codes == nullptr) {
      throw std::invalid_argument("basis output pointer is null");
    }
    if (point_count != 0 &&
        (point_codes == nullptr || independent_points == nullptr)) {
      throw std::invalid_argument("point or independence pointer is null");
    }
    if (query_count != 0 &&
        (query_codes == nullptr || query_members == nullptr ||
         query_coordinates == nullptr || query_quotient_codes == nullptr)) {
      throw std::invalid_argument("query output pointer is null");
    }
    set_error(error_message, error_message_size, "");
    const FieldSpec field(prime, width);
    validate_codes(
        point_codes,
        point_count,
        field,
        "point code is outside the prime^width range");
    validate_codes(
        query_codes,
        query_count,
        field,
        "query code is outside the prime^width range");

    const auto started = Clock::now();
    SpanBasis basis(field);
    for (std::size_t index = 0; index < point_count; ++index) {
      independent_points[index] = basis.add(point_codes[index], index) ? 1 : 0;
    }
    const auto rank = basis.rank();
    stats->rank_sum = rank;
    for (std::uint32_t row = 0; row < rank; ++row) {
      pivot_indices[row] = basis.pivot_index(row);
      pivot_columns[row] = basis.pivot_column(row);
      reduced_basis_codes[row] = encode(basis.row(row), field);
    }
    for (std::uint32_t row = rank; row < width; ++row) {
      pivot_indices[row] = 0;
      pivot_columns[row] = 0;
      reduced_basis_codes[row] = 0;
    }

    for (std::size_t index = 0; index < query_count; ++index) {
      auto* coordinates =
          query_coordinates + index * static_cast<std::size_t>(width);
      const auto quotient = basis.reduce(query_codes[index], coordinates);
      query_quotient_codes[index] = quotient;
      query_members[index] = quotient == 0 ? 1 : 0;
    }
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

}  // extern "C"
