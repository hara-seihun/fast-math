#include "fast_math.h"
#include "modulus_u32.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Entry {
  std::uint32_t column;
  std::uint32_t value;
};

using Row = std::vector<Entry>;

struct BasisRow {
  std::size_t offset;
  std::size_t size;
};

enum class PeelKind : std::uint8_t {
  row,
  column,
};

struct PeelItem {
  PeelKind kind;
  std::size_t index;
};

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

using Modulus = fast_math_internal::PrimeModulusU32;

constexpr std::size_t kMaximumBlockRows = 16;

struct SmallBasisVector {
  std::array<std::uint32_t, kMaximumBlockRows> original{};
  std::array<std::uint32_t, kMaximumBlockRows> echelon{};
  std::uint32_t column = 0;
  std::uint32_t pivot_row = 0;
  bool replaceable = false;
};

struct LocalColoop {
  std::uint32_t column;
  std::array<std::uint32_t, kMaximumBlockRows> coefficients{};
};

std::vector<std::uint32_t> invert_small_square(
    std::vector<std::uint32_t> matrix,
    std::size_t order,
    const Modulus& modulus) {
  std::vector<std::uint32_t> inverse(order * order, 0);
  for (std::size_t index = 0; index < order; ++index) {
    inverse[index * order + index] = 1;
  }
  for (std::size_t column = 0; column < order; ++column) {
    auto pivot = column;
    while (pivot < order &&
           matrix[pivot * order + column] == 0) {
      ++pivot;
    }
    if (pivot == order) {
      throw std::logic_error("local coloop basis minor is singular");
    }
    if (pivot != column) {
      for (std::size_t entry = 0; entry < order; ++entry) {
        std::swap(
            matrix[column * order + entry],
            matrix[pivot * order + entry]);
        std::swap(
            inverse[column * order + entry],
            inverse[pivot * order + entry]);
      }
    }
    const auto pivot_value =
        matrix[column * order + column];
    const auto reciprocal = pivot_value == 1
        ? 1
        : modulus.inverse(pivot_value);
    if (reciprocal != 1) {
      for (std::size_t entry = 0; entry < order; ++entry) {
        matrix[column * order + entry] = modulus.multiply(
            matrix[column * order + entry],
            reciprocal);
        inverse[column * order + entry] = modulus.multiply(
            inverse[column * order + entry],
            reciprocal);
      }
    }
    for (std::size_t row = 0; row < order; ++row) {
      if (row == column) {
        continue;
      }
      const auto factor = matrix[row * order + column];
      if (factor == 0) {
        continue;
      }
      for (std::size_t entry = 0; entry < order; ++entry) {
        matrix[row * order + entry] = modulus.subtract(
            matrix[row * order + entry],
            modulus.multiply(
                factor,
                matrix[column * order + entry]));
        inverse[row * order + entry] = modulus.subtract(
            inverse[row * order + entry],
            modulus.multiply(
                factor,
                inverse[column * order + entry]));
      }
    }
  }
  return inverse;
}

std::vector<LocalColoop> find_local_coloops(
    const std::vector<std::uint32_t>& block_columns,
    const std::vector<std::uint32_t>& block_values,
    std::size_t begin,
    std::size_t end,
    std::uint32_t row_block_size,
    const std::vector<std::uint8_t>& active_columns,
    const Modulus& modulus) {
  std::array<SmallBasisVector, kMaximumBlockRows> basis{};
  std::size_t rank = 0;
  for (std::size_t entry = begin; entry < end; ++entry) {
    const auto column = block_columns[entry];
    if (!active_columns[column]) {
      continue;
    }
    std::array<std::uint32_t, kMaximumBlockRows> working{};
    const auto value_offset =
        entry * static_cast<std::size_t>(row_block_size);
    for (std::size_t row = 0; row < row_block_size; ++row) {
      working[row] = block_values[value_offset + row];
    }

    for (std::size_t index = 0; index < rank; ++index) {
      const auto factor = working[basis[index].pivot_row];
      if (factor == 0) {
        continue;
      }
      for (std::size_t row = 0; row < row_block_size; ++row) {
        working[row] = modulus.subtract(
            working[row],
            modulus.multiply(factor, basis[index].echelon[row]));
      }
    }

    std::size_t pivot_row = 0;
    while (pivot_row < row_block_size &&
           working[pivot_row] == 0) {
      ++pivot_row;
    }
    if (pivot_row == row_block_size) {
      continue;
    }

    const auto leading = working[pivot_row];
    const auto reciprocal =
        leading == 1 ? 1 : modulus.inverse(leading);
    auto& added = basis[rank];
    added.column = column;
    added.pivot_row = static_cast<std::uint32_t>(pivot_row);
    for (std::size_t row = 0; row < row_block_size; ++row) {
      added.original[row] = block_values[value_offset + row];
      added.echelon[row] = reciprocal == 1
          ? working[row]
          : modulus.multiply(working[row], reciprocal);
    }
    ++rank;
  }

  if (rank == 0) {
    return {};
  }
  std::vector<std::uint32_t> basis_minor(rank * rank);
  for (std::size_t row = 0; row < rank; ++row) {
    const auto original_row = basis[row].pivot_row;
    for (std::size_t column = 0; column < rank; ++column) {
      basis_minor[row * rank + column] =
          basis[column].original[original_row];
    }
  }
  const auto inverse =
      invert_small_square(std::move(basis_minor), rank, modulus);

  for (std::size_t entry = begin; entry < end; ++entry) {
    const auto column = block_columns[entry];
    if (!active_columns[column]) {
      continue;
    }
    bool is_basis_column = false;
    for (std::size_t index = 0; index < rank; ++index) {
      if (basis[index].column == column) {
        is_basis_column = true;
        break;
      }
    }
    if (is_basis_column) {
      continue;
    }
    const auto value_offset =
        entry * static_cast<std::size_t>(row_block_size);
    for (std::size_t index = 0; index < rank; ++index) {
      std::uint32_t coordinate = 0;
      for (std::size_t selected = 0;
           selected < rank;
           ++selected) {
        const auto product = modulus.multiply(
            inverse[index * rank + selected],
            block_values[
                value_offset + basis[selected].pivot_row]);
        coordinate = modulus.add(coordinate, product);
      }
      if (coordinate != 0) {
        basis[index].replaceable = true;
      }
    }
  }

  std::vector<LocalColoop> coloops;
  coloops.reserve(rank);
  for (std::size_t index = 0; index < rank; ++index) {
    if (basis[index].replaceable) {
      continue;
    }
    LocalColoop coloop{};
    coloop.column = basis[index].column;
    for (std::size_t coordinate = 0;
         coordinate < rank;
         ++coordinate) {
      coloop.coefficients[basis[coordinate].pivot_row] =
          inverse[index * rank + coordinate];
    }
    coloops.push_back(coloop);
  }
  return coloops;
}

std::uint64_t multiply_mod_u64(
    std::uint64_t left,
    std::uint64_t right,
    std::uint64_t modulus) {
#if defined(__SIZEOF_INT128__)
  return static_cast<std::uint64_t>(
      static_cast<unsigned __int128>(left) * right % modulus);
#else
  std::uint64_t result = 0;
  while (right != 0) {
    if ((right & 1U) != 0) {
      result = (result + left) % modulus;
    }
    left = (2 * left) % modulus;
    right >>= 1U;
  }
  return result;
#endif
}

std::uint64_t power_mod_u64(
    std::uint64_t base,
    std::uint64_t exponent,
    std::uint64_t modulus) {
  std::uint64_t result = 1;
  while (exponent != 0) {
    if ((exponent & 1U) != 0) {
      result = multiply_mod_u64(result, base, modulus);
    }
    exponent >>= 1U;
    if (exponent != 0) {
      base = multiply_mod_u64(base, base, modulus);
    }
  }
  return result;
}

bool is_prime_u32(std::uint32_t value) {
  if (value < 2) {
    return false;
  }
  for (const std::uint32_t small : {2U, 3U, 5U, 7U, 11U, 13U, 17U, 19U,
                                    23U, 29U, 31U, 37U}) {
    if (value == small) {
      return true;
    }
    if (value % small == 0) {
      return false;
    }
  }

  std::uint64_t odd_part = static_cast<std::uint64_t>(value) - 1;
  unsigned shifts = 0;
  while ((odd_part & 1U) == 0) {
    odd_part >>= 1U;
    ++shifts;
  }
  for (const std::uint32_t base : {2U, 3U, 5U, 7U, 11U, 13U, 17U}) {
    if (base >= value) {
      continue;
    }
    auto witness = power_mod_u64(base, odd_part, value);
    if (witness == 1 || witness == value - 1) {
      continue;
    }
    bool composite = true;
    for (unsigned round = 1; round < shifts; ++round) {
      witness = multiply_mod_u64(witness, witness, value);
      if (witness == value - 1) {
        composite = false;
        break;
      }
    }
    if (composite) {
      return false;
    }
  }
  return true;
}

std::size_t first_occupied_column(
    const std::vector<std::uint64_t>& occupancy,
    std::size_t first_word) {
  for (std::size_t word = first_word;
       word < occupancy.size();
       ++word) {
    if (occupancy[word] != 0) {
      return word * 64 +
          std::countr_zero(occupancy[word]);
    }
  }
  return std::numeric_limits<std::size_t>::max();
}

}  // namespace

extern "C" {

int fast_math_sparse_rank_mod_u32(
    const std::uint64_t* row_offsets,
    const std::uint32_t* column_indices,
    const std::uint32_t* values,
    std::size_t row_count,
    std::size_t column_count,
    std::size_t nonzero_count,
    std::uint32_t prime,
    std::size_t target_rank,
    std::uint64_t* pivot_rows,
    std::uint32_t* pivot_columns,
    std::size_t pivot_capacity,
    fast_math_sparse_rank_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (row_offsets == nullptr || stats == nullptr) {
      throw std::invalid_argument("sparse rank pointer is null");
    }
    if (nonzero_count != 0 &&
        (column_indices == nullptr || values == nullptr)) {
      throw std::invalid_argument("sparse rank entry pointer is null");
    }
    if (row_count == 0 || column_count == 0) {
      throw std::invalid_argument(
          "sparse rank matrix dimensions must be positive");
    }
    if (column_count > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("sparse rank column count exceeds uint32");
    }
    if (!is_prime_u32(prime)) {
      throw std::invalid_argument("sparse rank modulus must be prime");
    }
    const auto maximum_rank = std::min(row_count, column_count);
    const auto requested_rank =
        target_rank == 0 ? maximum_rank : target_rank;
    if (requested_rank > maximum_rank) {
      throw std::invalid_argument("sparse rank target exceeds matrix dimensions");
    }
    if ((pivot_rows == nullptr) != (pivot_columns == nullptr)) {
      throw std::invalid_argument(
          "sparse rank pivot outputs must both be null or nonnull");
    }
    if (pivot_rows != nullptr && pivot_capacity < requested_rank) {
      throw std::invalid_argument("sparse rank pivot capacity is too small");
    }
    if (row_offsets[0] != 0 ||
        row_offsets[row_count] != nonzero_count) {
      throw std::invalid_argument("invalid sparse rank row offsets");
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const Modulus modulus(prime);
    std::vector<std::uint32_t> reduced_values(nonzero_count);
    std::vector<std::size_t> row_sizes(row_count, 0);
    std::vector<std::size_t> column_degree(column_count, 0);
    std::uint64_t retained_nonzeros = 0;

    for (std::size_t row = 0; row < row_count; ++row) {
      const auto begin = row_offsets[row];
      const auto end = row_offsets[row + 1];
      if (begin > end || end > nonzero_count) {
        throw std::invalid_argument("sparse rank row offsets are not monotone");
      }
      std::uint32_t previous_column = 0;
      bool have_previous = false;
      for (auto offset = begin; offset < end; ++offset) {
        const auto column = column_indices[offset];
        if (column >= column_count) {
          throw std::invalid_argument(
              "sparse rank column index is out of range");
        }
        if (have_previous && column <= previous_column) {
          throw std::invalid_argument(
              "sparse rank columns must increase within each row");
        }
        previous_column = column;
        have_previous = true;
        const auto reduced = modulus.reduce(values[offset]);
        reduced_values[offset] = reduced;
        if (reduced == 0) {
          continue;
        }
        ++row_sizes[row];
        ++column_degree[column];
        ++retained_nonzeros;
      }
    }

    std::vector<std::size_t> column_offsets(column_count + 1, 0);
    for (std::size_t column = 0; column < column_count; ++column) {
      column_offsets[column + 1] =
          column_offsets[column] + column_degree[column];
    }
    std::vector<std::size_t> column_cursor = column_offsets;
    std::vector<std::size_t> column_rows(
        static_cast<std::size_t>(retained_nonzeros));
    for (std::size_t row = 0; row < row_count; ++row) {
      for (auto offset = row_offsets[row];
           offset < row_offsets[row + 1];
           ++offset) {
        if (reduced_values[offset] == 0) {
          continue;
        }
        const auto column = column_indices[offset];
        column_rows[column_cursor[column]++] = row;
      }
    }
    column_cursor.clear();
    column_cursor.shrink_to_fit();

    std::vector<std::uint8_t> row_active(row_count, 0);
    std::vector<std::uint8_t> column_active(column_count, 0);
    std::size_t initial_active_rows = 0;
    for (std::size_t row = 0; row < row_count; ++row) {
      if (row_sizes[row] != 0) {
        row_active[row] = 1;
        ++initial_active_rows;
      }
    }
    for (std::size_t column = 0; column < column_count; ++column) {
      column_active[column] =
          static_cast<std::uint8_t>(column_degree[column] != 0);
    }

    std::deque<PeelItem> peel_queue;
    for (std::size_t row = 0; row < row_count; ++row) {
      if (row_sizes[row] == 1) {
        peel_queue.push_back({PeelKind::row, row});
      }
    }
    for (std::size_t column = 0; column < column_count; ++column) {
      if (column_degree[column] == 1) {
        peel_queue.push_back({PeelKind::column, column});
      }
    }

    const auto missing = std::numeric_limits<std::size_t>::max();
    const auto sole_active_column =
        [&](std::size_t row) -> std::size_t {
      for (auto offset = row_offsets[row];
           offset < row_offsets[row + 1];
           ++offset) {
        const auto column = column_indices[offset];
        if (reduced_values[offset] != 0 && column_active[column]) {
          return column;
        }
      }
      return missing;
    };
    const auto sole_active_row =
        [&](std::size_t column) -> std::size_t {
      for (auto offset = column_offsets[column];
           offset < column_offsets[column + 1];
           ++offset) {
        const auto row = column_rows[offset];
        if (row_active[row]) {
          return row;
        }
      }
      return missing;
    };

    std::size_t rank = 0;
    std::size_t peeled = 0;
    std::size_t residual_nonzeros =
        static_cast<std::size_t>(retained_nonzeros);
    while (!peel_queue.empty() && rank < requested_rank) {
      const auto item = peel_queue.front();
      peel_queue.pop_front();
      std::size_t row = missing;
      std::size_t column = missing;
      if (item.kind == PeelKind::row) {
        row = item.index;
        if (!row_active[row] || row_sizes[row] != 1) {
          continue;
        }
        column = sole_active_column(row);
      } else {
        column = item.index;
        if (!column_active[column] || column_degree[column] != 1) {
          continue;
        }
        row = sole_active_row(column);
      }
      if (row == missing || column == missing ||
          !row_active[row] || !column_active[column]) {
        continue;
      }

      const auto removed_edges =
          row_sizes[row] + column_degree[column] - 1;
      if (removed_edges > residual_nonzeros) {
        throw std::logic_error("sparse rank peel accounting underflow");
      }
      residual_nonzeros -= removed_edges;
      row_active[row] = 0;
      column_active[column] = 0;
      if (pivot_rows != nullptr) {
        pivot_rows[rank] = row;
        pivot_columns[rank] =
            static_cast<std::uint32_t>(column);
      }
      ++rank;
      ++peeled;

      row_sizes[row] = 0;
      column_degree[column] = 0;
      for (auto offset = row_offsets[row];
           offset < row_offsets[row + 1];
           ++offset) {
        const auto neighbor = column_indices[offset];
        if (reduced_values[offset] == 0 ||
            !column_active[neighbor]) {
          continue;
        }
        if (--column_degree[neighbor] == 1) {
          peel_queue.push_back({PeelKind::column, neighbor});
        }
      }
      for (auto offset = column_offsets[column];
           offset < column_offsets[column + 1];
           ++offset) {
        const auto neighbor = column_rows[offset];
        if (!row_active[neighbor]) {
          continue;
        }
        if (--row_sizes[neighbor] == 1) {
          peel_queue.push_back({PeelKind::row, neighbor});
        }
      }
    }

    const auto preprocessing_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    std::vector<std::uint32_t> column_order;
    column_order.reserve(column_count);
    for (std::size_t column = 0; column < column_count; ++column) {
      if (column_active[column] && column_degree[column] != 0) {
        column_order.push_back(static_cast<std::uint32_t>(column));
      }
    }
    std::sort(
        column_order.begin(),
        column_order.end(),
        [&](std::uint32_t left, std::uint32_t right) {
          if (column_degree[left] != column_degree[right]) {
            return column_degree[left] < column_degree[right];
          }
          return left < right;
        });
    std::vector<std::uint32_t> reordered_column(
        column_count,
        std::numeric_limits<std::uint32_t>::max());
    for (std::size_t position = 0;
         position < column_order.size();
         ++position) {
      reordered_column[column_order[position]] =
          static_cast<std::uint32_t>(position);
    }

    std::vector<std::size_t> active_rows;
    active_rows.reserve(row_count);
    // Static Markowitz cost predicts pivot fill without changing exactness.
    std::vector<std::uint32_t> first_reordered_column(
        row_count,
        std::numeric_limits<std::uint32_t>::max());
    std::vector<std::uint64_t> row_markowitz_cost(
        row_count,
        std::numeric_limits<std::uint64_t>::max());
    for (std::size_t row = 0; row < row_count; ++row) {
      if (!row_active[row] || row_sizes[row] == 0) {
        continue;
      }
      for (auto offset = row_offsets[row];
           offset < row_offsets[row + 1];
           ++offset) {
        const auto column = column_indices[offset];
        if (reduced_values[offset] == 0 ||
            !column_active[column]) {
          continue;
        }
        first_reordered_column[row] = std::min(
            first_reordered_column[row],
            reordered_column[column]);
      }
      const auto pivot_column =
          column_order[first_reordered_column[row]];
      row_markowitz_cost[row] =
          static_cast<std::uint64_t>(row_sizes[row] - 1) *
          static_cast<std::uint64_t>(column_degree[pivot_column] - 1);
      active_rows.push_back(row);
    }
    std::sort(
        active_rows.begin(),
        active_rows.end(),
        [&](std::size_t left, std::size_t right) {
          if (row_markowitz_cost[left] != row_markowitz_cost[right]) {
            return row_markowitz_cost[left] < row_markowitz_cost[right];
          }
          if (row_sizes[left] != row_sizes[right]) {
            return row_sizes[left] < row_sizes[right];
          }
          if (first_reordered_column[left] !=
              first_reordered_column[right]) {
            return first_reordered_column[left] <
                first_reordered_column[right];
          }
          return left < right;
        });

    std::vector<BasisRow> basis(column_order.size(), {0, 0});
    Row basis_entries;
    basis_entries.reserve(residual_nonzeros);
    std::vector<std::uint32_t> working_values(column_order.size());
    std::vector<std::uint64_t> occupancy(
        (column_order.size() + 63) / 64,
        0);
    std::size_t processed = peeled;
    std::size_t dependent = 0;
    std::size_t maximum_working_size = 0;
    std::uint64_t elimination_steps = 0;

    for (const auto row_id : active_rows) {
      if (rank >= requested_rank) {
        break;
      }
      std::size_t working_size = 0;
      for (auto offset = row_offsets[row_id];
           offset < row_offsets[row_id + 1];
           ++offset) {
        const auto value = reduced_values[offset];
        const auto original_column = column_indices[offset];
        if (value == 0 || !column_active[original_column]) {
          continue;
        }
        const auto column =
            reordered_column[original_column];
        working_values[column] = value;
        occupancy[column / 64] |=
            std::uint64_t{1} << (column % 64);
        ++working_size;
      }
      bool independent = false;
      maximum_working_size =
          std::max(maximum_working_size, working_size);
      auto pivot_column = first_occupied_column(occupancy, 0);

      while (pivot_column !=
             std::numeric_limits<std::size_t>::max()) {
        if (basis[pivot_column].size == 0) {
          const auto leading_value =
              working_values[pivot_column];
          const auto inverse = leading_value == 1
              ? 1
              : modulus.inverse(leading_value);
          const auto basis_offset = basis_entries.size();
          for (std::size_t word = pivot_column / 64;
               word < occupancy.size();
               ++word) {
            auto active = occupancy[word];
            while (active != 0) {
              const auto bit = std::countr_zero(active);
              const auto column = word * 64 + bit;
              const auto value = inverse == 1
                  ? working_values[column]
                  : modulus.multiply(
                      working_values[column],
                      inverse);
              basis_entries.push_back({
                  static_cast<std::uint32_t>(column),
                  value,
              });
              active &= active - 1;
            }
          }
          if (pivot_rows != nullptr) {
            pivot_rows[rank] = row_id;
            pivot_columns[rank] = column_order[pivot_column];
          }
          basis[pivot_column] = {
              basis_offset,
              basis_entries.size() - basis_offset,
          };
          std::fill(occupancy.begin(), occupancy.end(), 0);
          ++rank;
          independent = true;
          break;
        }

        const auto factor = working_values[pivot_column];
        occupancy[pivot_column / 64] &=
            ~(std::uint64_t{1} << (pivot_column % 64));
        --working_size;
        const auto pivot = basis[pivot_column];
        const auto apply_product =
            [&](const Entry& pivot_entry, std::uint32_t product) {
          const auto column = pivot_entry.column;
          const auto word = column / 64;
          const auto mask =
              std::uint64_t{1} << (column % 64);
          if ((occupancy[word] & mask) == 0) {
            working_values[column] =
                modulus.negate_nonzero(product);
            occupancy[word] |= mask;
            ++working_size;
            return;
          }
          const auto value = modulus.subtract(
              working_values[column],
              product);
          working_values[column] = value;
          if (value == 0) {
            occupancy[word] &= ~mask;
            --working_size;
          }
        };
        if (factor == 1) {
          for (std::size_t index = 1;
               index < pivot.size;
               ++index) {
            const auto& pivot_entry =
                basis_entries[pivot.offset + index];
            apply_product(pivot_entry, pivot_entry.value);
          }
        } else {
          for (std::size_t index = 1;
               index < pivot.size;
               ++index) {
            const auto& pivot_entry =
                basis_entries[pivot.offset + index];
            apply_product(
                pivot_entry,
                modulus.multiply(factor, pivot_entry.value));
          }
        }
        ++elimination_steps;
        maximum_working_size =
            std::max(maximum_working_size, working_size);
        pivot_column = first_occupied_column(
            occupancy,
            pivot_column / 64);
      }
      if (!independent) {
        ++dependent;
      }
      ++processed;
    }

    std::size_t maximum_basis_size = 0;
    for (const auto& row : basis) {
      maximum_basis_size = std::max(maximum_basis_size, row.size);
    }

    stats->row_count = row_count;
    stats->column_count = column_count;
    stats->input_nonzeros = retained_nonzeros;
    stats->active_rows = initial_active_rows;
    stats->processed_rows = processed;
    stats->dependent_rows = dependent;
    stats->rank = rank;
    stats->elimination_steps = elimination_steps;
    stats->basis_nonzeros = basis_entries.size();
    stats->maximum_basis_size = maximum_basis_size;
    stats->maximum_working_size = maximum_working_size;
    stats->peeled_pivots = peeled;
    stats->residual_rows = active_rows.size();
    stats->residual_columns = column_order.size();
    stats->residual_nonzeros = residual_nonzeros;
    stats->prime = prime;
    stats->target_reached =
        static_cast<std::uint8_t>(rank >= requested_rank);
    stats->preprocessing_seconds = preprocessing_seconds;
    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(error_message, error_message_size, "unknown sparse rank error");
    return 2;
  }
}

int fast_math_sparse_block_coloops_mod_u32(
    const std::uint64_t* row_offsets,
    const std::uint32_t* column_indices,
    const std::uint32_t* values,
    std::size_t row_count,
    std::size_t column_count,
    std::size_t nonzero_count,
    std::uint32_t prime,
    std::uint32_t row_block_size,
    std::uint8_t* residual_columns,
    std::size_t residual_capacity,
    std::uint32_t* removed_columns,
    std::uint64_t* certificate_row_starts,
    std::uint32_t* certificate_coefficients,
    std::size_t removed_capacity,
    fast_math_sparse_block_coloop_stats* stats,
    char* error_message,
    std::size_t error_message_size) {
  try {
    if (row_offsets == nullptr || stats == nullptr) {
      throw std::invalid_argument(
          "sparse block coloop pointer is null");
    }
    if (nonzero_count != 0 &&
        (column_indices == nullptr || values == nullptr)) {
      throw std::invalid_argument(
          "sparse block coloop entry pointer is null");
    }
    if (row_count == 0 || column_count == 0) {
      throw std::invalid_argument(
          "sparse block coloop matrix dimensions must be positive");
    }
    if (column_count > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error(
          "sparse block coloop column count exceeds uint32");
    }
    if (!is_prime_u32(prime)) {
      throw std::invalid_argument(
          "sparse block coloop modulus must be prime");
    }
    if (row_block_size == 0 ||
        row_block_size > kMaximumBlockRows) {
      throw std::invalid_argument(
          "sparse block coloop row block size must be in [1, 16]");
    }
    if (residual_columns != nullptr &&
        residual_capacity < column_count) {
      throw std::invalid_argument(
          "sparse block coloop residual capacity is too small");
    }
    const auto any_certificate_output =
        removed_columns != nullptr ||
        certificate_row_starts != nullptr ||
        certificate_coefficients != nullptr;
    const auto all_certificate_outputs =
        removed_columns != nullptr &&
        certificate_row_starts != nullptr &&
        certificate_coefficients != nullptr;
    if (any_certificate_output && !all_certificate_outputs) {
      throw std::invalid_argument(
          "sparse block coloop certificate outputs must all be null or nonnull");
    }
    if (all_certificate_outputs && removed_capacity < column_count) {
      throw std::invalid_argument(
          "sparse block coloop certificate capacity is too small");
    }
    if (row_offsets[0] != 0 ||
        row_offsets[row_count] != nonzero_count) {
      throw std::invalid_argument(
          "invalid sparse block coloop row offsets");
    }
    if (nonzero_count >
        std::numeric_limits<std::size_t>::max() / row_block_size) {
      throw std::overflow_error(
          "sparse block coloop workspace is too large");
    }

    set_error(error_message, error_message_size, "");
    *stats = {};
    const auto started = Clock::now();
    const Modulus modulus(prime);
    std::uint64_t retained_nonzeros = 0;
    for (std::size_t row = 0; row < row_count; ++row) {
      const auto begin = row_offsets[row];
      const auto end = row_offsets[row + 1];
      if (begin > end || end > nonzero_count) {
        throw std::invalid_argument(
            "sparse block coloop row offsets are not monotone");
      }
      std::uint32_t previous_column = 0;
      bool have_previous = false;
      for (auto offset = begin; offset < end; ++offset) {
        const auto column = column_indices[offset];
        if (column >= column_count) {
          throw std::invalid_argument(
              "sparse block coloop column index is out of range");
        }
        if (have_previous && column <= previous_column) {
          throw std::invalid_argument(
              "sparse block coloop columns must increase within each row");
        }
        previous_column = column;
        have_previous = true;
        retained_nonzeros +=
            static_cast<std::uint64_t>(
                modulus.reduce(values[offset]) != 0);
      }
    }

    const auto block_count =
        (row_count + row_block_size - 1) / row_block_size;
    std::vector<std::size_t> block_offsets(block_count + 1, 0);
    std::vector<std::uint32_t> block_columns;
    std::vector<std::uint32_t> block_values;
    block_columns.reserve(nonzero_count);
    block_values.reserve(nonzero_count * row_block_size);
    std::vector<std::size_t> column_block_degree(column_count, 0);
    std::size_t maximum_block_columns = 0;

    for (std::size_t block = 0; block < block_count; ++block) {
      const auto first_row =
          block * static_cast<std::size_t>(row_block_size);
      const auto block_rows = std::min<std::size_t>(
          row_block_size,
          row_count - first_row);
      std::array<std::uint64_t, kMaximumBlockRows> cursors{};
      std::array<std::uint64_t, kMaximumBlockRows> ends{};
      for (std::size_t local_row = 0;
           local_row < block_rows;
           ++local_row) {
        cursors[local_row] = row_offsets[first_row + local_row];
        ends[local_row] = row_offsets[first_row + local_row + 1];
      }

      while (true) {
        auto next_column = std::numeric_limits<std::uint32_t>::max();
        for (std::size_t local_row = 0;
             local_row < block_rows;
             ++local_row) {
          if (cursors[local_row] < ends[local_row]) {
            next_column = std::min(
                next_column,
                column_indices[cursors[local_row]]);
          }
        }
        if (next_column ==
            std::numeric_limits<std::uint32_t>::max()) {
          break;
        }

        std::array<std::uint32_t, kMaximumBlockRows> local_values{};
        bool retained = false;
        for (std::size_t local_row = 0;
             local_row < block_rows;
             ++local_row) {
          if (cursors[local_row] == ends[local_row] ||
              column_indices[cursors[local_row]] != next_column) {
            continue;
          }
          const auto reduced =
              modulus.reduce(values[cursors[local_row]]);
          local_values[local_row] = reduced;
          retained = retained || reduced != 0;
          ++cursors[local_row];
        }
        if (!retained) {
          continue;
        }
        block_columns.push_back(next_column);
        block_values.insert(
            block_values.end(),
            local_values.begin(),
            local_values.begin() + row_block_size);
        ++column_block_degree[next_column];
      }
      block_offsets[block + 1] = block_columns.size();
      maximum_block_columns = std::max(
          maximum_block_columns,
          block_offsets[block + 1] - block_offsets[block]);
    }

    std::vector<std::size_t> column_block_offsets(
        column_count + 1,
        0);
    for (std::size_t column = 0; column < column_count; ++column) {
      column_block_offsets[column + 1] =
          column_block_offsets[column] + column_block_degree[column];
    }
    std::vector<std::size_t> column_block_cursor =
        column_block_offsets;
    std::vector<std::size_t> column_blocks(block_columns.size());
    for (std::size_t block = 0; block < block_count; ++block) {
      for (auto entry = block_offsets[block];
           entry < block_offsets[block + 1];
           ++entry) {
        const auto column = block_columns[entry];
        column_blocks[column_block_cursor[column]++] = block;
      }
    }
    column_block_cursor.clear();
    column_block_cursor.shrink_to_fit();

    std::vector<std::uint8_t> active_columns(column_count, 0);
    std::size_t initial_active_columns = 0;
    for (std::size_t column = 0; column < column_count; ++column) {
      if (column_block_degree[column] != 0) {
        active_columns[column] = 1;
        ++initial_active_columns;
      }
    }

    std::deque<std::size_t> queue;
    std::vector<std::uint8_t> queued(block_count, 1);
    for (std::size_t block = 0; block < block_count; ++block) {
      queue.push_back(block);
    }
    std::size_t removed_count = 0;
    std::size_t blocks_processed = 0;
    while (!queue.empty()) {
      const auto block = queue.front();
      queue.pop_front();
      queued[block] = 0;
      ++blocks_processed;
      const auto coloops = find_local_coloops(
          block_columns,
          block_values,
          block_offsets[block],
          block_offsets[block + 1],
          row_block_size,
          active_columns,
          modulus);
      for (const auto& coloop : coloops) {
        const auto column = coloop.column;
        if (!active_columns[column]) {
          continue;
        }
        active_columns[column] = 0;
        if (all_certificate_outputs) {
          removed_columns[removed_count] = column;
          certificate_row_starts[removed_count] =
              block * static_cast<std::size_t>(row_block_size);
          const auto coefficient_offset =
              removed_count * static_cast<std::size_t>(row_block_size);
          std::copy_n(
              coloop.coefficients.begin(),
              row_block_size,
              certificate_coefficients + coefficient_offset);
        }
        ++removed_count;
        for (auto offset = column_block_offsets[column];
             offset < column_block_offsets[column + 1];
             ++offset) {
          const auto affected = column_blocks[offset];
          if (!queued[affected]) {
            queue.push_back(affected);
            queued[affected] = 1;
          }
        }
      }
    }

    if (residual_columns != nullptr) {
      std::copy(
          active_columns.begin(),
          active_columns.end(),
          residual_columns);
    }
    const auto residual_count = static_cast<std::size_t>(
        std::count(active_columns.begin(), active_columns.end(), 1));
    stats->row_count = row_count;
    stats->column_count = column_count;
    stats->input_nonzeros = retained_nonzeros;
    stats->block_count = block_count;
    stats->block_incidences = block_columns.size();
    stats->active_columns = initial_active_columns;
    stats->removed_columns = removed_count;
    stats->residual_columns = residual_count;
    stats->blocks_processed = blocks_processed;
    stats->maximum_block_columns = maximum_block_columns;
    stats->row_block_size = row_block_size;
    stats->prime = prime;
    stats->elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - started).count();
    return 0;
  } catch (const std::exception& error) {
    set_error(error_message, error_message_size, error.what());
    return 1;
  } catch (...) {
    set_error(
        error_message,
        error_message_size,
        "unknown sparse block coloop error");
    return 2;
  }
}

}  // extern "C"
