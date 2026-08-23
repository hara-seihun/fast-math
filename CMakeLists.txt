cmake_minimum_required(VERSION 3.20)

project(fast_math VERSION 0.9.0 LANGUAGES C CXX)

option(
  FAST_MATH_NATIVE_ARCH
  "Tune generated code for the machine performing the build"
  ON
)

option(
  FAST_MATH_USE_COMMONCRYPTO
  "Use Apple's platform SHA-256 implementation when available"
  ON
)

option(
  FAST_MATH_USE_NAUTY
  "Build the optional nauty-backed canonical graph API when available"
  ON
)

option(
  FAST_MATH_VERIFY_MODULAR_ARITHMETIC
  "Cross-check sparse finite-field arithmetic against integer remainder"
  OFF
)

find_package(Threads REQUIRED)
find_package(PkgConfig QUIET)
include(CTest)
include(FetchContent)

option(
  FAST_MATH_USE_FORKUNION
  "Use a persistent low-latency ForkUnion scheduler"
  ON
)

if(FAST_MATH_USE_FORKUNION)
  FetchContent_Declare(
    forkunion
    URL
      https://github.com/ashvardanian/ForkUnion/archive/9f1d91c3529fc735f0ab60cbc895f449c7139d28.tar.gz
    URL_HASH
      SHA256=149e7a443e50f157847c8614596341e12c9c98c6ba8287dc6996e5554ebd3bcf
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR include
  )
  FetchContent_MakeAvailable(forkunion)
  add_library(fast_math_forkunion INTERFACE)
  target_include_directories(
    fast_math_forkunion
    INTERFACE
      ${forkunion_SOURCE_DIR}/include
  )
endif()

if(PkgConfig_FOUND)
  pkg_check_modules(FLINT QUIET IMPORTED_TARGET flint)
  if(FAST_MATH_USE_NAUTY)
    pkg_check_modules(NAUTY_TLS QUIET IMPORTED_TARGET libnautyT)
    if(NOT TARGET PkgConfig::NAUTY_TLS)
      pkg_check_modules(NAUTY QUIET IMPORTED_TARGET libnauty)
    endif()
  endif()
endif()

set(FAST_MATH_CORE_SOURCES
  cpp/src/base_p.cpp
  cpp/src/cnf.cpp
  cpp/src/colex.cpp
  cpp/src/digest.cpp
  cpp/src/graph64.cpp
  cpp/src/groups.cpp
  cpp/src/large_graph.cpp
  cpp/src/modular.cpp
  cpp/src/segments.cpp
  cpp/src/sparse_rank.cpp
  cpp/src/sparse_rank_batch.cpp
  cpp/src/subset_action.cpp
  cpp/src/taylor.cpp
)

set(FAST_MATH_DOMAIN_SOURCES
  cpp/src/adaptive.cpp
  cpp/src/shift_gates.cpp
  cpp/src/ci.cpp
  cpp/src/elliptic.cpp
  cpp/src/filon.cpp
  cpp/src/packing.cpp
  cpp/src/union.cpp
)

set(FAST_MATH_LAMBDA_SOURCES
  cpp/src/accumulate.cpp
  cpp/src/inverse.cpp
  cpp/src/moments.cpp
  cpp/src/two_level.cpp
)

add_library(fast_math SHARED
  ${FAST_MATH_CORE_SOURCES}
  ${FAST_MATH_DOMAIN_SOURCES}
  ${FAST_MATH_LAMBDA_SOURCES}
)

target_include_directories(fast_math
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src
)

target_compile_features(fast_math PUBLIC cxx_std_20)
target_link_libraries(fast_math PRIVATE Threads::Threads)

if(TARGET PkgConfig::NAUTY_TLS)
  target_sources(fast_math PRIVATE cpp/src/canonical_graph.cpp)
  target_link_libraries(fast_math PRIVATE PkgConfig::NAUTY_TLS)
  target_compile_definitions(fast_math PUBLIC FAST_MATH_HAVE_NAUTY=1)
elseif(TARGET PkgConfig::NAUTY)
  target_sources(fast_math PRIVATE cpp/src/canonical_graph.cpp)
  target_link_libraries(fast_math PRIVATE PkgConfig::NAUTY)
  target_compile_definitions(fast_math PUBLIC FAST_MATH_HAVE_NAUTY=1)
endif()

if(APPLE AND FAST_MATH_USE_COMMONCRYPTO)
  target_compile_definitions(
    fast_math
    PRIVATE
      FAST_MATH_USE_COMMONCRYPTO=1
  )
endif()

if(FAST_MATH_VERIFY_MODULAR_ARITHMETIC)
  target_compile_definitions(
    fast_math
    PRIVATE
      FAST_MATH_VERIFY_MODULAR_ARITHMETIC=1
  )
endif()

if(BUILD_TESTING)
  add_executable(fast_math_sparse_rank_core_test
    tests/sparse_rank_core_test.cpp
    cpp/src/sparse_rank.cpp
  )
  target_include_directories(
    fast_math_sparse_rank_core_test
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
  )
  target_compile_features(
    fast_math_sparse_rank_core_test
    PRIVATE
      cxx_std_20
  )
  if(FAST_MATH_VERIFY_MODULAR_ARITHMETIC)
    target_compile_definitions(
      fast_math_sparse_rank_core_test
      PRIVATE
        FAST_MATH_VERIFY_MODULAR_ARITHMETIC=1
    )
  endif()
  add_test(
    NAME fast_math_sparse_rank_core
    COMMAND fast_math_sparse_rank_core_test
  )
endif()

add_executable(fast_math_sparse_rank_benchmark
  benchmarks/benchmark_sparse_rank.cpp
)
target_compile_features(
  fast_math_sparse_rank_benchmark
  PRIVATE
    cxx_std_20
)
target_link_libraries(
  fast_math_sparse_rank_benchmark
  PRIVATE
    fast_math
)

add_executable(fast_math_sparse_rank_batch_benchmark
  benchmarks/benchmark_sparse_rank_batch.cpp
)
target_compile_features(
  fast_math_sparse_rank_batch_benchmark
  PRIVATE
    cxx_std_20
)
target_link_libraries(
  fast_math_sparse_rank_batch_benchmark
  PRIVATE
    fast_math
)

if(FAST_MATH_USE_FORKUNION)
  target_link_libraries(fast_math PRIVATE fast_math_forkunion)
  target_compile_definitions(fast_math PRIVATE FAST_MATH_USE_FORKUNION=1)

  if(BUILD_TESTING)
    add_executable(fast_math_parallel_idle_test
      tests/parallel_idle_test.cpp
    )
    target_include_directories(
      fast_math_parallel_idle_test
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/cpp/src
    )
    target_compile_features(
      fast_math_parallel_idle_test
      PRIVATE
        cxx_std_20
    )
    target_compile_definitions(
      fast_math_parallel_idle_test
      PRIVATE
        FAST_MATH_USE_FORKUNION=1
    )
    target_link_libraries(
      fast_math_parallel_idle_test
      PRIVATE
        fast_math_forkunion
        Threads::Threads
    )
    add_test(
      NAME fast_math_parallel_idle
      COMMAND fast_math_parallel_idle_test
    )
  endif()

  add_executable(fast_math_parallel_dispatch_benchmark
    benchmarks/benchmark_parallel_dispatch.cpp
  )
  target_compile_features(
    fast_math_parallel_dispatch_benchmark
    PRIVATE
      cxx_std_20
  )
  target_link_libraries(
    fast_math_parallel_dispatch_benchmark
    PRIVATE
      fast_math_forkunion
      Threads::Threads
  )
endif()

if(TARGET PkgConfig::FLINT)
  find_library(
    FAST_MATH_FLINT_LIBRARY
    NAMES flint
    HINTS ${FLINT_LIBRARY_DIRS}
    REQUIRED
  )
  target_sources(fast_math PRIVATE
    arb/src/cache.c
    arb/src/ordered_reduce.cpp
    arb/src/weight_intervals.cpp
  )
  target_include_directories(fast_math
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/arb/include
  )
  target_link_libraries(
    fast_math
    PRIVATE
      PkgConfig::FLINT
      ${FAST_MATH_FLINT_LIBRARY}
  )
  target_compile_definitions(fast_math PUBLIC FAST_MATH_HAVE_ARB=1)
  target_compile_definitions(fast_math PRIVATE FAST_MATH_HAVE_FLINT=1)

  if(BUILD_TESTING)
    add_executable(fast_math_arb_cache_test
      arb/tests/cache_test.c
    )
    target_link_libraries(
      fast_math_arb_cache_test
      PRIVATE
        fast_math
        PkgConfig::FLINT
    )
    add_test(NAME fast_math_arb_cache COMMAND fast_math_arb_cache_test)

    add_executable(fast_math_arb_ordered_reduce_test
      arb/tests/ordered_reduce_test.c
    )
    target_link_libraries(
      fast_math_arb_ordered_reduce_test
      PRIVATE
        fast_math
        PkgConfig::FLINT
    )
    add_test(
      NAME fast_math_arb_ordered_reduce
      COMMAND fast_math_arb_ordered_reduce_test
    )

    add_executable(fast_math_arb_weight_intervals_test
      arb/tests/weight_intervals_test.c
    )
    target_link_libraries(
      fast_math_arb_weight_intervals_test
      PRIVATE
        fast_math
        PkgConfig::FLINT
    )
    add_test(
      NAME fast_math_arb_weight_intervals
      COMMAND fast_math_arb_weight_intervals_test
    )
  endif()

  add_executable(fast_math_arb_cache_benchmark
    benchmarks/benchmark_arb_cache.c
  )
  target_link_libraries(
    fast_math_arb_cache_benchmark
    PRIVATE
      fast_math
      PkgConfig::FLINT
  )

  add_executable(fast_math_arb_weight_intervals_benchmark
    benchmarks/benchmark_arb_weight_intervals.c
  )
  target_link_libraries(
    fast_math_arb_weight_intervals_benchmark
    PRIVATE
      fast_math
      PkgConfig::FLINT
  )
endif()

if(MSVC)
  target_compile_options(fast_math PRIVATE /O2 /W4)
  if(TARGET fast_math_sparse_rank_core_test)
    target_compile_options(
      fast_math_sparse_rank_core_test
      PRIVATE
        /O2
        /W4
    )
  endif()
else()
  target_compile_options(fast_math PRIVATE
    -O3
    -Wall
    -Wextra
    -Wpedantic
  )
  if(TARGET fast_math_sparse_rank_core_test)
    target_compile_options(
      fast_math_sparse_rank_core_test
      PRIVATE
        -O3
        -Wall
        -Wextra
        -Wpedantic
    )
  endif()
  if(FAST_MATH_NATIVE_ARCH)
    if(APPLE)
      target_compile_options(fast_math PRIVATE -mcpu=native)
      if(TARGET fast_math_sparse_rank_core_test)
        target_compile_options(
          fast_math_sparse_rank_core_test
          PRIVATE
            -mcpu=native
        )
      endif()
    else()
      target_compile_options(fast_math PRIVATE -march=native)
      if(TARGET fast_math_sparse_rank_core_test)
        target_compile_options(
          fast_math_sparse_rank_core_test
          PRIVATE
            -march=native
        )
      endif()
    endif()
  endif()
endif()

set_target_properties(fast_math PROPERTIES
  CXX_EXTENSIONS OFF
  OUTPUT_NAME fast_math
)

set(
  LAMBDA_FAST_COMPAT_LIBRARY
  "${CMAKE_SHARED_LIBRARY_PREFIX}lambda_fast${CMAKE_SHARED_LIBRARY_SUFFIX}"
)
if(WIN32)
  add_custom_command(TARGET fast_math POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:fast_math>"
      "$<TARGET_FILE_DIR:fast_math>/${LAMBDA_FAST_COMPAT_LIBRARY}"
  )
else()
  add_custom_command(TARGET fast_math POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E rm -f
      "$<TARGET_FILE_DIR:fast_math>/${LAMBDA_FAST_COMPAT_LIBRARY}"
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      "$<TARGET_FILE_NAME:fast_math>"
      "$<TARGET_FILE_DIR:fast_math>/${LAMBDA_FAST_COMPAT_LIBRARY}"
  )
endif()
