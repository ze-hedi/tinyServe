cmake_minimum_required(VERSION 3.20)
project(tinyServe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(TINYSERVE_NATIVE "Optimize the mmap benchmark for the build host CPU" OFF)

include(FetchContent)

set(PCRE2_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(PCRE2_BUILD_PCRE2GREP OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  pcre2
  GIT_REPOSITORY https://github.com/PCRE2Project/pcre2.git
  GIT_TAG pcre2-10.44
)
FetchContent_MakeAvailable(pcre2)

enable_testing()
add_subdirectory(tests)

add_executable(bench_block bench_block.cpp)
target_include_directories(bench_block PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(bench_block PRIVATE -O3)

add_executable(bench_block_mmap bench_block_mmap.cpp)
target_include_directories(bench_block_mmap PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(bench_block_mmap PRIVATE -O3 -Wall -Wextra -Wpedantic)
if(TINYSERVE_NATIVE)
    target_compile_options(bench_block_mmap PRIVATE -march=native)
endif()

add_executable(bench_smollm3_mmap bench_smollm3_mmap.cpp)
target_include_directories(bench_smollm3_mmap PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(bench_smollm3_mmap PRIVATE -O3 -Wall -Wextra -Wpedantic)
if(TINYSERVE_NATIVE)
    target_compile_options(bench_smollm3_mmap PRIVATE -march=native)
endif()

add_executable(tokenizer_test tokenizer_test.cpp)
target_include_directories(tokenizer_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(tokenizer_test PRIVATE pcre2-8-static)
target_compile_options(tokenizer_test PRIVATE -O3)
