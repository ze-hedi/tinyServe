cmake_minimum_required(VERSION 3.20)
project(tinyServe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()
add_subdirectory(tests)

add_executable(bench_block bench_block.cpp)
target_include_directories(bench_block PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_compile_options(bench_block PRIVATE -O3)
