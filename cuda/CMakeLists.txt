# GPU decode path. Only included when configuring with -DTINYSERVE_CUDA=ON.
find_package(CUDAToolkit REQUIRED)

# Kernels: the only code compiled by nvcc. Plain pointers/ints, no tinyServe headers.
add_library(gpu_kernels STATIC gpu_kernels.cu)
set_target_properties(gpu_kernels PROPERTIES
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES 86)  # RTX A1000 (Ampere); change for another GPU
target_include_directories(gpu_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gpu_kernels PUBLIC CUDA::cudart)
target_compile_options(gpu_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

# TailBackend implementation: last layers + head on the GPU, built from a SmolLM3's mmap weights.
add_library(gpu_tail STATIC gpu_tail.cpp)
target_include_directories(gpu_tail PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
                                           ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gpu_tail PUBLIC gpu_kernels PRIVATE pcre2-8-static)
target_compile_options(gpu_tail PRIVATE -O3 -Wall -Wextra -Wpedantic)
if(TINYSERVE_NATIVE)
    target_compile_options(gpu_tail PRIVATE -march=native)
endif()

# Check program: GPU decode vs the CPU blocks on real weights (needs the converted smollm3/).
add_executable(gpu_decode_check gpu_decode_check.cpp)
target_include_directories(gpu_decode_check PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gpu_decode_check PRIVATE gpu_kernels)
target_compile_options(gpu_decode_check PRIVATE -O3 -Wall -Wextra -Wpedantic)
if(TINYSERVE_NATIVE)
    target_compile_options(gpu_decode_check PRIVATE -march=native)
endif()

# Check program: GPU head (final norm + tied lm_head) vs the CPU head on real hidden states.
add_executable(gpu_head_check gpu_head_check.cpp)
target_include_directories(gpu_head_check PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(gpu_head_check PRIVATE gpu_kernels)
target_compile_options(gpu_head_check PRIVATE -O3 -Wall -Wextra -Wpedantic)
if(TINYSERVE_NATIVE)
    target_compile_options(gpu_head_check PRIVATE -march=native)
endif()

# Check program: whole model, CPU-only vs GPU tail, over a scripted cache-transition sequence.
add_executable(gpu_tail_check gpu_tail_check.cpp)
target_link_libraries(gpu_tail_check PRIVATE gpu_tail pcre2-8-static)
target_compile_options(gpu_tail_check PRIVATE -O3 -Wall -Wextra -Wpedantic)
if(TINYSERVE_NATIVE)
    target_compile_options(gpu_tail_check PRIVATE -march=native)
endif()
