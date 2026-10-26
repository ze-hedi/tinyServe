include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
FetchContent_MakeAvailable(googletest)

add_executable(test_mat test_mat.cpp)
target_include_directories(test_mat PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_mat PRIVATE gtest)

add_executable(test_embeddings test_embeddings.cpp)
target_include_directories(test_embeddings PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_embeddings PRIVATE gtest)

add_executable(test_gqa test_gqa.cpp)
target_include_directories(test_gqa PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_gqa PRIVATE gtest)

add_executable(test_llama_block test_llama_block.cpp)
target_include_directories(test_llama_block PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_llama_block PRIVATE gtest)

add_executable(test_kv_cache test_kv_cache.cpp)
target_include_directories(test_kv_cache PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_kv_cache PRIVATE gtest gtest_main)

add_executable(test_chat_template test_chat_template.cpp)
target_include_directories(test_chat_template PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_chat_template PRIVATE gtest gtest_main pcre2-8-static)
target_compile_definitions(test_chat_template PRIVATE TINYSERVE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_test(NAME test_mat COMMAND $<TARGET_FILE:test_mat>)
add_test(NAME test_embeddings COMMAND $<TARGET_FILE:test_embeddings>)
add_test(NAME test_gqa COMMAND $<TARGET_FILE:test_gqa>)
add_test(NAME test_llama_block COMMAND $<TARGET_FILE:test_llama_block>)
add_test(NAME test_kv_cache COMMAND $<TARGET_FILE:test_kv_cache>)
add_test(NAME test_chat_template COMMAND $<TARGET_FILE:test_chat_template>)

# add_executable(test_tokenizer test_tokenizer.cpp)
# target_include_directories(test_tokenizer PRIVATE ${CMAKE_SOURCE_DIR}/include)
# target_link_libraries(test_tokenizer PRIVATE gtest pcre2-8)
# add_test(NAME test_tokenizer COMMAND $<TARGET_FILE:test_tokenizer>)
