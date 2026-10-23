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

add_test(NAME test_mat COMMAND $<TARGET_FILE:test_mat>)
add_test(NAME test_embeddings COMMAND $<TARGET_FILE:test_embeddings>)
add_test(NAME test_gqa COMMAND $<TARGET_FILE:test_gqa>)
add_test(NAME test_llama_block COMMAND $<TARGET_FILE:test_llama_block>)
