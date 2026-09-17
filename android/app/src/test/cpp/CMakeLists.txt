cmake_minimum_required(VERSION 3.22)
project(qwenmobile-host-tests LANGUAGES CXX)

# The host tests of the engine parts that need no llama.cpp and no NDK:
# the caches and the draft length policy.
#   cmake -S . -B build && cmake --build build && ./build/cache_test && ./build/spec_policy_test

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../main/cpp")

add_executable(cache_test
    cache_test.cpp
    "${CPP_DIR}/cache_io.cpp"
    "${CPP_DIR}/image_cache.cpp"
    "${CPP_DIR}/state_cache.cpp")

add_executable(spec_policy_test
    spec_policy_test.cpp
    "${CPP_DIR}/spec_policy.cpp")

find_package(Threads REQUIRED)
target_link_libraries(cache_test PRIVATE Threads::Threads)

foreach(test cache_test spec_policy_test)
    target_include_directories(${test} PRIVATE "${CPP_DIR}")
    target_compile_options(${test} PRIVATE -Wall -Wextra -fsanitize=address,undefined -g)
    target_link_options(${test} PRIVATE -fsanitize=address,undefined)
endforeach()
