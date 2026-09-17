cmake_minimum_required(VERSION 3.22)
project(qwenmobile-cache-test LANGUAGES CXX)

# The host test of the engine caches. No llama.cpp and no NDK is necessary.
#   cmake -S . -B build && cmake --build build && ./build/cache_test

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../main/cpp")

add_executable(cache_test
    cache_test.cpp
    "${CPP_DIR}/cache_io.cpp"
    "${CPP_DIR}/image_cache.cpp"
    "${CPP_DIR}/state_cache.cpp")

target_include_directories(cache_test PRIVATE "${CPP_DIR}")
target_compile_options(cache_test PRIVATE -Wall -Wextra -fsanitize=address,undefined -g)
target_link_options(cache_test PRIVATE -fsanitize=address,undefined)
find_package(Threads REQUIRED)
target_link_libraries(cache_test PRIVATE Threads::Threads)
