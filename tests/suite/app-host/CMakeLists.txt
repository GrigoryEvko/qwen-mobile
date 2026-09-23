# The host unit tests of the app (android/app/src/test/cpp), built with one
# sanitizer configuration of tests/sanitizers/<config>.cmake.
#
# android/app/src/test/cpp/CMakeLists.txt adds -fsanitize=address,undefined
# to each target. That gives two sanitizers in one build (rule R1), and it
# cannot build tsan or msan. This project compiles the same sources with only
# the flags of the configuration. The sources stay in android/ (read only).
# tests/suite/app-host-test.sh makes sure that the two lists of executables
# agree, thus a new test in android/ does not go missing here.
#
# Usage:
#   cmake -G Ninja -S tests/suite/app-host -B build/fuzz/matrix-app-host-<config> \
#       -C tests/sanitizers/<config>.cmake
cmake_minimum_required(VERSION 3.22)
project(qwenmobile-app-host-tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

get_filename_component(APP_TEST_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../android/app/src/test/cpp" ABSOLUTE)
get_filename_component(APP_CPP_DIR "${APP_TEST_DIR}/../../main/cpp" ABSOLUTE)

add_executable(cache_test
    "${APP_TEST_DIR}/cache_test.cpp"
    "${APP_CPP_DIR}/cache_io.cpp"
    "${APP_CPP_DIR}/image_cache.cpp"
    "${APP_CPP_DIR}/state_cache.cpp")

add_executable(spec_policy_test
    "${APP_TEST_DIR}/spec_policy_test.cpp"
    "${APP_CPP_DIR}/spec_policy.cpp")

find_package(Threads REQUIRED)
target_link_libraries(cache_test PRIVATE Threads::Threads)

foreach(test cache_test spec_policy_test)
    target_include_directories(${test} PRIVATE "${APP_CPP_DIR}")
    target_compile_options(${test} PRIVATE -Wall -Wextra)
endforeach()
