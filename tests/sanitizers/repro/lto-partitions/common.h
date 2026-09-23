// The reproducer of the rule LTO-PART of tests/sanitizers/check-rules.sh.
// g_logger is a C++17 inline variable with a dynamic initializer (the lambda
// default member initializer), and a.cpp, b.cpp and main.cpp use it. With
// full LTO and more than one code generation partition, lld and clang
// 22.1.8 can leave its initializer out of .init_array. Then log_callback
// stays null, and the first log call stops the program with SIGSEGV.
#pragma once
#include <cstdio>
typedef void (*log_cb)(int level, const char * text, void * user);
struct logger {
    log_cb default_callback = [](int level, const char * text, void * user) {
        (void) level; (void) user; std::fputs(text, stdout);
    };
    log_cb log_callback = default_callback;
    void * user_data;
    void log(const char * text) { log_callback(0, text, user_data); }
};
inline logger g_logger;
int helper_a(int x);
int helper_b(int x);
