// The crash diagnostics of the replay driver: a signal handler that prints the state of a fatal
// signal before the process stops. It names the cause of a SIGILL on the phone: an instruction
// that the CPU does not have, a branch target (BTI) exception, or a pointer authentication (FPAC)
// failure.

#pragma once

namespace fo {

// Install the handler for SIGILL, and also for SIGSEGV, SIGBUS and SIGTRAP when no address
// sanitizer handles them. The handler prints to stderr, then gives the signal back to the default
// action, thus the exit status of the process stays the status of the signal.
void install_crash_diag();

// Run `real_main` in a child process that a parent process traces with ptrace, and print the
// state of each fatal signal (SIGILL, SIGSEGV, SIGBUS, SIGTRAP, SIGFPE, SIGABRT, SIGSYS) of each
// thread: the thread, si_code, si_addr, the registers, the instruction word at the pc, the module
// and the offset of the pc and of the link register, and a frame-pointer backtrace. The tracer
// sees a signal also when the thread blocks it (the kernel then forces the default action, and no
// handler of the process runs), for example in the start of a thread under ASan. Return the exit
// code of the child, or 128 plus the signal that stopped it. When ptrace is not permitted, the
// child runs untraced.
int run_traced(int argc, char ** argv, int (*real_main)(int, char **));

// Create and join one thread with pthread_create and one with std::thread, and print each step.
// A minimal reproducer of a fault in the thread start of a sanitizer runtime. Return 0.
int selftest_threads();

} // namespace fo
