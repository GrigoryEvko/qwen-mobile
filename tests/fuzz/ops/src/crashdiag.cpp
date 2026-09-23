// The crash diagnostics of the replay driver. Refer to crashdiag.h.
//
// The handler uses functions that are not async-signal-safe (dladdr, snprintf, the unwinder). The
// process stops after the handler, thus a deadlock is the worst result, and the outer timeout of
// the phone command ends it.

#include "crashdiag.h"

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <elf.h>
#include <pthread.h>
#include <string>
#include <sys/ptrace.h>
#include <sys/uio.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <thread>
#include <ucontext.h>
#include <unistd.h>
#include <unwind.h>

#ifndef __has_feature
#define __has_feature(x) 0
#endif

namespace fo {

namespace {

// Write a text to stderr.
void say(const char * s) {
    size_t n = std::strlen(s);
    while (n > 0) {
        const ssize_t k = ::write(2, s, n);
        if (k <= 0) {
            return;
        }
        s += k;
        n -= (size_t) k;
    }
}

// Print an address with its module and its symbol.
void say_addr(const char * label, uintptr_t a) {
    char    buf[512];
    Dl_info di;
    std::memset(&di, 0, sizeof(di));
    if (a != 0 && dladdr((void *) a, &di) != 0 && di.dli_fname) {
        std::snprintf(buf, sizeof(buf), "%s 0x%llx %s+0x%llx (%s+0x%llx)\n", label, (unsigned long long) a, di.dli_fname,
                      (unsigned long long) (a - (uintptr_t) di.dli_fbase), di.dli_sname ? di.dli_sname : "?",
                      (unsigned long long) (di.dli_saddr ? a - (uintptr_t) di.dli_saddr : 0));
    } else {
        std::snprintf(buf, sizeof(buf), "%s 0x%llx (no module)\n", label, (unsigned long long) a);
    }
    say(buf);
}

// Read 4 bytes of this process at `a`. Return false when the page is not readable, for example
// execute-only code.
bool read_word(uintptr_t a, uint32_t & w) {
    struct iovec local  = { &w, 4 };
    struct iovec remote = { (void *) a, 4 };
    return ::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0) == 4;
}

// Return a short name of an AArch64 instruction class that explains a SIGILL.
const char * insn_class(uint32_t w) {
    if ((w & 0xffffff3fu) == 0xd503241fu) {
        return "BTI landing pad";
    }
    if (w == 0xd503233fu || w == 0xd503237fu) {
        return "PACIASP/PACIBSP (an implicit BTI c landing pad)";
    }
    if ((w & 0xffff0000u) == 0x00000000u) {
        return "UDF (permanently undefined)";
    }
    if ((w & 0xffe0001fu) == 0xd4200000u) {
        return "BRK";
    }
    if ((w & 0x1e000000u) == 0x04000000u) {
        return "the SVE/SME encoding space";
    }
    return "another instruction";
}

_Unwind_Reason_Code trace_one(struct _Unwind_Context * ctx, void * arg) {
    int &           n  = *(int *) arg;
    const uintptr_t ip = (uintptr_t) _Unwind_GetIP(ctx);
    if (ip == 0 || n >= 40) {
        return _URC_END_OF_STACK;
    }
    char label[32];
    std::snprintf(label, sizeof(label), "  #%d", n++);
    say_addr(label, ip);
    return _URC_NO_REASON;
}

void handler(int sig, siginfo_t * si, void * uctx) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "\nops_replay: FATAL SIGNAL %d (%s) si_code %d si_addr %p pid %d tid %d\n", sig,
                  strsignal(sig), si ? si->si_code : -1, si ? si->si_addr : nullptr, (int) ::getpid(), (int) ::gettid());
    say(buf);
#if defined(__aarch64__)
    const ucontext_t * uc = (const ucontext_t *) uctx;
    const uintptr_t    pc = (uintptr_t) uc->uc_mcontext.pc;
    const uint64_t     ps = uc->uc_mcontext.pstate;
    say_addr("pc", pc);
    say_addr("lr", (uintptr_t) uc->uc_mcontext.regs[30]);
    uint32_t w = 0;
    if (read_word(pc, w)) {
        std::snprintf(buf, sizeof(buf), "insn at pc 0x%08x: %s\n", w, insn_class(w));
    } else {
        std::snprintf(buf, sizeof(buf), "insn at pc: not readable (errno %d, execute-only code?)\n", errno);
    }
    say(buf);
    // PSTATE.BTYPE (bits 11:10) is not zero after an indirect branch: a BTI exception at a pad
    // that does not accept that branch type gives SIGILL.
    std::snprintf(buf, sizeof(buf), "pstate 0x%llx btype %llu sp 0x%llx x16 0x%llx x17 0x%llx fault_address 0x%llx\n",
                  (unsigned long long) ps, (unsigned long long) ((ps >> 10) & 3),
                  (unsigned long long) uc->uc_mcontext.sp, (unsigned long long) uc->uc_mcontext.regs[16],
                  (unsigned long long) uc->uc_mcontext.regs[17], (unsigned long long) uc->uc_mcontext.fault_address);
    say(buf);
    // the ESR record of the signal frame, when the kernel gives it
    const uint8_t * p   = (const uint8_t *) uc->uc_mcontext.__reserved;
    size_t          off = 0;
    while (off + 8 <= sizeof(uc->uc_mcontext.__reserved)) {
        uint32_t magic, size;
        std::memcpy(&magic, p + off, 4);
        std::memcpy(&size, p + off + 4, 4);
        if (magic == 0 || size == 0) {
            break;
        }
        if (magic == 0x45535201u && size >= 16) {
            uint64_t esr;
            std::memcpy(&esr, p + off + 8, 8);
            std::snprintf(buf, sizeof(buf), "esr 0x%llx (EC 0x%llx)\n", (unsigned long long) esr,
                          (unsigned long long) ((esr >> 26) & 0x3f));
            say(buf);
        }
        off += size;
    }
#else
    (void) uctx;
#endif
    say("backtrace:\n");
    int n = 0;
    _Unwind_Backtrace(trace_one, &n);
    // SA_RESETHAND gave the default action back: a fault instruction runs again and stops the
    // process, and a raise ends it now.
    if (si == nullptr || si->si_code <= 0) {
        ::raise(sig);
    }
}

void install(int sig) {
    struct sigaction sa;
    std::memset(&sa, 0, sizeof(sa));
    sa.sa_sigaction = handler;
    sa.sa_flags     = SA_SIGINFO | SA_RESETHAND | SA_ONSTACK;
    sigemptyset(&sa.sa_mask);
    ::sigaction(sig, &sa, nullptr);
}

} // namespace

namespace {

// Print the maps line that holds the address, as "module+offset".
std::string where(pid_t pid, uint64_t a) {
    char path[64];
    std::snprintf(path, sizeof(path), "/proc/%d/maps", (int) pid);
    FILE * f = std::fopen(path, "r");
    if (!f) {
        return "?";
    }
    char        line[1024];
    std::string r = "(no mapping)";
    while (std::fgets(line, sizeof(line), f)) {
        unsigned long long lo = 0, hi = 0, off = 0;
        char               perms[8] = { 0 };
        int                n        = 0;
        if (std::sscanf(line, "%llx-%llx %7s %llx %*s %*s %n", &lo, &hi, perms, &off, &n) >= 4 && a >= lo && a < hi) {
            std::string mod = n > 0 ? std::string(line + n) : std::string();
            while (!mod.empty() && (mod.back() == '\n' || mod.back() == ' ')) {
                mod.pop_back();
            }
            char buf[1200];
            std::snprintf(buf, sizeof(buf), "%s+0x%llx [%s]", mod.empty() ? "[anon]" : mod.c_str(),
                          (unsigned long long) (a - lo + off), perms);
            r = buf;
            break;
        }
    }
    std::fclose(f);
    return r;
}

// Read 8 bytes of the tracee. Return false on an error.
bool peek(pid_t tid, uint64_t a, uint64_t & v) {
    errno   = 0;
    long w  = ::ptrace(PTRACE_PEEKDATA, tid, (void *) a, nullptr);
    v       = (uint64_t) w;
    return errno == 0;
}

// Print the state of a fatal signal of one traced thread.
void dump_stop(pid_t pid, pid_t tid, int sig) {
    char comm[64] = "?";
    {
        char path[96];
        std::snprintf(path, sizeof(path), "/proc/%d/task/%d/comm", (int) pid, (int) tid);
        if (FILE * f = std::fopen(path, "r")) {
            if (std::fgets(comm, sizeof(comm), f)) {
                comm[std::strcspn(comm, "\n")] = 0;
            }
            std::fclose(f);
        }
    }
    siginfo_t si;
    std::memset(&si, 0, sizeof(si));
    ::ptrace(PTRACE_GETSIGINFO, tid, nullptr, &si);
    std::fprintf(stderr, "\ntrace: SIGNAL %d (%s) in tid %d '%s'%s: si_code %d si_addr %p\n", sig, strsignal(sig), (int) tid,
                 comm, tid == pid ? " (main thread)" : "", si.si_code, si.si_addr);
#if defined(__aarch64__)
    struct user_regs_struct regs;
    struct iovec            io = { &regs, sizeof(regs) };
    if (::ptrace(PTRACE_GETREGSET, tid, (void *) NT_PRSTATUS, &io) != 0) {
        std::fprintf(stderr, "trace: PTRACE_GETREGSET failed: %s\n", std::strerror(errno));
        return;
    }
    // The PAC bits of a code address: the kernel gives the mask (NT_ARM_PAC_MASK). Android uses a
    // 39-bit address space, thus a fixed 48-bit mask keeps some PAC bits.
    uint64_t pac_mask[2] = { 0, 0 };   // data_mask, insn_mask
    {
        struct iovec pio = { pac_mask, sizeof(pac_mask) };
        if (::ptrace(PTRACE_GETREGSET, tid, (void *) 0x406 /* NT_ARM_PAC_MASK */, &pio) != 0) {
            pac_mask[1] = 0xffff000000000000ull;
        }
    }
    const uint64_t code_mask = ~pac_mask[1];
    const uint64_t pc = regs.pc, lr = regs.regs[30] & code_mask;
    uint64_t       w  = 0;
    const bool     ok = peek(tid, pc & ~7ull, w);
    const uint32_t insn = ok ? (uint32_t) (w >> ((pc & 4) ? 32 : 0)) : 0;
    std::fprintf(stderr, "trace: pc 0x%llx %s\n", (unsigned long long) pc, where(pid, pc).c_str());
    std::fprintf(stderr, "trace: lr 0x%llx %s\n", (unsigned long long) lr, where(pid, lr).c_str());
    std::fprintf(stderr, "trace: insn at pc %s0x%08x (%s), pstate 0x%llx btype %llu, sp 0x%llx\n", ok ? "" : "(unreadable) ",
                 insn, ok ? insn_class(insn) : "?", (unsigned long long) regs.pstate,
                 (unsigned long long) ((regs.pstate >> 10) & 3), (unsigned long long) regs.sp);
    for (int r = 0; r < 31; r += 4) {
        std::fprintf(stderr, "trace:");
        for (int k = r; k < r + 4 && k < 31; k++) {
            std::fprintf(stderr, " x%-2d 0x%016llx", k, (unsigned long long) regs.regs[k]);
        }
        std::fprintf(stderr, "\n");
    }
    // the frame records: [fp] = the previous fp, [fp + 8] = the return address (PAC bits removed)
    uint64_t fp = regs.regs[29];
    for (int k = 0; k < 32 && fp != 0; k++) {
        uint64_t next = 0, ret = 0;
        if (!peek(tid, fp, next) || !peek(tid, fp + 8, ret)) {
            break;
        }
        ret &= code_mask;
        std::fprintf(stderr, "trace:   #%d 0x%llx %s\n", k, (unsigned long long) ret, where(pid, ret).c_str());
        if (next <= fp) {
            break;
        }
        fp = next;
    }
#elif defined(__x86_64__)
    struct user_regs_struct regs;
    if (::ptrace(PTRACE_GETREGS, tid, nullptr, &regs) == 0) {
        std::fprintf(stderr, "trace: rip 0x%llx %s rsp 0x%llx\n", (unsigned long long) regs.rip,
                     where(pid, regs.rip).c_str(), (unsigned long long) regs.rsp);
    }
#endif
}

} // namespace

int run_traced(int argc, char ** argv, int (*real_main)(int, char **)) {
    const pid_t child = ::fork();
    if (child < 0) {
        std::fprintf(stderr, "trace: fork failed: %s, running untraced\n", std::strerror(errno));
        return real_main(argc, argv);
    }
    if (child == 0) {
        if (::ptrace(PTRACE_TRACEME, 0, nullptr, nullptr) != 0) {
            std::fprintf(stderr, "trace: PTRACE_TRACEME failed: %s, running untraced\n", std::strerror(errno));
        } else {
            ::raise(SIGSTOP);
        }
        std::_Exit(real_main(argc, argv));
    }
    int st = 0;
    if (::waitpid(child, &st, 0) != child || !WIFSTOPPED(st)) {
        // the child did not stop: it runs untraced, thus only wait for it
        while (::waitpid(child, &st, 0) == child && !WIFEXITED(st) && !WIFSIGNALED(st)) {
        }
        return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
    }
    ::ptrace(PTRACE_SETOPTIONS, child, nullptr, (void *) (long) (PTRACE_O_TRACECLONE | PTRACE_O_EXITKILL));
    ::ptrace(PTRACE_CONT, child, nullptr, nullptr);
    std::fprintf(stderr, "trace: tracing pid %d\n", (int) child);
    int result = 1;
    while (true) {
        const pid_t tid = ::waitpid(-1, &st, __WALL);
        if (tid < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if (WIFEXITED(st) || WIFSIGNALED(st)) {
            if (tid == child) {
                result = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
                std::fprintf(stderr, "trace: pid %d %s %d\n", (int) child, WIFEXITED(st) ? "exited with" : "stopped by signal",
                             WIFEXITED(st) ? WEXITSTATUS(st) : WTERMSIG(st));
                break;
            }
            continue;
        }
        if (!WIFSTOPPED(st)) {
            continue;
        }
        const int sig   = WSTOPSIG(st);
        const int event = st >> 16;
        if (event != 0 || sig == SIGSTOP) {
            // a clone event, or the first stop of a new thread: nothing to deliver
            ::ptrace(PTRACE_CONT, tid, nullptr, nullptr);
            continue;
        }
        if (sig == SIGILL || sig == SIGSEGV || sig == SIGBUS || sig == SIGTRAP || sig == SIGFPE || sig == SIGABRT ||
            sig == SIGSYS) {
            dump_stop(child, tid, sig);
        }
        ::ptrace(PTRACE_CONT, tid, nullptr, (void *) (long) sig);
    }
    return result;
}

int selftest_threads() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // FUZZ_OPS_SELFTEST_FAULT=1 makes the thread block every signal and then execute a trap
    // instruction: the validation of the tracer (the silent death of a thread start).
    const char * fault = std::getenv("FUZZ_OPS_SELFTEST_FAULT");
    const bool   do_fault = fault != nullptr && fault[0] == '1';
    std::printf("selftest: pthread_create%s\n", do_fault ? " (with a trap in the thread, signals blocked)" : "");
    pthread_t t;
    const int rc = ::pthread_create(
        &t, nullptr,
        [](void * arg) -> void * {
            if (arg != nullptr) {
                sigset_t all;
                sigfillset(&all);
                ::pthread_sigmask(SIG_SETMASK, &all, nullptr);
                __builtin_trap();
            }
            return nullptr;
        },
        do_fault ? (void *) 1 : nullptr);
    std::printf("selftest: pthread_create returned %d\n", rc);
    if (rc == 0) {
        ::pthread_join(t, nullptr);
        std::printf("selftest: pthread_join done\n");
    }
    std::printf("selftest: std::thread\n");
    std::thread th([] {});
    th.join();
    std::printf("selftest: std::thread done\n");
    return 0;
}

void install_crash_diag() {
    // an alternate stack, thus a stack overflow also gives a report
    static uint8_t alt[64 * 1024];
    stack_t        st;
    st.ss_sp    = alt;
    st.ss_size  = sizeof(alt);
    st.ss_flags = 0;
    ::sigaltstack(&st, nullptr);
    install(SIGILL);
#if !__has_feature(address_sanitizer) && !__has_feature(hwaddress_sanitizer) && !__has_feature(thread_sanitizer) && \
    !__has_feature(memory_sanitizer)
    install(SIGSEGV);
    install(SIGBUS);
    install(SIGTRAP);
#endif
}

} // namespace fo
