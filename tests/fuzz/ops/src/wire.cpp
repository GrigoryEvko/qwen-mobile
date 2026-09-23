// The file formats and the pipe protocol. Refer to wire.h.

#include "wire.h"

#include <cerrno>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

extern char ** environ;

namespace fo {

namespace {

constexpr uint32_t PACK_MAGIC   = 0x4b504f46u; // "FOPK"
constexpr uint32_t RESULT_MAGIC = 0x52524f46u; // "FORR"
constexpr uint32_t REQ_MAGIC    = 0x51524f46u; // "FORQ"
constexpr uint32_t PACK_VERSION = 1;

// Write every byte, and retry after a signal. Return false on an error.
bool write_all(int fd, const void * data, size_t size) {
    const uint8_t * p = (const uint8_t *) data;
    while (size > 0) {
        const ssize_t k = ::write(fd, p, size);
        if (k < 0 && errno == EINTR) {
            continue;
        }
        if (k <= 0) {
            return false;
        }
        p += k;
        size -= (size_t) k;
    }
    return true;
}

// Read exactly `size` bytes. Return false at the end of the input or on an error.
bool read_all(int fd, void * data, size_t size) {
    uint8_t * p = (uint8_t *) data;
    while (size > 0) {
        const ssize_t k = ::read(fd, p, size);
        if (k < 0 && errno == EINTR) {
            continue;
        }
        if (k <= 0) {
            return false;
        }
        p += k;
        size -= (size_t) k;
    }
    return true;
}

// Send a length-prefixed frame.
bool send_frame(int fd, const std::vector<uint8_t> & b) {
    const uint64_t n = b.size();
    return write_all(fd, &n, 8) && write_all(fd, b.data(), b.size());
}

// Receive a length-prefixed frame of at most 4 GiB.
bool recv_frame(int fd, std::vector<uint8_t> & b) {
    uint64_t n = 0;
    if (!read_all(fd, &n, 8) || n > (uint64_t(1) << 32)) {
        return false;
    }
    b.resize((size_t) n);
    return read_all(fd, b.data(), b.size());
}

} // namespace

void wbuf::u32(uint32_t v) {
    for (int k = 0; k < 4; k++) {
        b.push_back((uint8_t) (v >> (8 * k)));
    }
}

void wbuf::u64(uint64_t v) {
    for (int k = 0; k < 8; k++) {
        b.push_back((uint8_t) (v >> (8 * k)));
    }
}

void wbuf::f64(double v) {
    uint64_t u;
    std::memcpy(&u, &v, 8);
    u64(u);
}

void wbuf::str(const std::string & s) {
    u32((uint32_t) s.size());
    b.insert(b.end(), s.begin(), s.end());
}

void wbuf::bytes(const std::vector<uint8_t> & v) {
    u64(v.size());
    b.insert(b.end(), v.begin(), v.end());
}

uint32_t rbuf::u32() {
    if (i + 4 > n) {
        ok = false;
        i  = n;
        return 0;
    }
    uint32_t v = 0;
    for (int k = 0; k < 4; k++) {
        v |= (uint32_t) p[i + k] << (8 * k);
    }
    i += 4;
    return v;
}

uint64_t rbuf::u64() {
    const uint64_t lo = u32();
    return lo | ((uint64_t) u32() << 32);
}

double rbuf::f64() {
    const uint64_t u = u64();
    double         v;
    std::memcpy(&v, &u, 8);
    return v;
}

std::string rbuf::str() {
    const uint32_t len = u32();
    if (!ok || i + len > n) {
        ok = false;
        i  = n;
        return {};
    }
    std::string s((const char *) p + i, len);
    i += len;
    return s;
}

std::vector<uint8_t> rbuf::bytes() {
    const uint64_t len = u64();
    if (!ok || i + len > n) {
        ok = false;
        i  = n;
        return {};
    }
    std::vector<uint8_t> v(p + i, p + i + len);
    i += (size_t) len;
    return v;
}

void encode_result(wbuf & w, const run_result & r) {
    w.u32(r.status);
    w.u32(r.flags);
    w.str(r.detail);
    w.f64(r.ms);
    w.u64(r.input_hash);
    w.u32((uint32_t) r.outs.size());
    for (const auto & o : r.outs) {
        w.bytes(o);
    }
}

bool decode_result(rbuf & r, run_result & out) {
    out.status = r.u32();
    out.flags  = r.u32();
    out.detail     = r.str();
    out.ms         = r.f64();
    out.input_hash = r.u64();
    const uint32_t n = r.u32();
    if (!r.ok || n > 4096) {
        return false;
    }
    out.outs.resize(n);
    for (auto & o : out.outs) {
        o = r.bytes();
    }
    return r.ok;
}

bool read_file(const std::string & path, std::vector<uint8_t> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    out.clear();
    uint8_t buf[65536];
    size_t  k;
    while ((k = std::fread(buf, 1, sizeof(buf), f)) > 0) {
        out.insert(out.end(), buf, buf + k);
    }
    const bool ok = !std::ferror(f);
    std::fclose(f);
    return ok;
}

bool write_file(const std::string & path, const void * data, size_t size) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    const bool ok = std::fwrite(data, 1, size, f) == size;
    return std::fclose(f) == 0 && ok;
}

bool read_pack(const std::string & path, std::vector<pack_case> & cases, std::string & why) {
    std::vector<uint8_t> b;
    if (!read_file(path, b)) {
        why = "cannot read the pack " + path;
        return false;
    }
    rbuf r(b.data(), b.size());
    if (r.u32() != PACK_MAGIC || r.u32() != PACK_VERSION) {
        why = path + " is not a pack of version 1";
        return false;
    }
    const uint32_t n = r.u32();
    cases.resize(n);
    for (auto & c : cases) {
        c.forced = (int32_t) r.u32();
        c.name   = r.str();
        c.bytes  = r.bytes();
    }
    if (!r.ok) {
        why = path + " is truncated";
        return false;
    }
    return true;
}

bool write_pack(const std::string & path, const std::vector<pack_case> & cases, std::string & why) {
    wbuf w;
    w.u32(PACK_MAGIC);
    w.u32(PACK_VERSION);
    w.u32((uint32_t) cases.size());
    for (const auto & c : cases) {
        w.u32((uint32_t) c.forced);
        w.str(c.name);
        w.bytes(c.bytes);
    }
    if (!write_file(path, w.b.data(), w.b.size())) {
        why = "cannot write the pack " + path;
        return false;
    }
    return true;
}

bool append_result(FILE * f, const result_rec & rec) {
    wbuf w;
    w.u32(RESULT_MAGIC);
    w.u32(rec.idx);
    w.str(rec.tag);
    encode_result(w, rec.rr);
    const uint64_t n = w.b.size();
    if (std::fwrite(&n, 8, 1, f) != 1 || std::fwrite(w.b.data(), 1, w.b.size(), f) != w.b.size()) {
        return false;
    }
    std::fflush(f);
    ::fsync(fileno(f));
    return true;
}

bool read_results(const std::string & path, std::vector<result_rec> & recs, std::string & why) {
    std::vector<uint8_t> b;
    if (!read_file(path, b)) {
        why = "cannot read the results " + path;
        return false;
    }
    size_t off = 0;
    while (off + 8 <= b.size()) {
        uint64_t n;
        std::memcpy(&n, b.data() + off, 8);
        if (off + 8 + n > b.size()) {
            break;
        }
        rbuf r(b.data() + off + 8, (size_t) n);
        result_rec rec;
        if (r.u32() != RESULT_MAGIC) {
            why = path + ": a record has no magic";
            return false;
        }
        rec.idx = r.u32();
        rec.tag = r.str();
        if (!decode_result(r, rec.rr)) {
            why = path + ": a record does not decode";
            return false;
        }
        recs.push_back(std::move(rec));
        off += 8 + (size_t) n;
    }
    return true;
}

bool oracle_proc::start() {
    int in_pipe[2], out_pipe[2];
    if (::pipe(in_pipe) != 0) {
        return false;
    }
    if (::pipe(out_pipe) != 0) {
        ::close(in_pipe[0]);
        ::close(in_pipe[1]);
        return false;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, in_pipe[0], 0);
    posix_spawn_file_actions_adddup2(&fa, out_pipe[1], 1);
    posix_spawn_file_actions_addclose(&fa, in_pipe[1]);
    posix_spawn_file_actions_addclose(&fa, out_pipe[0]);
    char * argv[] = { const_cast<char *>(path.c_str()), const_cast<char *>("serve"), nullptr };
    pid_t  child  = -1;
    const int rc  = posix_spawn(&child, path.c_str(), &fa, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    ::close(in_pipe[0]);
    ::close(out_pipe[1]);
    if (rc != 0) {
        ::close(in_pipe[1]);
        ::close(out_pipe[0]);
        return false;
    }
    // a dead child must give EPIPE, not a SIGPIPE that stops the fuzzer
    std::signal(SIGPIPE, SIG_IGN);
    pid  = child;
    to   = in_pipe[1];
    from = out_pipe[0];
    return true;
}

void oracle_proc::stop() {
    if (to >= 0) {
        ::close(to);
    }
    if (from >= 0) {
        ::close(from);
    }
    if (pid > 0) {
        int st = 0;
        ::kill(pid, SIGKILL);
        ::waitpid(pid, &st, 0);
    }
    pid  = -1;
    to   = -1;
    from = -1;
}

bool oracle_proc::call(int32_t forced, uint64_t max_bytes, const uint8_t * data, size_t size, run_result & out) {
    if (pid < 0 && !start()) {
        return false;
    }
    wbuf w;
    w.u32(REQ_MAGIC);
    w.u32((uint32_t) forced);
    w.u64(max_bytes);
    w.bytes(std::vector<uint8_t>(data, data + size));
    std::vector<uint8_t> resp;
    if (!send_frame(to, w.b) || !recv_frame(from, resp)) {
        stop();
        return false;
    }
    rbuf r(resp.data(), resp.size());
    return decode_result(r, out);
}

int serve_loop(run_result (*run)(int32_t forced, uint64_t max_bytes, const uint8_t * data, size_t size)) {
    std::vector<uint8_t> req;
    while (recv_frame(0, req)) {
        rbuf r(req.data(), req.size());
        if (r.u32() != REQ_MAGIC) {
            return 2;
        }
        const int32_t              forced    = (int32_t) r.u32();
        const uint64_t             max_bytes = r.u64();
        const std::vector<uint8_t> bytes     = r.bytes();
        if (!r.ok) {
            return 2;
        }
        const run_result res = run(forced, max_bytes, bytes.data(), bytes.size());
        wbuf             w;
        encode_result(w, res);
        if (!send_frame(1, w.b)) {
            return 3;
        }
    }
    return 0;
}

} // namespace fo
