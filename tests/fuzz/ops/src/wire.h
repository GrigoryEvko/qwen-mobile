// The file formats and the pipe protocol of the op fuzzer.
//
// A pack file holds cases: "FOPK", the version, the count, then for each case the forced kind
// (-1 for none), the name, and the bytes. A result file holds one record for each run of a case on
// a backend, and the replay driver appends and syncs each record, thus a crash loses no finished
// record. The oracle process of the libFuzzer harness reads a request (the case bytes) on stdin
// and writes a result on stdout. Every integer is little endian.

#pragma once

#include "exec.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace fo {

// One case of a pack.
struct pack_case {
    int32_t              forced = -1;
    std::string          name;
    std::vector<uint8_t> bytes;
};

// One record of a result file.
struct result_rec {
    uint32_t    idx = 0;  // the index of the case in the pack
    std::string tag;      // the backend and its configuration, for example "HTP0" or "HTP0-nofuse"
    run_result  rr;
};

// A byte writer.
struct wbuf {
    std::vector<uint8_t> b;
    void u32(uint32_t v);
    void u64(uint64_t v);
    void f64(double v);
    void str(const std::string & s);
    void bytes(const std::vector<uint8_t> & v);
};

// A byte reader. Each read after the end sets ok to false and gives zero.
struct rbuf {
    const uint8_t * p;
    size_t          n;
    size_t          i  = 0;
    bool            ok = true;
    rbuf(const uint8_t * data, size_t size) : p(data), n(size) {}
    uint32_t    u32();
    uint64_t    u64();
    double      f64();
    std::string str();
    std::vector<uint8_t> bytes();
};

// Encode and decode a run result.
void encode_result(wbuf & w, const run_result & r);
bool decode_result(rbuf & r, run_result & out);

// Read and write pack files. Return false with a reason on an error.
bool read_pack(const std::string & path, std::vector<pack_case> & cases, std::string & why);
bool write_pack(const std::string & path, const std::vector<pack_case> & cases, std::string & why);

// Append one record to an open result file, then flush and sync it.
bool append_result(FILE * f, const result_rec & rec);

// Read every complete record of a result file. A truncated last record is ignored.
bool read_results(const std::string & path, std::vector<result_rec> & recs, std::string & why);

// Read a whole file. Return false when it cannot be read.
bool read_file(const std::string & path, std::vector<uint8_t> & out);

// Write a whole file. Return false when it cannot be written.
bool write_file(const std::string & path, const void * data, size_t size);

// The oracle child process of the libFuzzer harness.
struct oracle_proc {
    int         pid  = -1;
    int         to   = -1;  // the stdin of the child
    int         from = -1;  // the stdout of the child
    std::string path;

    // Start the child "<path> serve". Return false when it cannot start.
    bool start();
    // Stop the child and wait for it.
    void stop();
    // Send one case and receive the oracle result. Return false when the child fails; the next
    // call starts a new child.
    bool call(int32_t forced, uint64_t max_bytes, const uint8_t * data, size_t size, run_result & out);
};

// Serve oracle requests on stdin and stdout until the end of the input. `run` computes one case.
int serve_loop(run_result (*run)(int32_t forced, uint64_t max_bytes, const uint8_t * data, size_t size));

} // namespace fo
