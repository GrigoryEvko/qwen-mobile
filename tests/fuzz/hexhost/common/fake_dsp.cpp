// The fake DSP of the hexhost fuzz harness: the runtime part.
//
// This file gives the C functions of the Hexagon SDK that the host part of the
// Hexagon backend calls (rpcmem, FastRPC session control, dspqueue, htp_iface,
// htpdrv), a parser of the op batches, and the record of the executed ops. The
// model of the DSP side checks of each op is in dsp_model.cpp.

#include "fake_dsp.h"
#include "dsp_model.h"

#include <AEEStdErr.h>
#include <dspqueue.h>
#include <remote.h>
#include <rpcmem.h>
#include <htp_iface.h>

#include "htp-ops.h"

#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>

namespace fakedsp {

namespace {

// One rpcmem allocation
struct alloc_info {
    int    fd   = -1;
    size_t size = 0;
};

// One packet of a dspqueue
struct packet {
    std::vector<uint8_t>   msg;
    struct dspqueue_buffer buf {};
    uint32_t               n_bufs = 0;
};

struct handle_info;

// One dspqueue, with the requests of the host and the responses of the DSP
struct queue_info {
    uint64_t                id     = 0;
    handle_info *           handle = nullptr;   // the session that imported the queue
    std::mutex              mu;
    std::condition_variable cv;
    std::deque<packet>      req;
    std::deque<packet>      rsp;
    bool                    closing = false;
    std::thread             dsp;                // the DSP thread (async mode)
    dspqueue_callback_t     error_cb = nullptr; // the error callback of the host, or nullptr
    void *                  cb_ctx   = nullptr; // the context of error_cb
    bool                    failed   = false;   // HEXHOST_EINTR=cancel:N stopped the queue
    std::thread             error_thread;       // the thread that calls error_cb after the stop
};

// One htp_iface session (a remote handle)
struct handle_info {
    uint64_t     id          = 0;
    queue_info * queue       = nullptr;
    uint32_t     n_threads   = 0;
    uint32_t     n_hmx       = 0;
    uint64_t     max_vmem    = 0;
    uint32_t     profiler    = 0;
    bool         started     = false;
};

std::mutex                                          g_mu;          // guards every map below
config                                              g_cfg;
std::map<uint64_t, alloc_info>                      g_allocs;      // base address -> allocation
std::unordered_map<int, uint64_t>                   g_fd_base;     // fd -> base address
int                                                 g_next_fd = 100;
std::atomic<uint64_t>                               g_alloc_gen{0}; // + 1 at each allocation and release
std::unordered_map<uint64_t, std::unique_ptr<queue_info>>  g_queues;   // queue id -> queue
std::unordered_map<uint64_t, std::unique_ptr<handle_info>> g_handles;  // handle id -> session
uint64_t                                            g_next_id = 1;

std::mutex                g_rec_mu;      // guards the record
std::vector<batch_record> g_batches;

std::once_flag            g_ignore_once;
std::set<std::string>     g_ignore;

// Reads HEXHOST_IGNORE once into the set of ignored violation ids.
void load_ignore() {
    std::call_once(g_ignore_once, [] {
        const char * s = getenv("HEXHOST_IGNORE");
        if (!s) {
            return;
        }
        std::string all(s);
        size_t      pos = 0;
        while (pos <= all.size()) {
            size_t end = all.find(',', pos);
            if (end == std::string::npos) {
                end = all.size();
            }
            if (end > pos) {
                g_ignore.insert(all.substr(pos, end - pos));
            }
            pos = end + 1;
        }
    });
}

// Gives the queue of an id, or nullptr. The caller holds g_mu.
queue_info * find_queue(uint64_t id) {
    auto it = g_queues.find(id);
    return it == g_queues.end() ? nullptr : it->second.get();
}

// Processes one request packet of the host and gives the response packet.
// It runs on the host thread (sync mode) or on the DSP thread (async mode).
packet process_request(queue_info * q, const packet & in) {
    packet out;
    out.buf    = in.buf;
    out.n_bufs = 1;

    if (in.n_bufs != 1 || in.msg.size() != sizeof(htp_opbatch_req)) {
        violation("desc-request", "the request has %u buffers and %zu message bytes (the DSP skips it and the host waits forever)",
                  in.n_bufs, in.msg.size());
    }

    htp_opbatch_req req;
    memcpy(&req, in.msg.data(), sizeof(req));

    handle_info * h = q->handle;
    dsp_ctx       ctx;
    ctx.n_threads = h ? h->n_threads : g_cfg.n_threads;
    ctx.n_hmx     = h ? h->n_hmx : g_cfg.n_hmx;
    ctx.vtcm_size = g_cfg.vtcm_size;
    ctx.profiler  = h ? h->profiler : 0;
    ctx.max_vmem  = h ? h->max_vmem : 0;
    ctx.touch     = g_cfg.touch;
    ctx.fill      = g_cfg.fill;

    batch_record rec;
    rec.seq   = req.seq;
    rec.queue = q->id;

    htp_opbatch_rsp rsp;
    memset(&rsp, 0, sizeof(rsp));
    rsp.seq       = req.seq;
    rsp.status    = HTP_STATUS_OK;
    rsp.n_bufs    = req.n_bufs;
    rsp.n_tensors = req.n_tensors;
    rsp.n_ops     = req.n_ops;
    rsp.err_op    = 0xffffffffu;

    process_batch(ctx, req, in.buf, rec, rsp);

    rec.status = rsp.status;
    {
        std::lock_guard<std::mutex> lock(g_rec_mu);
        g_batches.push_back(std::move(rec));
    }

    out.msg.resize(sizeof(rsp));
    memcpy(out.msg.data(), &rsp, sizeof(rsp));
    return out;
}

// The loop of the DSP thread of one queue (async mode)
void dsp_thread_main(queue_info * q) {
    std::unique_lock<std::mutex> lock(q->mu);
    for (;;) {
        q->cv.wait(lock, [q] { return q->closing || !q->req.empty(); });
        if (q->req.empty()) {
            return;
        }
        packet in = std::move(q->req.front());
        q->req.pop_front();
        lock.unlock();
        packet out = process_request(q, in);
        lock.lock();
        q->rsp.push_back(std::move(out));
        q->cv.notify_all();
    }
}

// The model of the code AEE_EINTERRUPTED of dspqueue_read and dspqueue_write (HEXHOST_EINTR)
struct intr_config {
    uint64_t every  = 0;   // each Nth read and each Nth write gives the code (0: off)
    uint64_t cancel = 0;   // the Nth read or write of the process stops its queue (0: off)
};

std::atomic<uint64_t> g_reads{0};    // the dspqueue_read calls of the process
std::atomic<uint64_t> g_writes{0};   // the dspqueue_write calls of the process
std::atomic<uint64_t> g_calls{0};    // the dspqueue_read and dspqueue_write calls of the process

// Reads HEXHOST_EINTR one time. A value that is not valid stops the process with a message.
const intr_config & intr() {
    static const intr_config c = [] {
        intr_config  r;
        const char * s = getenv("HEXHOST_EINTR");
        if (!s || !*s || strcmp(s, "0") == 0) {
            return r;
        }
        const bool   cancel = strncmp(s, "cancel:", 7) == 0;
        const char * num    = cancel ? s + 7 : s;
        char *       end    = nullptr;
        errno               = 0;
        const unsigned long long n = strtoull(num, &end, 10);
        if (!isdigit((unsigned char) num[0]) || errno != 0 || *end != '\0' || n < (cancel ? 1u : 2u)) {
            fprintf(stderr,
                    "hexhost: HEXHOST_EINTR is '%s'. Give N (2 or more: each Nth dspqueue_read and each Nth "
                    "dspqueue_write gives AEE_EINTERRUPTED) or cancel:N (1 or more: the Nth call stops its queue).\n",
                    s);
            abort();
        }
        (cancel ? r.cancel : r.every) = n;
        return r;
    }();
    return c;
}

// Stops a queue, as the dspqueue library does when the DSP process stops: each later read and
// write of the queue gives AEE_EINTERRUPTED immediately, and a different thread calls the error
// callback of the host 2 ms later. The caller holds q->mu.
void stop_queue(queue_info * q) {
    q->failed = true;
    fprintf(stderr, "hexhost: HEXHOST_EINTR=cancel: the queue %llu stops\n", (unsigned long long) q->id);
    q->error_thread = std::thread([q] {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (q->error_cb) {
            q->error_cb((dspqueue_t) (uintptr_t) q->id, AEE_ENOSUCH, q->cb_ctx);
        }
    });
}

std::once_flag g_exit_once;

// When the process exits, the destructor of the FastRPC library stops the DSP process, and the
// dspqueue library then calls the error callback of each open queue with AEE_ENOSUCH (seen on the
// phone). This function does the same for each queue that the host did not close. atexit() calls it.
void exit_notify() {
    std::vector<queue_info *> open;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        for (const auto & kv : g_queues) {
            open.push_back(kv.second.get());
        }
    }
    for (queue_info * q : open) {
        if (q->error_cb) {
            q->error_cb((dspqueue_t) (uintptr_t) q->id, AEE_ENOSUCH, q->cb_ctx);
        }
    }
}

// Gives true when a dspqueue_read or a dspqueue_write of the queue gives AEE_EINTERRUPTED and
// moves no packet (HEXHOST_EINTR). calls is the counter of that function.
bool interrupted(queue_info * q, std::atomic<uint64_t> & calls) {
    const intr_config & c = intr();
    if (c.every == 0 && c.cancel == 0) {
        return false;
    }
    std::lock_guard<std::mutex> lock(q->mu);
    if (q->failed) {
        return true;
    }
    if (c.cancel != 0) {
        if (g_calls.fetch_add(1) + 1 == c.cancel) {
            stop_queue(q);
            return true;
        }
        return false;
    }
    if ((calls.fetch_add(1) + 1) % c.every == 0) {
        count("dspqueue calls interrupted");
        return true;
    }
    return false;
}

} // namespace

void configure(const config & cfg) {
    std::lock_guard<std::mutex> lock(g_mu);
    g_cfg = cfg;
}

const config & get_config() {
    return g_cfg;
}

std::vector<batch_record> take_batches() {
    std::lock_guard<std::mutex> lock(g_rec_mu);
    std::vector<batch_record> out;
    out.swap(g_batches);
    return out;
}

void reset_record() {
    std::lock_guard<std::mutex> lock(g_rec_mu);
    g_batches.clear();
}

bool is_ignored(const char * id) {
    load_ignore();
    return g_ignore.count(id) != 0;
}

bool violation(const char * id, const char * fmt, ...) {
    if (is_ignored(id)) {
        return true;
    }
    char    msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    fprintf(stderr, "\nhexhost: VIOLATION %s: %s\n", id, msg);
    fflush(stderr);
    abort();
}

bool lookup_alloc(uint64_t addr, uint64_t * base, uint64_t * size, int * fd) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_allocs.upper_bound(addr);
    if (it == g_allocs.begin()) {
        return false;
    }
    --it;
    if (addr >= it->first + it->second.size) {
        return false;
    }
    *base = it->first;
    *size = it->second.size;
    *fd   = it->second.fd;
    return true;
}

uint64_t alloc_generation() {
    return g_alloc_gen.load(std::memory_order_acquire);
}

size_t live_allocs() {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_allocs.size();
}

size_t live_queues() {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_queues.size();
}

size_t live_handles() {
    std::lock_guard<std::mutex> lock(g_mu);
    return g_handles.size();
}

namespace {

std::mutex                      g_stat_mu;
std::map<std::string, uint64_t> g_stats;

// Prints the counters at exit.
void print_stats() {
    std::lock_guard<std::mutex> lock(g_stat_mu);
    fprintf(stderr, "hexhost: counters\n");
    for (const auto & kv : g_stats) {
        fprintf(stderr, "  %-40s %llu\n", kv.first.c_str(), (unsigned long long) kv.second);
    }
}

} // namespace

void count(const std::string & name, uint64_t n) {
    static const bool on = [] {
        const bool e = getenv("HEXHOST_STATS") != nullptr;
        if (e) {
            atexit(print_stats);
        }
        return e;
    }();
    if (!on) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_stat_mu);
    g_stats[name] += n;
}

} // namespace fakedsp

using namespace fakedsp;

extern "C" {

// ---- rpcmem

// Allocates a zeroed, page aligned block and gives it an fd.
void * rpcmem_alloc2(int heapid, uint32_t flags, size_t size) {
    (void) heapid;
    (void) flags;
    if (size == 0 || size > (size_t) 2 << 30) {
        return nullptr;
    }
    const size_t rounded = (size + 4095) & ~(size_t) 4095;
    void *       p       = aligned_alloc(4096, rounded);
    if (!p) {
        return nullptr;
    }
    memset(p, 0, rounded);
    std::lock_guard<std::mutex> lock(g_mu);
    alloc_info info;
    info.fd   = g_next_fd++;
    info.size = size;
    g_allocs[(uint64_t) (uintptr_t) p] = info;
    g_fd_base[info.fd]                 = (uint64_t) (uintptr_t) p;
    g_alloc_gen.fetch_add(1, std::memory_order_release);
    return p;
}

// Allocates a block with an int size (the old API).
void * rpcmem_alloc(int heapid, uint32_t flags, int size) {
    return size > 0 ? rpcmem_alloc2(heapid, flags, (size_t) size) : nullptr;
}

// Frees a block of rpcmem_alloc2. A pointer that is not the base of a block is a violation.
void rpcmem_free(void * po) {
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_allocs.find((uint64_t) (uintptr_t) po);
        if (it == g_allocs.end()) {
            violation("rpcmem-free", "rpcmem_free of %p, which is not the base of an allocation", po);
            return;
        }
        g_fd_base.erase(it->second.fd);
        g_allocs.erase(it);
        g_alloc_gen.fetch_add(1, std::memory_order_release);
    }
    free(po);
}

// Gives the fd of a block, or -1.
int rpcmem_to_fd(void * po) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_allocs.find((uint64_t) (uintptr_t) po);
    return it == g_allocs.end() ? -1 : it->second.fd;
}

// ---- FastRPC session control

// Gives the answers of the session requests of ggml_hexagon_session::allocate.
int remote_session_control(uint32_t req, void * data, uint32_t datalen) {
    (void) datalen;
    switch (req) {
        case FASTRPC_RESERVE_NEW_SESSION: {
            auto * n = (struct remote_rpc_reserve_new_session *) data;
            n->effective_domain_id = CDSP_DOMAIN_ID;
            return AEE_SUCCESS;
        }
        case FASTRPC_GET_EFFECTIVE_DOMAIN_ID: {
            auto * e = (struct remote_rpc_effective_domain_id *) data;
            e->effective_domain_id = CDSP_DOMAIN_ID;
            return AEE_SUCCESS;
        }
        case DSPRPC_CONTROL_UNSIGNED_MODULE:
            return AEE_SUCCESS;
        case FASTRPC_GET_URI: {
            auto * u = (struct remote_rpc_get_uri *) data;
            if (!u->uri || u->uri_len == 0) {
                return AEE_EBADPARM;
            }
            snprintf(u->uri, u->uri_len, "%.*s&_dom=cdsp&_session=%u", (int) u->module_uri_len, u->module_uri,
                     u->session_id);
            return AEE_SUCCESS;
        }
        default:
            return AEE_EUNSUPPORTED;
    }
}

// Accepts every handle control request.
int remote_handle_control(uint32_t req, void * data, uint32_t datalen) {
    (void) req;
    (void) data;
    (void) datalen;
    return AEE_SUCCESS;
}

// Accepts every handle control request of a session.
int remote_handle64_control(remote_handle64 h, uint32_t req, void * data, uint32_t datalen) {
    (void) h;
    (void) req;
    (void) data;
    (void) datalen;
    return AEE_SUCCESS;
}

// Maps an rpcmem block. An unknown fd, a wrong base, or a length above the block size is a violation.
int fastrpc_mmap(int domain, int fd, void * addr, int offset, size_t length, enum fastrpc_map_flags flags) {
    (void) domain;
    (void) flags;
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_fd_base.find(fd);
    if (it == g_fd_base.end()) {
        violation("mmap-fd", "fastrpc_mmap of fd %d, which is not an rpcmem allocation", fd);
        return AEE_EBADPARM;
    }
    const alloc_info & info = g_allocs[it->second];
    if ((uint64_t) (uintptr_t) addr != it->second || offset != 0 || length > info.size) {
        violation("mmap-range", "fastrpc_mmap of fd %d with addr %p offset %d length %zu, the block is %p size %zu", fd,
                  addr, offset, length, (void *) (uintptr_t) it->second, info.size);
        return AEE_EBADPARM;
    }
    return AEE_SUCCESS;
}

// Unmaps an rpcmem block.
int fastrpc_munmap(int domain, int fd, void * addr, size_t length) {
    (void) domain;
    (void) fd;
    (void) addr;
    (void) length;
    return AEE_SUCCESS;
}

// Gives one NSP domain or an error, as the configuration selects.
int remote_system_request(system_req_payload * req) {
    if (!req || req->id != FASTRPC_GET_DOMAINS || g_cfg.discovery == 0) {
        return AEE_EUNSUPPORTEDAPI;
    }
    if (req->sys.domains == nullptr) {
        req->sys.num_domains = 1;
        return AEE_SUCCESS;
    }
    if (req->sys.max_domains >= 1) {
        fastrpc_domain & d = req->sys.domains[0];
        memset(&d, 0, sizeof(d));
        d.type        = FASTRPC_NSP;
        d.id          = CDSP_DOMAIN_ID;
        d.status      = 1;
        d.instance_id = 0;
        snprintf(d.name, sizeof(d.name), "cdsp");
        req->sys.num_domains = 1;
    }
    return AEE_SUCCESS;
}

// ---- htpdrv

int htpdrv_init(void) {
    return AEE_SUCCESS;
}

domain * htpdrv_get_domain(int domain_id) {
    static domain d = { CDSP_DOMAIN_ID, "cdsp" };
    return domain_id == CDSP_DOMAIN_ID ? &d : nullptr;
}

int htpdrv_get_arch(int dom, int * arch) {
    (void) dom;
    if (g_cfg.arch <= 0) {
        return AEE_EFAILED;
    }
    *arch = g_cfg.arch;
    return AEE_SUCCESS;
}

// ---- dspqueue

AEEResult dspqueue_create(int dom, uint32_t flags, uint32_t req_queue_size, uint32_t resp_queue_size,
                          dspqueue_callback_t packet_callback, dspqueue_callback_t error_callback,
                          void * callback_context, dspqueue_t * queue) {
    (void) dom;
    (void) flags;
    if (packet_callback) {
        violation("queue-callback", "dspqueue_create with a packet callback, the host must read the responses itself");
    }
    if (req_queue_size < sizeof(htp_opbatch_req) || resp_queue_size < sizeof(htp_opbatch_rsp)) {
        violation("queue-size", "dspqueue_create with queue sizes %u and %u", req_queue_size, resp_queue_size);
    }
    std::call_once(g_exit_once, [] { atexit(exit_notify); });
    std::lock_guard<std::mutex> lock(g_mu);
    auto q      = std::make_unique<queue_info>();
    q->id       = g_next_id++;
    q->error_cb = error_callback;
    q->cb_ctx   = callback_context;
    if (g_cfg.async) {
        q->dsp = std::thread(dsp_thread_main, q.get());
    }
    *queue = (dspqueue_t) (uintptr_t) q->id;
    g_queues[q->id] = std::move(q);
    return AEE_SUCCESS;
}

AEEResult dspqueue_close(dspqueue_t queue) {
    std::unique_ptr<queue_info> q;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        auto it = g_queues.find((uint64_t) (uintptr_t) queue);
        if (it == g_queues.end()) {
            violation("queue-close", "dspqueue_close of an unknown queue");
            return AEE_EBADPARM;
        }
        q = std::move(it->second);
        g_queues.erase(it);
    }
    {
        std::lock_guard<std::mutex> lock(q->mu);
        q->closing = true;
        if (!q->req.empty()) {
            violation("queue-close-pending", "dspqueue_close with %zu requests that the DSP did not read", q->req.size());
        }
        q->cv.notify_all();
    }
    if (q->dsp.joinable()) {
        q->dsp.join();
    }
    // As the dspqueue library, the close waits for the error callback
    if (q->error_thread.joinable()) {
        q->error_thread.join();
    }
    if (q->handle) {
        q->handle->queue = nullptr;
    }
    return AEE_SUCCESS;
}

AEEResult dspqueue_export(dspqueue_t queue, uint64_t * queue_id) {
    *queue_id = (uint64_t) (uintptr_t) queue;
    return AEE_SUCCESS;
}

AEEResult dspqueue_write(dspqueue_t queue, uint32_t flags, uint32_t num_buffers, struct dspqueue_buffer * buffers,
                         uint32_t message_length, const uint8_t * message, uint32_t timeout_us) {
    (void) flags;
    (void) timeout_us;
    queue_info * q;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        q = find_queue((uint64_t) (uintptr_t) queue);
    }
    if (!q) {
        violation("queue-write", "dspqueue_write to an unknown queue");
        return AEE_EBADPARM;
    }
    if (interrupted(q, g_writes)) {
        return AEE_EINTERRUPTED;
    }
    packet in;
    in.msg.assign(message, message + message_length);
    in.n_bufs = num_buffers;
    if (num_buffers >= 1) {
        in.buf = buffers[0];
    }
    if (g_cfg.async) {
        std::lock_guard<std::mutex> lock(q->mu);
        q->req.push_back(std::move(in));
        q->cv.notify_all();
        return AEE_SUCCESS;
    }
    packet out = process_request(q, in);
    std::lock_guard<std::mutex> lock(q->mu);
    q->rsp.push_back(std::move(out));
    return AEE_SUCCESS;
}

AEEResult dspqueue_read(dspqueue_t queue, uint32_t * flags, uint32_t max_buffers, uint32_t * num_buffers,
                        struct dspqueue_buffer * buffers, uint32_t max_message_length, uint32_t * message_length,
                        uint8_t * message, uint32_t timeout_us) {
    queue_info * q;
    {
        std::lock_guard<std::mutex> lock(g_mu);
        q = find_queue((uint64_t) (uintptr_t) queue);
    }
    if (!q) {
        violation("queue-read", "dspqueue_read of an unknown queue");
        return AEE_EBADPARM;
    }
    if (interrupted(q, g_reads)) {
        return AEE_EINTERRUPTED;
    }
    std::unique_lock<std::mutex> lock(q->mu);
    if (q->rsp.empty()) {
        if (!g_cfg.async) {
            // In sync mode each request has its response at once. An empty queue
            // here means that the host waits for a response that never comes.
            violation("queue-read-empty", "dspqueue_read with no response pending (the host waits forever)");
            return AEE_EEXPIRED;
        }
        if (timeout_us == 0) {
            return AEE_EWOULDBLOCK;
        }
        q->cv.wait_for(lock, std::chrono::seconds(20), [q] { return !q->rsp.empty(); });
        if (q->rsp.empty()) {
            violation("queue-read-timeout", "dspqueue_read waited 20 s for a response of the DSP thread");
            return AEE_EEXPIRED;
        }
    }
    packet out = std::move(q->rsp.front());
    q->rsp.pop_front();
    lock.unlock();

    *flags = 0;
    if (max_buffers >= 1) {
        buffers[0]   = out.buf;
        *num_buffers = 1;
    } else {
        *num_buffers = 0;
    }
    const uint32_t n = (uint32_t) out.msg.size() < max_message_length ? (uint32_t) out.msg.size() : max_message_length;
    memcpy(message, out.msg.data(), n);
    *message_length = n;
    return AEE_SUCCESS;
}

// ---- htp_iface

AEEResult htp_iface_open(const char * uri, remote_handle64 * handle) {
    if (!uri || strstr(uri, "libggml-htp-v") == nullptr) {
        violation("iface-uri", "htp_iface_open with the URI '%s'", uri ? uri : "(null)");
    }
    std::lock_guard<std::mutex> lock(g_mu);
    auto h = std::make_unique<handle_info>();
    h->id  = g_next_id++;
    *handle = h->id;
    g_handles[h->id] = std::move(h);
    return AEE_SUCCESS;
}

AEEResult htp_iface_close(remote_handle64 handle) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        violation("iface-close", "htp_iface_close of an unknown handle");
        return AEE_EBADPARM;
    }
    if (it->second->started) {
        violation("iface-close-started", "htp_iface_close of a session that the host did not stop");
    }
    g_handles.erase(it);
    return AEE_SUCCESS;
}

AEEResult htp_iface_start(remote_handle64 handle, uint32_t sess_id, uint64_t dsp_queue_id, uint32_t n_hvx,
                          uint32_t n_hmx, uint64_t max_vmem) {
    (void) sess_id;
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        return AEE_EBADPARM;
    }
    handle_info * h = it->second.get();
    queue_info *  q = find_queue(dsp_queue_id);
    if (!q) {
        violation("iface-start-queue", "htp_iface_start with an unknown queue id");
        return AEE_EBADPARM;
    }
    // The thread count of the DSP context, as htp_iface_start in main.c computes it
    uint32_t n = n_hvx == 0 ? g_cfg.n_threads : n_hvx;
    if (n > g_cfg.n_threads) {
        n = g_cfg.n_threads;
    }
    if (n > HTP_MAX_NTHREADS) {
        n = HTP_MAX_NTHREADS;
    }
    h->n_threads = n;
    h->n_hmx     = n_hmx ? g_cfg.n_hmx : 0;
    h->max_vmem  = max_vmem;
    h->queue     = q;
    h->started   = true;
    q->handle    = h;
    return AEE_SUCCESS;
}

AEEResult htp_iface_stop(remote_handle64 handle) {
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        return AEE_EBADPARM;
    }
    it->second->started = false;
    if (it->second->queue) {
        it->second->queue->handle = nullptr;
    }
    return AEE_SUCCESS;
}

AEEResult htp_iface_mmap(remote_handle64 handle, uint32_t fd, uint32_t size) {
    (void) handle;
    (void) fd;
    (void) size;
    return AEE_SUCCESS;
}

AEEResult htp_iface_munmap(remote_handle64 handle, uint32_t fd) {
    (void) handle;
    (void) fd;
    return AEE_SUCCESS;
}

AEEResult htp_iface_profiler(remote_handle64 handle, uint32_t mode, const htp_iface_pmu_conf * pmu) {
    (void) pmu;
    std::lock_guard<std::mutex> lock(g_mu);
    auto it = g_handles.find(handle);
    if (it == g_handles.end()) {
        return AEE_EBADPARM;
    }
    it->second->profiler = mode;
    return AEE_SUCCESS;
}

AEEResult htp_iface_etm(remote_handle64 handle, uint32_t enable) {
    (void) handle;
    (void) enable;
    return AEE_SUCCESS;
}

AEEResult htp_iface_hwinfo(remote_handle64 handle, unsigned int * n_threads, unsigned int * n_hvx,
                           unsigned int * n_hmx, unsigned long long * vtcm_size) {
    (void) handle;
    uint32_t n = g_cfg.n_threads > HTP_MAX_NTHREADS ? HTP_MAX_NTHREADS : g_cfg.n_threads;
    *n_threads = n;
    *n_hvx     = n;
    *n_hmx     = g_cfg.n_hmx;
    *vtcm_size = g_cfg.vtcm_size;
    return AEE_SUCCESS;
}

} // extern "C"
