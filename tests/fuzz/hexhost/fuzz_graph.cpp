// fuzz_graph: the graph-level behavior of the host part of the Hexagon backend.
//
// Each input gives a session (the hardware of the fake DSP and the switches
// OPBATCH, OPQUEUE, OPPOLL, OPFUSION, OPFUSION_STATE, BATCHCACHE, GRAPHCACHE,
// PROFILE, HOSTPROF), a model of Qwen3.5 shape with random mutations
// (graphgen), and a sequence of steps. A step sets the inputs (the slot index
// of the recurrent cache can be valid or not), runs the splits of one graph
// with graph_compute, sometimes records and waits on an event, rebuilds a
// graph, and synchronizes. After each step the symbolic check (symexec)
// compares the caches and the outputs of the device with the meaning of the
// graph. The fake DSP checks each batch and each op on the way.
//
// With ThreadSanitizer (HEXHOST_ASYNC_DSP=1) each queue has a DSP thread that
// reads the inputs and writes the outputs of each op in real memory. In the
// other builds the DSP runs on the host thread when the host sends a batch,
// and for most inputs it also writes the outputs in real memory (the touch
// mode), thus a host read of reused bytes gets the value of a later tensor.
//
// The phone driver (phone/driver.cpp) reads the same input bytes and runs the
// same graphs on HTP0 and on the CPU backend. HEXHOST_PHONE_OPTIONS=1 gives the
// x86 run the switches and the hardware of such a phone run, and
// HEXHOST_OPFUSION=N sets the fusion switch.

#include "dsp_model.h"
#include "fake_dsp.h"
#include "fuzz_death.h"
#include "graphgen.h"
#include "harness.h"
#include "hexhost.h"
#include "symexec.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <fuzzer/FuzzedDataProvider.h>

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#ifndef HEXHOST_ASYNC_DSP
#define HEXHOST_ASYNC_DSP 0
#endif

namespace {

// Gives the names of the tensors of a world whose bytes hold the address: the caches and the
// owners of the bytes of each graph node. O(caches + nodes).
std::string names_at(const graphgen::world & w, uint64_t addr) {
    std::string s;
    auto        add = [&](const ggml_tensor * t) {
        const uint64_t a = (uint64_t) (uintptr_t) t->data;
        if (t->data && addr >= a && addr < a + ggml_nbytes(t)) {
            s += s.empty() ? "" : ",";
            s += t->name[0] ? t->name : "?";
        }
    };
    for (const ggml_tensor * c : w.caches) {
        add(c);
    }
    for (const auto & g : w.graphs) {
        for (const ggml_tensor * n : g.order) {
            if (!n->view_src) {
                add(n);
            }
        }
    }
    return s.empty() ? "-" : s;
}

// HEXHOST_DUMP_OPS=1: prints each op of the recorded batches with the byte range and the owner
// names of each input and output. O(ops * (caches + nodes)).
void dump_batches(const graphgen::world & w, uint32_t step, const std::vector<fakedsp::batch_record> & batches) {
    for (size_t b = 0; b < batches.size(); b++) {
        for (size_t i = 0; i < batches[b].ops.size(); i++) {
            const auto & op = batches[b].ops[i];
            printf("step %u batch %zu op %zu %s\n", step, b, i, fakedsp::opcode_name(op.opcode));
            auto one = [&](const char * kind, int k, const fakedsp::tensor_ref & t) {
                if (t.present) {
                    printf("    %s%d [0x%" PRIx64 ", 0x%" PRIx64 ") %u bytes flags 0x%x %s\n", kind, k, t.addr,
                           t.addr + fakedsp::tensor_extent(t), t.size, t.flags, names_at(w, t.addr).c_str());
                }
            };
            for (int k = 0; k < 10; k++) {
                one("src", k, op.src[k]);
            }
            for (int k = 0; k < 4; k++) {
                one("dst", k, op.dst[k]);
            }
        }
    }
    fflush(stdout);
}

} // namespace

extern "C" int LLVMFuzzerInitialize(int * argc, char *** argv) {
    (void) argc;
    (void) argv;
    harness::init();
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size) {
    fuzz_death_note_input(data, size);
    FuzzedDataProvider fdp(data, size);
    fakedsp::reset_record();

    fakedsp::config  cfg;
    hexhost::options o;
    graphgen::decode_session(fdp, HEXHOST_ASYNC_DSP != 0, cfg, o);
    // HEXHOST_PHONE_OPTIONS: the switches and the hardware of a phone driver run with no
    // GGML_HEXAGON_* variable (the defaults on the v79), in place of the values of the input. The
    // input bytes that decode_session reads do not change, thus the graph stays the same.
    if (getenv("HEXHOST_PHONE_OPTIONS")) {
        const size_t vmem = o.vmem;
        o                 = hexhost::options();
        o.vmem            = vmem;
        cfg.n_threads     = 6;
        cfg.n_hmx         = 1;
        cfg.vtcm_size     = 8u << 20;
        cfg.touch         = true;
    }
    // HEXHOST_OPFUSION: the fusion switch (GGML_HEXAGON_OPFUSION) in place of the value of the input
    if (const char * f = getenv("HEXHOST_OPFUSION")) {
        o.opfusion = atoi(f);
    }
    if (const char * f = getenv("HEXHOST_TOUCH_FILL")) {
        cfg.fill = (uint8_t) strtoul(f, nullptr, 0);
    }
    if (getenv("HEXHOST_STATS")) {
        o.hostprof = 1;
    }
    o.verbose = getenv("HEXHOST_VERBOSE") ? atoi(getenv("HEXHOST_VERBOSE")) : 0;
    fakedsp::configure(cfg);
    hexhost::set_options(o);

    hexhost::device * dev = hexhost::device_new();
    if (!hexhost::device_open(dev)) {
        hexhost::device_free(dev);
        return 0;
    }

    graphgen::world w;
    if (graphgen::build_world(fdp, dev, w)) {
        symexec::checker    chk(w);
        graphgen::step_hooks h;
        h.on_input  = [&](ggml_tensor * t, const std::vector<int32_t> & v, uint32_t step) { chk.set_input(t, v, step); };
        h.on_forget = [&](const graphgen::graph_spec & g) { chk.forget(g); };
        h.on_slots  = [](bool valid) { harness::expect_valid_slots(valid); };
        h.on_split  = [](const graphgen::graph_spec & g, size_t s, ggml_status st) {
            if (st != GGML_STATUS_SUCCESS) {
                fakedsp::violation("graph-compute-failed", "graph_compute of split %zu of %s gave status %d", s,
                                   g.desc.c_str(), (int) st);
            }
        };
        h.after_step = [&](graphgen::graph_spec & g, uint32_t step, bool valid) {
            if (o.hostprof) {
                uint32_t hits = 0, replays = 0, verified = 0;
                hexhost::session_counters(dev, &hits, &replays, &verified);
                fakedsp::count("graph cache hits", hits);
                fakedsp::count("batch replays", replays);
                fakedsp::count("batch replays verified", verified);
            }
            fakedsp::count("steps");
            std::vector<fakedsp::batch_record> batches = fakedsp::take_batches();
            if (getenv("HEXHOST_DUMP_OPS")) {
                dump_batches(w, step, batches);
            }
            chk.run_device(batches);
            chk.run_reference(g);
            if (!valid) {
                // A slot index that is not a row of the table has no defined result: the graph
                // reads outside the state table. The host checks and the DSP checks of the step
                // ran. The states of the two executions differ from here, thus the input stops.
                fakedsp::count("steps with a slot that is not valid");
                return false;
            }
            chk.compare({ &g }, step, g.desc);
            return true;
        };
        graphgen::run_steps(fdp, w, h);
    }
    graphgen::free_world(w);
    hexhost::device_free(dev);

    if (fakedsp::live_allocs() || fakedsp::live_queues() || fakedsp::live_handles()) {
        fakedsp::violation("leak", "after the device release %zu rpcmem blocks, %zu queues and %zu sessions are alive",
                           fakedsp::live_allocs(), fakedsp::live_queues(), fakedsp::live_handles());
    }
    return 0;
}
