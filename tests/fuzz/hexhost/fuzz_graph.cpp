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
// same graphs on HTP0 and on the CPU backend.

#include "fake_dsp.h"
#include "fuzz_death.h"
#include "graphgen.h"
#include "harness.h"
#include "hexhost.h"
#include "symexec.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <fuzzer/FuzzedDataProvider.h>

#include <cstdint>
#include <cstdlib>
#include <vector>

#ifndef HEXHOST_ASYNC_DSP
#define HEXHOST_ASYNC_DSP 0
#endif

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
            chk.run_device(fakedsp::take_batches());
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
