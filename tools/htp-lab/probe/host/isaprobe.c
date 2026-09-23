// isaprobe: the host program of the ISA silicon probe. It runs on the ARM CPU of an Android phone
// and drives the Hexagon DSP in the unsigned PD through FastRPC.
//
// Usage (from adb shell, with ADSP_LIBRARY_PATH set to the directory of libisaprobe_skel.so):
//   isaprobe info [--allow-newer-skel]
//   isaprobe census <corpus_dir> <out_dir> [--only <text>] [--no-oracle] [--allow-newer-skel]
//   isaprobe kernel <id or name> <problem.bin> <out_dir> [--iters N] [--allow-newer-skel]
//
// "info" prints the chip version, the version of the DSP library, the result of the HVX, HMX and
// VTCM setup, and the candidate kernels of the library.
//
// "census" runs the HVX census of tools/htp-lab/isa (the ops whose names contain <text> for
// --only) on the corpus streams <corpus_dir>/corpus_<stream>.bin of the simulator run. For each
// op it writes <out_dir>/out_<op name>.bin, the DSP output in the file format of isa_kernels.h,
// and, when the op has an IEEE counterpart, <out_dir>/out_<op name>.oracle.bin, the result of the
// naive IEEE C code of host/oracle.c on the ARM CPU (source 2 in the header). The record is
// <out_dir>/isaprobe.txt. The op table comes from the DSP library, thus it is always the table of
// the kernels that run. An op that the library does not have (a v81 op in the v79 library) gets
// no output file, as in the simulator run of the same version.
//
// "kernel" runs one candidate kernel of dsp/candidates.c on a problem file of mv_case.py
// (mv_format.h), and the ggml scalar reference of q8_0_ref.c on the ARM CPU. It writes
// <out_dir>/<kernel name>.bin and <out_dir>/ggml-ref.bin and prints the comparison and the time.
//
// The version gate: a DSP library for a newer Hexagon version than the chip gives incorrect values
// with no error (a v81 library on the v79 chip). Thus the program stops with the exit code 3 when
// the library is newer than the chip, or when the chip version is unknown. --allow-newer-skel
// runs all the same, for the negative test, and the census record holds "negative_test 1".
//
// Exit codes: 0 success, 1 a usage, corpus or file error, 2 a FastRPC or setup error, 3 the
// version gate stopped the run, 4 at least one op or the kernel gave a nonzero status.

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "AEEStdErr.h"
#include "candidates.h"
#include "cdsprpc.h"
#include "isaprobe.h"
#include "isaprobe_status.h"
#include "mv_format.h"
#include "q8_0_ref.h"
#include "remote.h"

#ifdef ISAPROBE_HAVE_ISA_KERNELS_H
#    include "census_io.h"
#    include "isa_kernels.h"
#    include "oracle.h"
#endif

#define EXIT_USAGE   1
#define EXIT_RPC     2
#define EXIT_GATE    3
#define EXIT_OP_FAIL 4

#define RECORD_NAME "isaprobe.txt"
#define NAME_BYTES  256
#define MAX_REOPEN  3  // stop after this many DSP restarts in a row

// The DSP facts and the setup result that info() gives.
struct dsp_info {
    int      chip_arch;      // from the host query, for example 79, or 0 if unknown
    int      chip_arch_dsp;  // decoded from the DSP value, 0 if unknown
    uint32_t arch_raw;       // the raw value of qurt_sysenv_get_arch_version
    uint32_t skel_arch;      // the __HVX_ARCH__ of the library build
    uint32_t hvx_threads;
    uint32_t hmx_count;
    uint32_t vtcm_bytes;
    int32_t  setup_status;
    int32_t  setup_error;
    uint32_t n_ops;
    uint32_t n_streams;
};

// Return the AEE code of an error without the DSP offset 0x80000400, thus the host and the DSP
// forms of one error compare equal.
static int aee_base(int err) {
    const uint32_t u = (uint32_t) err;
    return (u & 0xFFFFFC00u) == 0x80000400u ? (int) (u & 0x3FFu) : err;
}

// Return a hint for a FastRPC error, or "" when the program knows no hint.
static const char * rpc_hint(int err) {
    switch (aee_base(err)) {
        case AEE_EUNABLETOLOAD & 0x3FF:
            return " (the DSP could not load libisaprobe_skel.so: check ADSP_LIBRARY_PATH, and that the file exists "
                   "and is a Hexagon library)";
        case AEE_ENOMEMORY & 0x3FF:
            return " (no memory on the DSP or for the FastRPC copy)";
        case AEE_EBADPARM & 0x3FF:
            return " (a bad parameter)";
        case AEE_EUNSUPPORTED & 0x3FF:
            return " (the library has the stub kernels and no census: build it with tools/htp-lab/isa/isa_kernels.c)";
        case AEE_ECONNRESET:
            return " (the DSP process stopped: an exception in the op, for example an instruction that this chip "
                   "does not have)";
        default:
            return "";
    }
}

// Return the name of a setup step of isaprobe_status.h.
static const char * setup_step_name(int step) {
    switch (step) {
        case ISAPROBE_SETUP_OK:           return "ok";
        case ISAPROBE_SETUP_APPTYPE:      return "power apptype vote";
        case ISAPROBE_SETUP_HVX_POWER:    return "HVX power vote";
        case ISAPROBE_SETUP_HMX_POWER:    return "HMX power vote";
        case ISAPROBE_SETUP_VTCM_QUERY:   return "VTCM query";
        case ISAPROBE_SETUP_ACQUIRE_HMX:  return "acquire of VTCM and HMX";
        case ISAPROBE_SETUP_ACQUIRE_VTCM: return "acquire of VTCM only";
        case ISAPROBE_SETUP_VTCM_PTR:     return "VTCM address";
        default:                          return "unknown step";
    }
}

// Return the name of a probe status, or "" for a value of isa_run.
static const char * probe_status_name(int status) {
    switch (status) {
        case ISAPROBE_STATUS_HVX_LOCK:  return " (the HVX lock failed, the op did not run)";
        case ISAPROBE_STATUS_HMX_LOCK:  return " (the HMX lock failed, the op did not run)";
        case ISAPROBE_STATUS_OVERRUN:   return " (the op wrote after the end of its output)";
        case ISAPROBE_STATUS_VTCM_FULL: return " (lab_vtcm_alloc had no VTCM for the op)";
        case -1:                        return " (the op is not in this build)";
        case -2:                        return " (the op ID is not valid)";
        default:                        return "";
    }
}

// Decode a version byte in the form 0x79 to 79. Returns 0 when a digit is not decimal.
static int decode_arch_byte(uint32_t v) {
    const uint32_t hi = (v >> 4) & 0xFu;
    const uint32_t lo = v & 0xFu;
    return (hi > 9 || lo > 9 || hi == 0) ? 0 : (int) (hi * 10 + lo);
}

// Query one attribute of the compute DSP. Returns 0 and the value, or the error of FastRPC.
static int dsp_capability(uint32_t attribute, uint32_t * value) {
    struct remote_dsp_capability cap = { .domain = CDSP_DOMAIN_ID, .attribute_ID = attribute, .capability = 0 };
    const int err = remote_handle_control(DSPRPC_GET_DSP_INFO, &cap, sizeof(cap));
    if (err == AEE_SUCCESS) {
        *value = cap.capability;
    }
    return err;
}

// Enable the unsigned PD and open the DSP library. Returns 0, or EXIT_RPC after a message.
static int probe_open(remote_handle64 * handle) {
    struct remote_rpc_control_unsigned_module u = { .domain = CDSP_DOMAIN_ID, .enable = 1 };
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &u, sizeof(u));
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "isaprobe: error: remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE) failed with 0x%08x. "
                        "This phone does not permit an unsigned PD on the compute DSP.\n", (unsigned) err);
        return EXIT_RPC;
    }

    const char * uri = isaprobe_URI CDSP_DOMAIN;
    err = isaprobe_open(uri, handle);
    if (err != AEE_SUCCESS) {
        const char * path = getenv("ADSP_LIBRARY_PATH");
        fprintf(stderr, "isaprobe: error: isaprobe_open(%s) failed with 0x%08x%s. ADSP_LIBRARY_PATH=%s\n", uri,
                (unsigned) err, rpc_hint(err), path != NULL ? path : "(not set)");
        return EXIT_RPC;
    }
    return 0;
}

// Close the DSP library. A failure only gives a message.
static void probe_close(remote_handle64 handle) {
    const int err = isaprobe_close(handle);
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "isaprobe: note: isaprobe_close failed with 0x%08x%s\n", (unsigned) err, rpc_hint(err));
    }
}

// Get and print the DSP facts. Returns 0, or EXIT_RPC after a message.
static int probe_info(remote_handle64 handle, struct dsp_info * info) {
    memset(info, 0, sizeof(*info));

    uint32_t cap = 0;
    int      err = dsp_capability(ARCH_VER, &cap);
    if (err == AEE_SUCCESS) {
        info->chip_arch = decode_arch_byte(cap & 0xFFu);
    } else {
        fprintf(stderr, "isaprobe: note: the host query of the DSP version failed with 0x%08x\n", (unsigned) err);
    }

    err = isaprobe_info(handle, &info->arch_raw, &info->skel_arch, &info->hvx_threads, &info->hmx_count,
                        &info->vtcm_bytes, &info->setup_status, &info->setup_error, &info->n_ops, &info->n_streams);
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "isaprobe: error: isaprobe_info failed with 0x%08x%s\n", (unsigned) err, rpc_hint(err));
        return EXIT_RPC;
    }
    info->chip_arch_dsp = decode_arch_byte(info->arch_raw & 0xFFu);

    char chip_text[16] = "unknown";
    char dsp_text[16]  = "not decoded";
    if (info->chip_arch != 0) {
        snprintf(chip_text, sizeof(chip_text), "v%d", info->chip_arch);
    }
    if (info->chip_arch_dsp != 0) {
        snprintf(dsp_text, sizeof(dsp_text), "v%d", info->chip_arch_dsp);
    }
    printf("isaprobe: chip %s (host query), DSP arch value 0x%08" PRIx32 " (%s)\n", chip_text, info->arch_raw,
           dsp_text);
    printf("isaprobe: library v%" PRIu32 ", census ops %" PRIu32 ", streams %" PRIu32 "\n", info->skel_arch,
           info->n_ops, info->n_streams);
    printf("isaprobe: hvx contexts %" PRIu32 ", hmx %" PRIu32 ", vtcm %" PRIu32 " bytes\n", info->hvx_threads,
           info->hmx_count, info->vtcm_bytes);
    printf("isaprobe: setup %s (step %d, error 0x%08x)\n", setup_step_name(info->setup_status),
           (int) info->setup_status, (unsigned) info->setup_error);

    if (info->chip_arch != 0 && info->chip_arch_dsp != 0 && info->chip_arch != info->chip_arch_dsp) {
        printf("isaprobe: note: the host query (v%d) and the DSP value (v%d) do not agree\n", info->chip_arch,
               info->chip_arch_dsp);
    }
    if (ISAPROBE_SETUP_IS_FATAL(info->setup_status)) {
        fprintf(stderr, "isaprobe: error: the %s failed with 0x%08x. The DSP library cannot run an op.\n",
                setup_step_name(info->setup_status), (unsigned) info->setup_error);
        return EXIT_RPC;
    }
    if (info->hmx_count == 0) {
        printf("isaprobe: note: the library holds no HMX. An HMX op stops the DSP process. Close the apps that "
               "use the NPU and run again.\n");
    }
    return 0;
}

// Return the chip version: the host query, else the DSP value, else 0.
static int chip_arch(const struct dsp_info * info) {
    return info->chip_arch != 0 ? info->chip_arch : info->chip_arch_dsp;
}

// Return 1 when the library can give incorrect values with no error: the library is for a newer
// DSP than the chip, or the chip version is unknown.
static int library_too_new(const struct dsp_info * info) {
    const int chip = chip_arch(info);
    return chip == 0 || (int) info->skel_arch > chip;
}

// Apply the version gate. Returns 0 when the ops can run, else EXIT_GATE after a message.
static int version_gate(const struct dsp_info * info, int allow_newer) {
    const int chip = chip_arch(info);
    const int skel = (int) info->skel_arch;

    if (library_too_new(info)) {
        const char * why = chip == 0 ? "the chip version is unknown" : "the library is for a newer DSP than the chip";
        if (!allow_newer) {
            fprintf(stderr, "isaprobe: error: %s (library v%d, chip v%d). A newer library gives incorrect values "
                            "with no error. Use the library for this chip, or give --allow-newer-skel for the "
                            "negative test.\n", why, skel, chip);
            return EXIT_GATE;
        }
        printf("isaprobe: WARNING: NEGATIVE TEST: %s (library v%d, chip v%d). Expect incorrect values.\n", why, skel,
               chip);
    } else if (skel < chip) {
        printf("isaprobe: note: the library v%d is older than the chip v%d. The chip runs the v%d instructions with "
               "its own rounding, not the rounding of a real v%d chip.\n", skel, chip, skel, skel);
    }
    return 0;
}

// ---- The census -------------------------------------------------------------------------------

#ifdef ISAPROBE_HAVE_ISA_KERNELS_H

// The corpus streams that the ops read. Each stream is loaded one time, at its first use.
struct corpus {
    const char * dir;
    uint8_t *    file[ISA_S_COUNT];   // the full file, NULL until loaded
    size_t       bytes[ISA_S_COUNT];  // the number of bytes of vectors after the header
};

// The counters and the files of one census run.
struct census_run {
    remote_handle64 * handle;
    struct corpus     corpus;
    const char *      out_dir;
    FILE *            record;
    uint32_t          skel_arch;
    int               oracle;  // 1: write out_<op>.oracle.bin for each op with an oracle
    int               n_ok;
    int               n_bad;
    int               n_unavailable;
    int               n_oracle;
    int               restarts;
};

// Load and check one corpus stream. Returns 0, or -1 after a message. O(size of the stream).
static int load_stream(struct census_run * run, uint32_t id) {
    struct corpus * c = &run->corpus;
    if (c->file[id] != NULL) {
        return 0;
    }
    char name[NAME_BYTES];
    int  err = isaprobe_stream_name(*run->handle, id, name, (int) sizeof(name));
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "isaprobe: error: isaprobe_stream_name(%" PRIu32 ") failed with 0x%08x%s\n", id,
                (unsigned) err, rpc_hint(err));
        return -1;
    }
    char file[NAME_BYTES + 16];
    char path[4096];
    snprintf(file, sizeof(file), "corpus_%s.bin", name);
    if (census_join(path, sizeof(path), c->dir, file) != 0) {
        return -1;
    }
    if (census_read_stream(path, id, &c->file[id], &c->bytes[id]) != 0) {
        fprintf(stderr, "isaprobe: error: the corpus needs a valid %s. Push the corpus files of the simulator run.\n",
                file);
        return -1;
    }
    return 0;
}

// Compute the oracle of one op and write out_<name>.oracle.bin. Returns 1 when written, 0 when
// the op has no oracle, -1 after a file error.
static int write_oracle(struct census_run * run, uint32_t op_id, const char * name, const isaprobe_op * d,
                        const uint8_t * const in[3], size_t out_len) {
    const struct oracle_op op = {
        .name      = name,
        .in_type   = { d->in_type[0], d->in_type[1], d->in_type[2] },
        .in_bytes  = { d->in_bytes[0], d->in_bytes[1], d->in_bytes[2] },
        .out_type  = d->out_type,
        .out_bytes = d->out_bytes,
        .n_vectors = d->n_vectors,
    };
    struct oracle_spec spec;
    if (!oracle_find(&op, &spec)) {
        return 0;
    }
    uint8_t * buf = (uint8_t *) calloc(out_len > 0 ? out_len : 1u, 1);
    if (buf == NULL) {
        fprintf(stderr, "isaprobe: error: no memory for the oracle of %s\n", name);
        return -1;
    }
    oracle_run(&op, &spec, in, buf);

    char file[NAME_BYTES + 16];
    char path[4096];
    snprintf(file, sizeof(file), "out_%s.oracle.bin", name);
    int ok = census_join(path, sizeof(path), run->out_dir, file) == 0 &&
             census_write_output(path, op_id, name, d->n_vectors, d->out_bytes, 0, CENSUS_SOURCE_ORACLE, buf, out_len) == 0;
    if (ok) {
        fprintf(run->record, "oracle %" PRIu32 " %s %s 0x%08" PRIx32 "\n", op_id, name, oracle_kind_name(&spec),
                isa_hash(buf, out_len));
    }
    free(buf);
    return ok ? 1 : -1;
}

// Run one op, write its output and its oracle, and record the result. Returns 0 to continue,
// EXIT_USAGE for a corpus or file error, or EXIT_RPC when the DSP process cannot restart.
static int census_op(struct census_run * run, uint32_t op_id, const char * only) {
    isaprobe_op d;
    char        name[NAME_BYTES];
    int         err = isaprobe_op_info(*run->handle, op_id, &d, name, (int) sizeof(name));
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "isaprobe: error: isaprobe_op_info(%" PRIu32 ") failed with 0x%08x%s\n", op_id, (unsigned) err,
                rpc_hint(err));
        return EXIT_RPC;
    }
    if (only != NULL && strstr(name, only) == NULL) {
        return 0;
    }
    if (strchr(name, '/') != NULL) {
        fprintf(stderr, "isaprobe: error: the op name %s has a \"/\"\n", name);
        return EXIT_USAGE;
    }
    if (!d.available) {
        fprintf(run->record, "op %" PRIu32 " %s unavailable\n", op_id, name);
        run->n_unavailable++;
        return 0;
    }

    const uint8_t * in[3]     = { NULL, NULL, NULL };
    size_t          in_len[3] = { 0, 0, 0 };
    for (int j = 0; j < 3; j++) {
        const uint32_t s = d.in_stream[j];
        if (s == 0 || d.in_bytes[j] == 0) {
            continue;
        }
        if (s >= ISA_S_COUNT) {
            fprintf(stderr, "isaprobe: error: op %s reads the stream %" PRIu32 ", and the census has %d streams\n", name,
                    s, ISA_S_COUNT);
            return EXIT_USAGE;
        }
        if (load_stream(run, s) != 0) {
            return EXIT_USAGE;
        }
        in_len[j] = (size_t) d.n_vectors * d.in_bytes[j];
        if (in_len[j] > run->corpus.bytes[s] || in_len[j] > INT_MAX) {
            fprintf(stderr, "isaprobe: error: op %s reads %zu bytes of input %d, and the stream has %zu bytes\n", name,
                    in_len[j], j, run->corpus.bytes[s]);
            return EXIT_USAGE;
        }
        in[j] = run->corpus.file[s] + sizeof(isa_file_header);
    }
    const size_t out_len = (size_t) d.n_vectors * d.out_bytes;
    if (out_len > INT_MAX || d.n_vectors > INT_MAX) {
        fprintf(stderr, "isaprobe: error: op %s has %zu output bytes, more than FastRPC permits\n", name, out_len);
        return EXIT_USAGE;
    }
    uint8_t * out = (uint8_t *) calloc(out_len > 0 ? out_len : 1u, 1);
    if (out == NULL) {
        fprintf(stderr, "isaprobe: error: no memory for the %zu output bytes of %s\n", out_len, name);
        return EXIT_USAGE;
    }

    // The line before the run tells which op was in progress when "timeout -s KILL" stops a hang.
    fprintf(run->record, "# run %" PRIu32 " %s\n", op_id, name);
    int32_t status = 0;
    err = isaprobe_run(*run->handle, (int32_t) op_id, in[0], (int) in_len[0], in[1], (int) in_len[1], in[2],
                       (int) in_len[2], out, (int) out_len, (int32_t) d.n_vectors, &status);
    int code = 0;
    if (err == AEE_SUCCESS && status == 0) {
        char file[NAME_BYTES + 16];
        char path[4096];
        snprintf(file, sizeof(file), "out_%s.bin", name);
        code = census_join(path, sizeof(path), run->out_dir, file) == 0 &&
                       census_write_output(path, op_id, name, d.n_vectors, d.out_bytes, run->skel_arch,
                                           CENSUS_SOURCE_CHIP, out, out_len) == 0 ?
                   0 :
                   EXIT_USAGE;
        if (code == 0) {
            fprintf(run->record, "op %" PRIu32 " %s ok 0x%08" PRIx32 "\n", op_id, name, isa_hash(out, out_len));
            run->n_ok++;
            run->restarts = 0;
        }
    } else if (err == AEE_SUCCESS) {
        printf("isaprobe: op %s (%" PRIu32 "): status %d%s\n", name, op_id, (int) status, probe_status_name(status));
        fprintf(run->record, "op %" PRIu32 " %s status %d\n", op_id, name, (int) status);
        run->n_bad++;
        run->restarts = 0;
    } else {
        fprintf(stderr, "isaprobe: error: op %s (%" PRIu32 "): isaprobe_run failed with 0x%08x%s\n", name, op_id,
                (unsigned) err, rpc_hint(err));
        fprintf(run->record, "op %" PRIu32 " %s rpc_error 0x%08x\n", op_id, name, (unsigned) err);
        run->n_bad++;
        // The DSP process can be dead. Open a new one for the next op.
        if (++run->restarts > MAX_REOPEN) {
            fprintf(stderr, "isaprobe: error: %d ops in a row failed. The run stops.\n", MAX_REOPEN + 1);
            code = EXIT_RPC;
        } else {
            probe_close(*run->handle);
            *run->handle = 0;
            code         = probe_open(run->handle);
            if (code == 0) {
                printf("isaprobe: note: a new DSP session is open after the failure of op %s\n", name);
            }
        }
    }
    free(out);

    // The oracle needs only the inputs, thus it runs also for an op that failed on the DSP.
    if (code == 0 && run->oracle) {
        const int w = write_oracle(run, op_id, name, &d, in, out_len);
        if (w < 0) {
            code = EXIT_USAGE;
        } else {
            run->n_oracle += w;
        }
    }
    return code;
}

// Write the header lines of the record.
static void record_header(FILE * f, const struct dsp_info * info, int negative_test, const char * only) {
    fprintf(f, "isaprobe_record 2\n");
    fprintf(f, "chip_arch %d\n", info->chip_arch);
    fprintf(f, "chip_arch_dsp_raw 0x%08" PRIx32 "\n", info->arch_raw);
    fprintf(f, "skel_arch %" PRIu32 "\n", info->skel_arch);
    fprintf(f, "hvx_contexts %" PRIu32 "\n", info->hvx_threads);
    fprintf(f, "hmx %" PRIu32 "\n", info->hmx_count);
    fprintf(f, "vtcm_bytes %" PRIu32 "\n", info->vtcm_bytes);
    fprintf(f, "setup_status %d\n", (int) info->setup_status);
    fprintf(f, "setup_error 0x%08x\n", (unsigned) info->setup_error);
    fprintf(f, "negative_test %d\n", negative_test);
    fprintf(f, "only %s\n", only != NULL ? only : "-");
    fprintf(f, "# op <id> <name> ok <hash> | status <isa_run value> | rpc_error <code> | unavailable\n");
    fprintf(f, "# oracle <id> <name> <operation> <hash>: out_<name>.oracle.bin, the ARM CPU oracle\n");
    fprintf(f, "# A \"# run <id> <name>\" line with no result after it names the op that did not finish.\n");
}

// Run the census. Returns the exit code of the program.
static int run_census(remote_handle64 * handle, const struct dsp_info * info, const char * corpus_dir,
                      const char * out_dir, const char * only, int negative_test, int oracle) {
    if (info->n_ops == 0) {
        fprintf(stderr, "isaprobe: error: the DSP library has the stub kernels and no census. Build it with "
                        "tools/htp-lab/isa/isa_kernels.c.\n");
        return EXIT_USAGE;
    }
    if (info->n_ops != ISA_N_OPS || info->n_streams != ISA_S_COUNT) {
        fprintf(stderr, "isaprobe: error: the library has %" PRIu32 " ops and %" PRIu32 " streams, the program %d and "
                        "%d. Build the two from the same isa_kernels.h (tools/htp-lab/probe/build.sh).\n",
                info->n_ops, info->n_streams, ISA_N_OPS, ISA_S_COUNT);
        return EXIT_USAGE;
    }

    char path[4096];
    if (census_join(path, sizeof(path), out_dir, RECORD_NAME) != 0) {
        return EXIT_USAGE;
    }
    struct census_run run;
    memset(&run, 0, sizeof(run));
    run.handle     = handle;
    run.corpus.dir = corpus_dir;
    run.out_dir    = out_dir;
    run.skel_arch  = info->skel_arch;
    run.oracle     = oracle;
    run.record     = fopen(path, "w");
    if (run.record == NULL) {
        fprintf(stderr, "isaprobe: error: cannot create the record %s: %s. Make the directory %s first.\n", path,
                strerror(errno), out_dir);
        return EXIT_USAGE;
    }
    // Each line reaches the file at once, thus a stop by "timeout -s KILL" keeps the record.
    setvbuf(run.record, NULL, _IOLBF, 0);
    record_header(run.record, info, negative_test, only);

    int code = 0;
    for (uint32_t k = 0; k < info->n_ops && code == 0; k++) {
        code = census_op(&run, k, only);
    }

    fprintf(run.record, "ops_ok %d\nops_not_ok %d\nops_unavailable %d\nops_oracle %d\n", run.n_ok, run.n_bad,
            run.n_unavailable, run.n_oracle);
    if (fclose(run.record) != 0) {
        fprintf(stderr, "isaprobe: error: cannot write the record in %s: %s\n", out_dir, strerror(errno));
        code = code != 0 ? code : EXIT_USAGE;
    }
    for (int s = 0; s < ISA_S_COUNT; s++) {
        free(run.corpus.file[s]);
    }
    printf("isaprobe: %d ops ok, %d not ok, %d not in this library, %d oracles, record %s/%s\n", run.n_ok, run.n_bad,
           run.n_unavailable, run.n_oracle, out_dir, RECORD_NAME);
    if (code == 0 && run.n_bad > 0) {
        code = EXIT_OP_FAIL;
    }
    return code;
}

#endif  // ISAPROBE_HAVE_ISA_KERNELS_H

// ---- The candidate kernels ----------------------------------------------------------------------

// Print the kernel table of the library. Returns 0, or EXIT_RPC after a message.
static int list_kernels(remote_handle64 handle) {
    uint32_t n  = 0;
    uint32_t op = 0;
    char     name[NAME_BYTES];
    int      err = isaprobe_kernel_info(handle, 0, &n, &op, name, (int) sizeof(name));
    if (err != AEE_SUCCESS && n != 0) {
        fprintf(stderr, "isaprobe: error: isaprobe_kernel_info failed with 0x%08x%s\n", (unsigned) err, rpc_hint(err));
        return EXIT_RPC;
    }
    printf("isaprobe: candidate kernels %" PRIu32 "\n", n);
    for (uint32_t k = 0; k < n; k++) {
        err = isaprobe_kernel_info(handle, k, &n, &op, name, (int) sizeof(name));
        if (err != AEE_SUCCESS) {
            fprintf(stderr, "isaprobe: error: isaprobe_kernel_info(%" PRIu32 ") failed with 0x%08x%s\n", k,
                    (unsigned) err, rpc_hint(err));
            return EXIT_RPC;
        }
        printf("isaprobe:   kernel %" PRIu32 " %s (op %" PRIu32 ")\n", k, name, op);
    }
    return 0;
}

// Find a kernel by its ID ("0") or its name. Returns 0 and the ID, op and name, or EXIT_USAGE.
static int find_kernel(remote_handle64 handle, const char * key, uint32_t * id, uint32_t * op, char * name, size_t size) {
    uint32_t n = 0;
    char *   end = NULL;
    errno        = 0;
    const unsigned long want = strtoul(key, &end, 10);
    const int           by_id = errno == 0 && end != key && *end == '\0';
    for (uint32_t k = 0; isaprobe_kernel_info(handle, k, &n, op, name, (int) size) == AEE_SUCCESS; k++) {
        if ((by_id && k == want) || (!by_id && strcmp(name, key) == 0)) {
            *id = k;
            return 0;
        }
    }
    fprintf(stderr, "isaprobe: error: the library has no kernel %s. \"isaprobe info\" lists the kernels.\n", key);
    return EXIT_USAGE;
}

// Return the time of the monotonic clock in microseconds.
static uint64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000u + (uint64_t) ts.tv_nsec / 1000u;
}

// Map an f32 bit pattern to an integer with the order of the values (+0 and -0 map to 0).
static int64_t ordered_f32(float f) {
    int32_t i;
    memcpy(&i, &f, 4);
    return i < 0 ? (int64_t) INT32_MIN - i : i;
}

// Write a harness result file. Returns 0, or -1 after a message.
static int write_result(const char * dir, const char * file, const mv_header * h, const float * y, uint32_t rows) {
    char path[4096];
    const int n = snprintf(path, sizeof(path), "%s/%s", dir, file);
    if (n < 0 || (size_t) n >= sizeof(path)) {
        fprintf(stderr, "isaprobe: error: the path %s/%s is too long\n", dir, file);
        return -1;
    }
    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "isaprobe: error: cannot create %s: %s\n", path, strerror(errno));
        return -1;
    }
    const int write_ok = fwrite(h, sizeof(*h), 1, f) == 1 && (rows == 0 || fwrite(y, sizeof(float), rows, f) == rows);
    const int close_ok = fclose(f) == 0;
    if (!write_ok || !close_ok) {
        fprintf(stderr, "isaprobe: error: cannot write %s: %s\n", path, strerror(errno));
        return -1;
    }
    return 0;
}

// Read and check a problem file. Returns 0 and the header and the full file, or EXIT_USAGE.
static int read_problem(const char * path, mv_header * h, uint8_t ** file) {
    uint8_t * buf = NULL;
    size_t    len = 0;
    FILE *    f   = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "isaprobe: error: cannot open the problem %s: %s\n", path, strerror(errno));
        return EXIT_USAGE;
    }
    long size = -1;
    if (fseek(f, 0, SEEK_END) == 0) {
        size = ftell(f);
    }
    if (size < (long) sizeof(*h) || fseek(f, 0, SEEK_SET) != 0 || (buf = (uint8_t *) malloc((size_t) size)) == NULL ||
        fread(buf, 1, (size_t) size, f) != (size_t) size) {
        fprintf(stderr, "isaprobe: error: cannot read the problem %s\n", path);
        fclose(f);
        free(buf);
        return EXIT_USAGE;
    }
    fclose(f);
    len = (size_t) size;
    memcpy(h, buf, sizeof(*h));

    const uint64_t w_bytes = (uint64_t) h->rows * (h->cols / QK8_0) * sizeof(block_q8_0);
    const uint64_t x_bytes = (uint64_t) h->cols * sizeof(float);
    const char *   why     = NULL;
    if (memcmp(h->magic, MV_MAGIC_PROBLEM, 4) != 0 || h->version != MV_VERSION) {
        why = "the magic or the version is not that of mv_format.h";
    } else if (h->op != CAND_OP_MATVEC_Q8_0) {
        why = "the op is not CAND_OP_MATVEC_Q8_0";
    } else if (h->rows == 0 || h->cols == 0 || h->cols % QK8_0 != 0) {
        why = "rows must be positive and cols a positive multiple of 32";
    } else if (sizeof(*h) + w_bytes + x_bytes != len || w_bytes > INT_MAX || x_bytes > INT_MAX) {
        why = "the file size is not the size of the header, the weights and the activation";
    }
    if (why != NULL) {
        fprintf(stderr, "isaprobe: error: %s is not a valid problem: %s\n", path, why);
        free(buf);
        return EXIT_USAGE;
    }
    *file = buf;
    return 0;
}

// Run a candidate kernel on the DSP and the ggml reference on the CPU, write the two results, and
// print the comparison. Returns the exit code of the program.
static int run_kernel(remote_handle64 handle, const char * key, const char * problem_path, const char * out_dir,
                      uint32_t iters) {
    uint32_t id = 0, op = 0;
    char     kname[NAME_BYTES];
    int      code = find_kernel(handle, key, &id, &op, kname, sizeof(kname));
    if (code != 0) {
        return code;
    }
    mv_header p;
    uint8_t * file = NULL;
    if ((code = read_problem(problem_path, &p, &file)) != 0) {
        return code;
    }
    if (op != p.op) {
        fprintf(stderr, "isaprobe: error: kernel %s solves op %" PRIu32 ", and the problem has op %" PRIu32 "\n", kname,
                op, p.op);
        free(file);
        return EXIT_USAGE;
    }
    const size_t        w_bytes = (size_t) p.rows * (p.cols / QK8_0) * sizeof(block_q8_0);
    const uint8_t *     w       = file + sizeof(p);
    const float *       x       = (const float *) (file + sizeof(p) + w_bytes);
    float *             y_dsp   = (float *) calloc(p.rows, sizeof(float));
    float *             y_cpu   = (float *) calloc(p.rows, sizeof(float));
    block_q8_0 *        q       = (block_q8_0 *) malloc((p.cols / QK8_0) * sizeof(block_q8_0));
    float *             x_copy  = (float *) malloc((size_t) p.cols * sizeof(float));
    if (y_dsp == NULL || y_cpu == NULL || q == NULL || x_copy == NULL) {
        fprintf(stderr, "isaprobe: error: no memory for the %" PRIu32 " x %" PRIu32 " problem\n", p.rows, p.cols);
        code = EXIT_USAGE;
        goto done;
    }
    memcpy(x_copy, x, (size_t) p.cols * sizeof(float));  // the file offset of x is not 4-byte aligned for all rows

    const isaprobe_kparams params = { .op = p.op, .rows = p.rows, .cols = p.cols, .iters = iters, .reserved = { 0 } };
    int32_t status  = 0;
    uint64  pcycles = 0, usecs = 0;  // the qaic type: unsigned long long on arm64
    int     err     = isaprobe_kernel_run(handle, id, &params, w, (int) w_bytes, (const uint8 *) x_copy,
                                           (int) (p.cols * sizeof(float)), (uint8 *) y_dsp, (int) (p.rows * sizeof(float)),
                                           &status, &pcycles, &usecs);
    if (err != AEE_SUCCESS) {
        fprintf(stderr, "isaprobe: error: isaprobe_kernel_run(%s) failed with 0x%08x%s\n", kname, (unsigned) err,
                rpc_hint(err));
        code = EXIT_RPC;
        goto done;
    }

    // The CPU oracle: the ggml scalar reference, the fastest of the iterations.
    uint64_t cpu_us = UINT64_MAX;
    for (uint32_t i = 0; i < iters; i++) {
        const uint64_t t0 = now_us();
        q8_0_ref_matvec((const block_q8_0 *) w, x_copy, y_cpu, q, p.rows, p.cols);
        const uint64_t dt = now_us() - t0;
        cpu_us            = dt < cpu_us ? dt : cpu_us;
    }

    mv_header r;
    memset(&r, 0, sizeof(r));
    memcpy(r.magic, MV_MAGIC_RESULT, 4);
    r.version   = MV_VERSION;
    r.op        = p.op;
    r.rows      = p.rows;
    r.cols      = p.cols;
    r.kernel_id = id;
    r.source    = MV_SOURCE_CHIP;
    r.iters     = iters;
    r.pcycles   = pcycles;
    r.usecs     = usecs;
    r.status    = status;
    r.seed      = p.seed;
    snprintf(r.name, sizeof(r.name), "%s", kname);
    char file_dsp[NAME_BYTES + 16];
    snprintf(file_dsp, sizeof(file_dsp), "%s.bin", kname);
    if (write_result(out_dir, file_dsp, &r, y_dsp, p.rows) != 0) {
        code = EXIT_USAGE;
        goto done;
    }
    r.kernel_id = MV_KERNEL_ORACLE;
    r.source    = MV_SOURCE_ORACLE;
    r.pcycles   = 0;
    r.usecs     = cpu_us;
    r.status    = 0;
    snprintf(r.name, sizeof(r.name), "%s", "ggml-ref");
    if (write_result(out_dir, "ggml-ref.bin", &r, y_cpu, p.rows) != 0) {
        code = EXIT_USAGE;
        goto done;
    }

    // The summary: the rows with the same bits, and the largest distance in f32 ulp.
    uint32_t same = 0;
    int64_t  worst = 0;
    uint32_t worst_row = 0;
    for (uint32_t i = 0; i < p.rows; i++) {
        const int64_t dist = llabs(ordered_f32(y_dsp[i]) - ordered_f32(y_cpu[i]));
        same += dist == 0;
        if (dist > worst) {
            worst     = dist;
            worst_row = i;
        }
    }
    printf("isaprobe: kernel %s: status %d, %" PRIu32 " x %" PRIu32 ", dsp %" PRIu64 " cycles %" PRIu64
           " us, cpu reference %" PRIu64 " us\n", kname, (int) status, p.rows, p.cols, (uint64_t) pcycles,
           (uint64_t) usecs, cpu_us);
    printf("isaprobe: kernel %s: %" PRIu32 " of %" PRIu32 " rows equal to the ggml reference, max %" PRId64
           " ulp (row %" PRIu32 ": dsp %.9g, cpu %.9g)\n", kname, same, p.rows, worst, worst_row,
           (double) y_dsp[worst_row], (double) y_cpu[worst_row]);
    code = status == 0 ? 0 : EXIT_OP_FAIL;

done:
    free(file);
    free(y_dsp);
    free(y_cpu);
    free(q);
    free(x_copy);
    return code;
}

// ---- main ----------------------------------------------------------------------------------------

// Print the usage text to stderr and return EXIT_USAGE.
static int usage(void) {
    fprintf(stderr, "usage: isaprobe info [--allow-newer-skel]\n"
                    "       isaprobe census <corpus_dir> <out_dir> [--only <text>] [--no-oracle] [--allow-newer-skel]\n"
                    "       isaprobe kernel <id or name> <problem.bin> <out_dir> [--iters N] [--allow-newer-skel]\n"
                    "Set ADSP_LIBRARY_PATH to the directory of libisaprobe_skel.so.\n");
    return EXIT_USAGE;
}

int main(int argc, char ** argv) {
    int          allow_newer = 0;
    int          oracle      = 1;
    long         iters       = 10;
    const char * only        = NULL;
    const char * args[4]     = { NULL, NULL, NULL, NULL };
    int          n_args      = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--allow-newer-skel") == 0) {
            allow_newer = 1;
        } else if (strcmp(argv[i], "--no-oracle") == 0) {
            oracle = 0;
        } else if (strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
            only = argv[++i];
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            char * end = NULL;
            iters      = strtol(argv[++i], &end, 10);
            if (*end != '\0' || iters < 1 || iters > 100000) {
                fprintf(stderr, "isaprobe: error: --iters needs a count in [1, 100000]\n");
                return EXIT_USAGE;
            }
        } else if (argv[i][0] == '-' || n_args == 4) {
            return usage();
        } else {
            args[n_args++] = argv[i];
        }
    }
    const int mode_info   = n_args == 1 && strcmp(args[0], "info") == 0;
    const int mode_census = n_args == 3 && strcmp(args[0], "census") == 0;
    const int mode_kernel = n_args == 4 && strcmp(args[0], "kernel") == 0;
    if (!mode_info && !mode_census && !mode_kernel) {
        return usage();
    }
#ifndef ISAPROBE_HAVE_ISA_KERNELS_H
    (void) only;
    (void) oracle;
    if (mode_census) {
        fprintf(stderr, "isaprobe: error: this program was built without tools/htp-lab/isa/isa_kernels.h and has no "
                        "census mode\n");
        return EXIT_USAGE;
    }
#endif

    // Each line reaches the log at once, thus a stop by "timeout -s KILL" keeps the lines before it.
    setvbuf(stdout, NULL, _IOLBF, 0);

    if (cdsprpc_load() != 0) {
        return EXIT_RPC;
    }
    uint32_t unsigned_pd = 0;
    if (dsp_capability(UNSIGNED_PD_SUPPORT, &unsigned_pd) == AEE_SUCCESS && unsigned_pd == 0) {
        fprintf(stderr, "isaprobe: error: the compute DSP of this phone has no unsigned PD support\n");
        return EXIT_RPC;
    }

    remote_handle64 handle = 0;
    int             code   = probe_open(&handle);
    if (code != 0) {
        return code;
    }

    struct dsp_info info;
    code = probe_info(handle, &info);
    if (code == 0 && mode_info) {
        code = list_kernels(handle);
    }
    if (code == 0) {
        code = version_gate(&info, allow_newer);
    }
#ifdef ISAPROBE_HAVE_ISA_KERNELS_H
    if (code == 0 && mode_census) {
        code = run_census(&handle, &info, args[1], args[2], only, library_too_new(&info), oracle);
    }
#endif
    if (code == 0 && mode_kernel) {
        code = run_kernel(handle, args[1], args[2], args[3], (uint32_t) iters);
    }

    if (handle != 0) {
        probe_close(handle);
    }
    return code;
}
