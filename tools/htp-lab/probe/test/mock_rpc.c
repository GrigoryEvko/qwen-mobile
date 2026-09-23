// A mock of FastRPC and of the DSP library, for a test of the host program isaprobe on the build
// machine. It replaces host/cdsprpc.c and the qaic stub. Refer to host-test.sh.
//
// The environment controls the mock:
//   MOCK_TABLE       The op table of census_fixture.py (table.txt). Necessary for op_info and run.
//   MOCK_N_OPS       The op count of info(). The default is ISA_N_OPS. 0 acts as the stub kernels.
//   MOCK_CHIP_ARCH   The chip version of the host query, for example 79. 0 makes the query fail.
//   MOCK_DSP_ARCH    The raw value of qurt_sysenv_get_arch_version. The default is 0.
//   MOCK_SKEL_ARCH   The version of the DSP library, for example 81. The default is 79.
//   MOCK_CRASH_OP    An op ID. Each run of that op gives AEE_ECONNRESET, as a dead DSP process does.
//   MOCK_STATUS_OP   An op ID. Each run of that op gives the status -1 and no output.
//   MOCK_SETUP       The setup_status of info(). The default is 0.
//   MOCK_NO_HMX      1 gives hmx_count 0.
//   MOCK_KERNEL_ULP  Candidate kernel 0 adds this many f32 ulp to row 0 of its result. Default 0.
//   MOCK_KERNEL_STATUS  The status of kernel_run. Default 0.
// The op rule is the one of census_fixture.py. The only candidate kernel, 0 ("ref.q8_0_generic"),
// runs q8_0_ref.c, as the DSP library does.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "AEEStdErr.h"
#include "candidates.h"
#include "cdsprpc.h"
#include "isa_kernels.h"
#include "isaprobe.h"
#include "isaprobe_status.h"
#include "q8_0_ref.h"
#include "remote.h"

#define MOCK_HANDLE 0x1234u
#define MOCK_MAX    1024

struct mock_op {
    char     name[64];
    uint32_t available;
    uint32_t stream[3];
    uint32_t bytes[3];
    uint32_t out_bytes;
    uint32_t n_vectors;
    uint32_t out_type;
    uint32_t in_type[3];
};

static int            g_open_handles;
static struct mock_op g_ops[MOCK_MAX];
static int            g_n_table = -1;  // -1 until the table is read

// Return the integer value of an environment variable, or def when it is not set.
static long env_long(const char * name, long def) {
    const char * v = getenv(name);
    return v != NULL && v[0] != '\0' ? strtol(v, NULL, 0) : def;
}

// Read the op table of MOCK_TABLE one time. Returns the number of ops, or 0 without a table.
static int table(void) {
    if (g_n_table >= 0) {
        return g_n_table;
    }
    g_n_table         = 0;
    const char * path = getenv("MOCK_TABLE");
    FILE *       f    = path != NULL ? fopen(path, "r") : NULL;
    if (f == NULL) {
        fprintf(stderr, "mock: no op table (MOCK_TABLE=%s)\n", path != NULL ? path : "");
        return 0;
    }
    char line[512];
    while (fgets(line, sizeof(line), f) != NULL && g_n_table < MOCK_MAX) {
        if (line[0] == '#') {
            continue;
        }
        struct mock_op * op = &g_ops[g_n_table];
        const int n = sscanf(line, "%63s %u %u %u %u %u %u %u %u %u %u %u %u %u", op->name, &op->available,
                             &op->stream[0], &op->bytes[0], &op->stream[1], &op->bytes[1], &op->stream[2], &op->bytes[2],
                             &op->out_bytes, &op->n_vectors, &op->out_type, &op->in_type[0], &op->in_type[1],
                             &op->in_type[2]);
        if (n == 14) {
            g_n_table++;
        }
    }
    fclose(f);
    return g_n_table;
}

// Return the op count of info().
static uint32_t n_ops(void) {
    return (uint32_t) env_long("MOCK_N_OPS", ISA_N_OPS);
}

int cdsprpc_load(void) {
    return 0;
}

int remote_handle_control(uint32_t req, void * data, uint32_t datalen) {
    if (req != DSPRPC_GET_DSP_INFO || datalen != sizeof(struct remote_dsp_capability)) {
        return AEE_EUNSUPPORTED;
    }
    struct remote_dsp_capability * cap = (struct remote_dsp_capability *) data;
    if (cap->domain != CDSP_DOMAIN_ID) {
        return AEE_EBADPARM;
    }
    if (cap->attribute_ID == ARCH_VER) {
        const long arch = env_long("MOCK_CHIP_ARCH", 79);
        if (arch == 0) {
            return AEE_EUNSUPPORTED;
        }
        cap->capability = (uint32_t) (((arch / 10) << 4) | (arch % 10));
        return AEE_SUCCESS;
    }
    if (cap->attribute_ID == UNSIGNED_PD_SUPPORT) {
        cap->capability = 1;
        return AEE_SUCCESS;
    }
    return AEE_EUNSUPPORTED;
}

int remote_session_control(uint32_t req, void * data, uint32_t datalen) {
    if (req != DSPRPC_CONTROL_UNSIGNED_MODULE || datalen != sizeof(struct remote_rpc_control_unsigned_module)) {
        return AEE_EUNSUPPORTED;
    }
    const struct remote_rpc_control_unsigned_module * u = (const struct remote_rpc_control_unsigned_module *) data;
    return u->domain == CDSP_DOMAIN_ID && u->enable == 1 ? AEE_SUCCESS : AEE_EBADPARM;
}

int isaprobe_open(const char * uri, remote_handle64 * h) {
    const char * want = "file:///libisaprobe_skel.so?isaprobe_skel_handle_invoke&_modver=1.0&_dom=cdsp";
    if (strcmp(uri, want) != 0) {
        fprintf(stderr, "mock: unexpected URI %s\n", uri);
        return AEE_EBADPARM;
    }
    g_open_handles++;
    *h = MOCK_HANDLE;
    return AEE_SUCCESS;
}

int isaprobe_close(remote_handle64 h) {
    if (h != MOCK_HANDLE || g_open_handles <= 0) {
        fprintf(stderr, "mock: close of a handle that is not open\n");
        return AEE_EBADPARM;
    }
    g_open_handles--;
    return AEE_SUCCESS;
}

AEEResult isaprobe_info(remote_handle64 h, uint32 * arch, uint32 * skel_arch, uint32 * hvx_threads,
                        uint32 * hmx_count, uint32 * vtcm_bytes, int32 * setup_status, int32 * setup_error,
                        uint32 * n_ops_out, uint32 * n_streams) {
    if (h != MOCK_HANDLE) {
        return AEE_EBADPARM;
    }
    *arch         = (uint32) env_long("MOCK_DSP_ARCH", 0);
    *skel_arch    = (uint32) env_long("MOCK_SKEL_ARCH", 79);
    *hvx_threads  = 6;
    *hmx_count    = env_long("MOCK_NO_HMX", 0) ? 0 : 1;
    *vtcm_bytes   = 8u << 20;
    *setup_status = (int32) env_long("MOCK_SETUP", 0);
    *setup_error  = *setup_status != 0 ? AEE_EFAILED : 0;
    *n_ops_out    = n_ops();
    *n_streams    = n_ops() != 0 ? ISA_S_COUNT : 0;
    return AEE_SUCCESS;
}

AEEResult isaprobe_op_info(remote_handle64 h, uint32 op_id, isaprobe_op * desc, char * name, int nameLen) {
    if (h != MOCK_HANDLE) {
        return AEE_EBADPARM;
    }
    if (n_ops() == 0) {
        return AEE_EUNSUPPORTED;
    }
    if (op_id >= n_ops() || (int) op_id >= table()) {
        return AEE_EBADPARM;
    }
    const struct mock_op * op = &g_ops[op_id];
    memset(desc, 0, sizeof(*desc));
    for (int j = 0; j < 3; j++) {
        desc->in_stream[j] = op->stream[j];
        desc->in_bytes[j]  = op->bytes[j];
        desc->in_type[j]   = op->in_type[j];
    }
    desc->out_type  = op->out_type;
    desc->out_bytes = op->out_bytes;
    desc->n_vectors = op->n_vectors;
    desc->available = op->available;
    if ((int) strlen(op->name) >= nameLen) {
        return AEE_EBUFFERTOOSMALL;
    }
    strcpy(name, op->name);
    return AEE_SUCCESS;
}

AEEResult isaprobe_stream_name(remote_handle64 h, uint32 stream_id, char * name, int nameLen) {
    if (h != MOCK_HANDLE || stream_id >= ISA_S_COUNT || nameLen < 16) {
        return AEE_EBADPARM;
    }
    snprintf(name, (size_t) nameLen, "s%u", (unsigned) stream_id);
    return AEE_SUCCESS;
}

AEEResult isaprobe_run(remote_handle64 h, int32 op_id, const uint8 * in0, int in0Len, const uint8 * in1, int in1Len,
                       const uint8 * in2, int in2Len, uint8 * dst, int dstLen, int32 n_vectors, int32 * status) {
    if (h != MOCK_HANDLE || op_id < 0 || op_id >= table() || n_vectors < 0 || dstLen < 0) {
        return AEE_EBADPARM;
    }
    if (op_id == env_long("MOCK_CRASH_OP", -1)) {
        return AEE_ECONNRESET;
    }
    if (op_id == env_long("MOCK_STATUS_OP", -1)) {
        *status = -1;
        return AEE_SUCCESS;
    }

    // The program must send exactly the bytes that the op reads, and ask for its full output.
    const struct mock_op * op     = &g_ops[op_id];
    const uint8 *          in[3]  = { in0, in1, in2 };
    const int              len[3] = { in0Len, in1Len, in2Len };
    for (int j = 0; j < 3; j++) {
        const long want = op->stream[j] != 0 ? (long) n_vectors * op->bytes[j] : 0;
        if (len[j] != want || (want > 0 && in[j] == NULL)) {
            fprintf(stderr, "mock: op %d input %d has %d bytes, the op reads %ld\n", (int) op_id, j, len[j], want);
            *status = ISAPROBE_STATUS_OVERRUN;
            return AEE_SUCCESS;
        }
    }
    if ((uint32_t) n_vectors != op->n_vectors || dstLen != n_vectors * (int) op->out_bytes) {
        fprintf(stderr, "mock: op %d got %d vectors and %d output bytes\n", (int) op_id, (int) n_vectors, dstLen);
        *status = ISAPROBE_STATUS_OVERRUN;
        return AEE_SUCCESS;
    }

    uint8 * out = dst;
    for (int i = 0; i < n_vectors; i++) {
        uint8 block[128];
        for (int b = 0; b < 128; b++) {
            uint8 v = (uint8) (op_id & 0xFF);
            for (int j = 0; j < 3; j++) {
                if (op->stream[j] != 0) {
                    v ^= in[j][(size_t) i * op->bytes[j] + (size_t) b];
                }
            }
            block[b] = v;
        }
        memcpy(out, block, 128);
        out += 128;
        if (op->out_bytes == 256) {
            for (int b = 0; b < 128; b++) {
                out[b] = (uint8) ~block[b];
            }
            out += 128;
        }
    }
    *status = 0;
    return AEE_SUCCESS;
}

AEEResult isaprobe_kernel_info(remote_handle64 h, uint32 kernel_id, uint32 * n_kernels, uint32 * op, char * name,
                               int nameLen) {
    if (h != MOCK_HANDLE) {
        return AEE_EBADPARM;
    }
    *n_kernels = 1;
    if (kernel_id != 0) {
        return AEE_EBADPARM;
    }
    *op = CAND_OP_MATVEC_Q8_0;
    snprintf(name, (size_t) nameLen, "%s", "ref.q8_0_generic");
    return AEE_SUCCESS;
}

AEEResult isaprobe_kernel_run(remote_handle64 h, uint32 kernel_id, const isaprobe_kparams * params, const uint8 * in0,
                              int in0Len, const uint8 * in1, int in1Len, uint8 * dst, int dstLen, int32 * status,
                              uint64 * pcycles, uint64 * usecs) {
    if (h != MOCK_HANDLE || kernel_id != 0 || params->op != CAND_OP_MATVEC_Q8_0 || params->iters == 0 ||
        (long) in0Len != (long) params->rows * (params->cols / 32) * 34 || in1Len != (int) (params->cols * 4) ||
        dstLen != (int) (params->rows * 4)) {
        return AEE_EBADPARM;
    }
    *status = (int32) env_long("MOCK_KERNEL_STATUS", 0);
    if (*status != 0) {
        return AEE_SUCCESS;
    }
    // The DSP library copies the inputs into aligned buffers. The mock does the same.
    float *      x = (float *) malloc((size_t) in1Len);
    block_q8_0 * q = (block_q8_0 *) malloc((params->cols / 32) * sizeof(block_q8_0));
    if (x == NULL || q == NULL) {
        free(x);
        free(q);
        return AEE_ENOMEMORY;
    }
    memcpy(x, in1, (size_t) in1Len);
    float * y = (float *) malloc((size_t) dstLen);
    if (y == NULL) {
        free(x);
        free(q);
        return AEE_ENOMEMORY;
    }
    q8_0_ref_matvec((const block_q8_0 *) in0, x, y, q, params->rows, params->cols);
    uint32_t bits;
    memcpy(&bits, &y[0], 4);
    bits += (uint32_t) env_long("MOCK_KERNEL_ULP", 0);
    memcpy(&y[0], &bits, 4);
    memcpy(dst, y, (size_t) dstLen);
    free(x);
    free(q);
    free(y);
    *pcycles = 12345;
    *usecs   = 7;
    return AEE_SUCCESS;
}

// The check of the handles at the exit: each open needs one close. A leak changes the exit code
// to 99, thus the test of that case fails.
__attribute__((destructor)) static void mock_exit_check(void) {
    if (g_open_handles != 0) {
        fprintf(stderr, "mock: LEAK: %d handles are open at the exit\n", g_open_handles);
        _exit(99);
    }
}
