// Probe: can vrdelta express the 3-way lane deinterleave and interleave that the conv step needs?
//
// The conv state of a channel is 3 consecutive floats, thus a block of 32 channels is 96 lanes in
// 3 vectors. The kernel wants the 3 tap planes (lane = channel) and, on the way out, the 96 lanes
// again. Every lane of a plane needs a different rotate amount, thus rotates and masks cost about
// 80 permutes per block, which is what the kernel of the checkout pays.
//
// vrdelta is a general within-vector byte permutation network: with the control byte i ^ j at each
// output byte i, the output byte i takes the input byte j. A permutation is realizable only when
// the network has no conflict, thus this program measures which of the needed patterns work.
//
// Arguments: none
#include "lab.h"

#include <stdio.h>
#include <string.h>

#include "hvx-utils.h"

#define TARGET "perm3"
#define NL     VLEN_FP32       // 32 float lanes in one vector

// The control vector of a float-lane gather: out lane i takes in lane src[i], or keeps whatever the
// network gives when src[i] < 0. All 4 bytes of a lane share the control 4i ^ 4j.
static HVX_Vector perm_ctrl(const int * src) {
    uint8_t buf[VLEN] __attribute__((aligned(VLEN)));
    memset(buf, 0, sizeof(buf));
    for (int i = 0; i < NL; i++) {
        const uint8_t c = (src[i] >= 0) ? (uint8_t) ((4 * i) ^ (4 * src[i])) : 0;
        for (int b = 0; b < 4; b++) {
            buf[4 * i + b] = c;
        }
    }
    return hvx_vmem(buf);
}

// Does vrdelta with that control really give the wanted gather? Returns the number of wrong lanes.
static int try_gather(const char * what, const int * src) {
    float in[NL]  __attribute__((aligned(VLEN)));
    float out[NL] __attribute__((aligned(VLEN)));
    for (int i = 0; i < NL; i++) {
        in[i] = (float) (1000 + i);
    }
    hvx_vmem(out) = Q6_V_vrdelta_VV(hvx_vmem(in), perm_ctrl(src));
    int bad = 0;
    for (int i = 0; i < NL; i++) {
        if (src[i] >= 0 && out[i] != (float) (1000 + src[i])) {
            bad++;
        }
    }
    printf("lab: %s %-34s %s (%d of %d wanted lanes wrong)\n", TARGET, what, bad ? "FAILS" : "works", bad, NL);
    return bad;
}

// The same question for vdelta, whose network runs the other way.
static int try_gather_d(const char * what, const int * src) {
    float in[NL]  __attribute__((aligned(VLEN)));
    float out[NL] __attribute__((aligned(VLEN)));
    for (int i = 0; i < NL; i++) {
        in[i] = (float) (1000 + i);
    }
    hvx_vmem(out) = Q6_V_vdelta_VV(hvx_vmem(in), perm_ctrl(src));
    int bad = 0;
    for (int i = 0; i < NL; i++) {
        if (src[i] >= 0 && out[i] != (float) (1000 + src[i])) {
            bad++;
        }
    }
    printf("lab: %s %-34s %s (%d of %d wanted lanes wrong)\n", TARGET, what, bad ? "FAILS" : "works", bad, NL);
    return bad;
}

int main(int argc, char ** argv) {
    (void) argc; (void) argv;
    lab_init();

    int src[NL];
    int bad = 0;

    // The three patterns that the kernel of the checkout builds with rotates and masks, all of
    // them inside one vector. A block holds 8 channels per window in the 3-lane form.
    for (int k = 0; k < NL; k++) src[k] = -1;
    for (int k = 0; k < 8; k++) for (int t = 0; t < 3; t++) src[4 * k + t] = 3 * k + t;
    bad += try_gather("expand 3 lanes to 4 (vrdelta)", src);
    bad += try_gather_d("expand 3 lanes to 4 (vdelta)", src);

    for (int k = 0; k < NL; k++) src[k] = -1;
    for (int k = 0; k < 8; k++) for (int t = 0; t < 3; t++) src[3 * k + t] = 4 * k + 1 + t;
    bad += try_gather("compact 4 lanes to 3 (vrdelta)", src);
    bad += try_gather_d("compact 4 lanes to 3 (vdelta)", src);

    for (int k = 0; k < NL; k++) src[k] = -1;
    for (int k = 0; k < 8; k++) src[4 * k + 3] = k;
    bad += try_gather("spread 1 lane into every 4th (vrdelta)", src);
    bad += try_gather_d("spread 1 lane into every 4th (vdelta)", src);

    // The whole-block forms, in case a 32-lane window is the wrong unit: 96 lanes of one block
    // sit in 3 vectors, thus a plane draws from each of them.
    for (int t = 0; t < 3; t++) {
        for (int v = 0; v < 3; v++) {
            char name[64];
            for (int c = 0; c < NL; c++) {
                const int L = 3 * c + t;
                src[c] = (L / NL == v) ? (L % NL) : -1;
            }
            snprintf(name, sizeof(name), "plane %d from vector %d (vdelta)", t, v);
            bad += try_gather_d(name, src);
        }
    }

    lab_report(TARGET, "wrong_lanes_total", (double) bad, "");
    return bad ? 1 : 0;
}
