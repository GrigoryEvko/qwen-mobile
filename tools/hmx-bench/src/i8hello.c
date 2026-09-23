// The smallest device program of the int8 HMX work: it checks that the lab runtime of dsp_lab.c
// starts in the unsigned protection domain, that the HVX runs, and that one int8 HMX product and
// its four-plane read give the reference value. Run it before the larger programs.
//
// Arguments: none. The results go to the log of the phone (logcat lines that contain "i8hello:").

#include "lab.h"

#include <stdio.h>

#include "HAP_farf.h"
#include "hexagon_protos.h"
#include "hexagon_types.h"

void lab_fini(void);

#define LOG(...)                             \
    do {                                     \
        printf("i8hello: " __VA_ARGS__);     \
        printf("\n");                        \
        FARF(ALWAYS, "i8hello: " __VA_ARGS__); \
    } while (0)

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;
    LOG("start");
    lab_init();
    LOG("lab_init done");

    uint8_t *  act = lab_vtcm_alloc(2048, 2048);
    uint8_t *  wgt = lab_vtcm_alloc(1024, 2048);
    uint8_t *  out = lab_vtcm_alloc(4 * 2048, 2048);
    uint32_t * tab = lab_vtcm_alloc(512, 512);

    // HVX: fill the activation with 3 and the weight with -2 (0xFE)
    *(HVX_Vector *) tab = Q6_V_vsplat_R(0);
    for (uint32_t i = 0; i < 2048; i += 128) {
        *(HVX_Vector *) (act + i) = Q6_Vb_vsplat_R(3);
    }
    for (uint32_t i = 0; i < 1024; i += 128) {
        *(HVX_Vector *) (wgt + i) = Q6_Vb_vsplat_R(0xFE);
    }
    LOG("hvx done");
    for (uint32_t b = 0; b < 4; b++) {
        for (uint32_t i = 0; i < 32; i++) {
            tab[b * 32 + i] = (24u - 8u * b) << 10;
        }
    }
    asm volatile("syncht\n" ::: "memory");

    // one product: each element is 32 * 3 * -2 = -192
    asm volatile("mxclracc\n" ::: "memory");
    asm volatile("{ activation.ub = mxmem(%0, %2):deep:cm\n weight.b = mxmem(%1, %3) }\n"
                 : : "r"(act), "r"(wgt), "r"(2047), "r"(1023) : "memory");
    for (uint32_t b = 0; b < 4; b++) {
        asm volatile("bias = mxmem(%0)\n" : : "r"(tab + b * 32) : "memory");
        asm volatile("mxmem(%0, %1):after:retain:cm.ub = acc\n" : : "r"(out + b * 2048), "r"(0) : "memory");
    }
    asm volatile("mxclracc\n" ::: "memory");
    asm volatile("syncht\n" ::: "memory");

    uint32_t bad = 0;
    for (uint32_t e = 0; e < 2048; e++) {
        const int32_t v = (int32_t) ((uint32_t) out[e] | ((uint32_t) out[2048 + e] << 8) |
                                     ((uint32_t) out[4096 + e] << 16) | ((uint32_t) out[6144 + e] << 24));
        bad += v != -192;
    }
    LOG("hmx product, %u of 2048 elements differ from -192", (unsigned) bad);
    lab_fini();
    LOG("done");
    return bad ? 1 : 0;
}
