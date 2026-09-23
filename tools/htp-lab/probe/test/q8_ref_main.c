// Run the ggml scalar reference of the Q8_0 matvec (q8_0_ref.c) on a problem file of mv_case.py
// and write the rows f32 results. q8-ref-sim-test.sh builds it for x86-64 and for the Hexagon
// simulator, and compares the two outputs.
//
//   q8_ref_main <problem.bin> <out.bin>
//
// Exit code 0, or 1 after a message.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mv_format.h"
#include "q8_0_ref.h"

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: q8_ref_main <problem.bin> <out.bin>\n");
        return 1;
    }
    FILE * f = fopen(argv[1], "rb");
    if (f == NULL) {
        fprintf(stderr, "q8_ref_main: error: cannot open %s\n", argv[1]);
        return 1;
    }
    mv_header h;
    if (fread(&h, sizeof(h), 1, f) != 1 || memcmp(h.magic, MV_MAGIC_PROBLEM, 4) != 0 || h.cols % QK8_0 != 0) {
        fprintf(stderr, "q8_ref_main: error: %s is not a problem file\n", argv[1]);
        fclose(f);
        return 1;
    }
    const size_t n_w = (size_t) h.rows * (h.cols / QK8_0);
    block_q8_0 * w   = malloc(n_w * sizeof(block_q8_0));
    float *      x   = malloc((size_t) h.cols * sizeof(float));
    float *      y   = malloc((size_t) h.rows * sizeof(float));
    block_q8_0 * q   = malloc((h.cols / QK8_0) * sizeof(block_q8_0));
    int          ok  = w && x && y && q && fread(w, sizeof(block_q8_0), n_w, f) == n_w &&
               fread(x, sizeof(float), h.cols, f) == h.cols;
    fclose(f);
    if (ok) {
        q8_0_ref_matvec(w, x, y, q, h.rows, h.cols);
        FILE * o = fopen(argv[2], "wb");
        ok       = o != NULL && fwrite(y, sizeof(float), h.rows, o) == h.rows;
        ok       = (o != NULL && fclose(o) == 0) && ok;
    }
    if (!ok) {
        fprintf(stderr, "q8_ref_main: error: cannot compute or write the result\n");
    }
    free(w);
    free(x);
    free(y);
    free(q);
    return ok ? 0 : 1;
}
