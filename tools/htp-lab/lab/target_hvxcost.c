// Target 7: the cost of each HVX op class on the v79 timing model, with written packets.
//
// A kernel design needs two numbers for each op class. The latency is the cycles that a dependent
// op waits for its producer: a chain of one stream shows it. The throughput is the cycles for
// each packet when no op waits: interleaved chains show it. Each case is a hardware loop of
// written packets, thus the compiler has no effect on the schedule. The assembler accepts only
// the packets that the v79 core accepts, thus each case is a legal schedule.
//
// The name suffix sN gives the number of independent chains. The file comes from a generator,
// refer to the header of each case for its packets.
//
// Arguments: --iters 2048
#include "lab.h"

#include <stdio.h>
#include <string.h>

typedef void (*cost_fn)(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_qf32_mul_s1\n .type cost_qf32_mul_s1,@function\ncost_qf32_mul_s1:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_qf32_mul_s1,r0)\n .p2align 5\n.Lc_qf32_mul_s1:\n"
        "    { v0.qf32 = vmpy(v0.qf32,v8.qf32) }\n"
        "    { v0.qf32 = vmpy(v0.qf32,v8.qf32) } :endloop0\n"
        " jumpr r31\n .size cost_qf32_mul_s1, .-cost_qf32_mul_s1\n");
void cost_qf32_mul_s1(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_qf32_mul_s2\n .type cost_qf32_mul_s2,@function\ncost_qf32_mul_s2:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_qf32_mul_s2,r0)\n .p2align 5\n.Lc_qf32_mul_s2:\n"
        "    { v0.qf32 = vmpy(v0.qf32,v8.qf32) }\n"
        "    { v1.qf32 = vmpy(v1.qf32,v8.qf32) } :endloop0\n"
        " jumpr r31\n .size cost_qf32_mul_s2, .-cost_qf32_mul_s2\n");
void cost_qf32_mul_s2(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_qf32_mul_s4\n .type cost_qf32_mul_s4,@function\ncost_qf32_mul_s4:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_qf32_mul_s4,r0)\n .p2align 5\n.Lc_qf32_mul_s4:\n"
        "    { v0.qf32 = vmpy(v0.qf32,v8.qf32) }\n"
        "    { v1.qf32 = vmpy(v1.qf32,v8.qf32) }\n"
        "    { v2.qf32 = vmpy(v2.qf32,v8.qf32) }\n"
        "    { v3.qf32 = vmpy(v3.qf32,v8.qf32) } :endloop0\n"
        " jumpr r31\n .size cost_qf32_mul_s4, .-cost_qf32_mul_s4\n");
void cost_qf32_mul_s4(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_qf32_mul_s8\n .type cost_qf32_mul_s8,@function\ncost_qf32_mul_s8:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_qf32_mul_s8,r0)\n .p2align 5\n.Lc_qf32_mul_s8:\n"
        "    { v0.qf32 = vmpy(v0.qf32,v8.qf32) }\n"
        "    { v1.qf32 = vmpy(v1.qf32,v8.qf32) }\n"
        "    { v2.qf32 = vmpy(v2.qf32,v8.qf32) }\n"
        "    { v3.qf32 = vmpy(v3.qf32,v8.qf32) }\n"
        "    { v4.qf32 = vmpy(v4.qf32,v8.qf32) }\n"
        "    { v5.qf32 = vmpy(v5.qf32,v8.qf32) }\n"
        "    { v6.qf32 = vmpy(v6.qf32,v8.qf32) }\n"
        "    { v7.qf32 = vmpy(v7.qf32,v8.qf32) } :endloop0\n"
        " jumpr r31\n .size cost_qf32_mul_s8, .-cost_qf32_mul_s8\n");
void cost_qf32_mul_s8(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_qf16_mul_s1\n .type cost_qf16_mul_s1,@function\ncost_qf16_mul_s1:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_qf16_mul_s1,r0)\n .p2align 5\n.Lc_qf16_mul_s1:\n"
        "    { v0.qf16 = vmpy(v0.qf16,v9.qf16) }\n"
        "    { v0.qf16 = vmpy(v0.qf16,v9.qf16) } :endloop0\n"
        " jumpr r31\n .size cost_qf16_mul_s1, .-cost_qf16_mul_s1\n");
void cost_qf16_mul_s1(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_qf16_mul_s2\n .type cost_qf16_mul_s2,@function\ncost_qf16_mul_s2:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_qf16_mul_s2,r0)\n .p2align 5\n.Lc_qf16_mul_s2:\n"
        "    { v0.qf16 = vmpy(v0.qf16,v9.qf16) }\n"
        "    { v1.qf16 = vmpy(v1.qf16,v9.qf16) } :endloop0\n"
        " jumpr r31\n .size cost_qf16_mul_s2, .-cost_qf16_mul_s2\n");
void cost_qf16_mul_s2(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_qf16_mul_s4\n .type cost_qf16_mul_s4,@function\ncost_qf16_mul_s4:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_qf16_mul_s4,r0)\n .p2align 5\n.Lc_qf16_mul_s4:\n"
        "    { v0.qf16 = vmpy(v0.qf16,v9.qf16) }\n"
        "    { v1.qf16 = vmpy(v1.qf16,v9.qf16) }\n"
        "    { v2.qf16 = vmpy(v2.qf16,v9.qf16) }\n"
        "    { v3.qf16 = vmpy(v3.qf16,v9.qf16) } :endloop0\n"
        " jumpr r31\n .size cost_qf16_mul_s4, .-cost_qf16_mul_s4\n");
void cost_qf16_mul_s4(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_qf32_add_s1\n .type cost_qf32_add_s1,@function\ncost_qf32_add_s1:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_qf32_add_s1,r0)\n .p2align 5\n.Lc_qf32_add_s1:\n"
        "    { v0.qf32 = vadd(v0.qf32,v8.qf32) }\n"
        "    { v0.qf32 = vadd(v0.qf32,v8.qf32) } :endloop0\n"
        " jumpr r31\n .size cost_qf32_add_s1, .-cost_qf32_add_s1\n");
void cost_qf32_add_s1(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_qf32_add_s2\n .type cost_qf32_add_s2,@function\ncost_qf32_add_s2:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_qf32_add_s2,r0)\n .p2align 5\n.Lc_qf32_add_s2:\n"
        "    { v0.qf32 = vadd(v0.qf32,v8.qf32); v1.qf32 = vadd(v1.qf32,v8.qf32) }\n"
        "    { v0.qf32 = vadd(v0.qf32,v8.qf32); v1.qf32 = vadd(v1.qf32,v8.qf32) } :endloop0\n"
        " jumpr r31\n .size cost_qf32_add_s2, .-cost_qf32_add_s2\n");
void cost_qf32_add_s2(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_qf32_add_s4\n .type cost_qf32_add_s4,@function\ncost_qf32_add_s4:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_qf32_add_s4,r0)\n .p2align 5\n.Lc_qf32_add_s4:\n"
        "    { v0.qf32 = vadd(v0.qf32,v8.qf32); v1.qf32 = vadd(v1.qf32,v8.qf32) }\n"
        "    { v2.qf32 = vadd(v2.qf32,v8.qf32); v3.qf32 = vadd(v3.qf32,v8.qf32) } :endloop0\n"
        " jumpr r31\n .size cost_qf32_add_s4, .-cost_qf32_add_s4\n");
void cost_qf32_add_s4(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_qf32_add_s8\n .type cost_qf32_add_s8,@function\ncost_qf32_add_s8:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_qf32_add_s8,r0)\n .p2align 5\n.Lc_qf32_add_s8:\n"
        "    { v0.qf32 = vadd(v0.qf32,v8.qf32); v1.qf32 = vadd(v1.qf32,v8.qf32) }\n"
        "    { v2.qf32 = vadd(v2.qf32,v8.qf32); v3.qf32 = vadd(v3.qf32,v8.qf32) }\n"
        "    { v4.qf32 = vadd(v4.qf32,v8.qf32); v5.qf32 = vadd(v5.qf32,v8.qf32) }\n"
        "    { v6.qf32 = vadd(v6.qf32,v8.qf32); v7.qf32 = vadd(v7.qf32,v8.qf32) } :endloop0\n"
        " jumpr r31\n .size cost_qf32_add_s8, .-cost_qf32_add_s8\n");
void cost_qf32_add_s8(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_i16_mul_s1\n .type cost_i16_mul_s1,@function\ncost_i16_mul_s1:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_i16_mul_s1,r0)\n .p2align 5\n.Lc_i16_mul_s1:\n"
        "    { v0.h = vmpy(v0.h,v9.h):<<1:rnd:sat }\n"
        "    { v0.h = vmpy(v0.h,v9.h):<<1:rnd:sat } :endloop0\n"
        " jumpr r31\n .size cost_i16_mul_s1, .-cost_i16_mul_s1\n");
void cost_i16_mul_s1(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_i16_mul_s2\n .type cost_i16_mul_s2,@function\ncost_i16_mul_s2:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_i16_mul_s2,r0)\n .p2align 5\n.Lc_i16_mul_s2:\n"
        "    { v0.h = vmpy(v0.h,v9.h):<<1:rnd:sat; v1.h = vmpy(v1.h,v9.h):<<1:rnd:sat }\n"
        "    { v0.h = vmpy(v0.h,v9.h):<<1:rnd:sat; v1.h = vmpy(v1.h,v9.h):<<1:rnd:sat } :endloop0\n"
        " jumpr r31\n .size cost_i16_mul_s2, .-cost_i16_mul_s2\n");
void cost_i16_mul_s2(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_i16_mul_s4\n .type cost_i16_mul_s4,@function\ncost_i16_mul_s4:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_i16_mul_s4,r0)\n .p2align 5\n.Lc_i16_mul_s4:\n"
        "    { v0.h = vmpy(v0.h,v9.h):<<1:rnd:sat; v1.h = vmpy(v1.h,v9.h):<<1:rnd:sat }\n"
        "    { v2.h = vmpy(v2.h,v9.h):<<1:rnd:sat; v3.h = vmpy(v3.h,v9.h):<<1:rnd:sat } :endloop0\n"
        " jumpr r31\n .size cost_i16_mul_s4, .-cost_i16_mul_s4\n");
void cost_i16_mul_s4(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_i16_mul_s8\n .type cost_i16_mul_s8,@function\ncost_i16_mul_s8:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_i16_mul_s8,r0)\n .p2align 5\n.Lc_i16_mul_s8:\n"
        "    { v0.h = vmpy(v0.h,v9.h):<<1:rnd:sat; v1.h = vmpy(v1.h,v9.h):<<1:rnd:sat }\n"
        "    { v2.h = vmpy(v2.h,v9.h):<<1:rnd:sat; v3.h = vmpy(v3.h,v9.h):<<1:rnd:sat }\n"
        "    { v4.h = vmpy(v4.h,v9.h):<<1:rnd:sat; v5.h = vmpy(v5.h,v9.h):<<1:rnd:sat }\n"
        "    { v6.h = vmpy(v6.h,v9.h):<<1:rnd:sat; v7.h = vmpy(v7.h,v9.h):<<1:rnd:sat } :endloop0\n"
        " jumpr r31\n .size cost_i16_mul_s8, .-cost_i16_mul_s8\n");
void cost_i16_mul_s8(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_alu_s1\n .type cost_alu_s1,@function\ncost_alu_s1:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_alu_s1,r0)\n .p2align 5\n.Lc_alu_s1:\n"
        "    { v0.h = vadd(v0.h,v9.h) }\n"
        "    { v0.h = vadd(v0.h,v9.h) } :endloop0\n"
        " jumpr r31\n .size cost_alu_s1, .-cost_alu_s1\n");
void cost_alu_s1(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_alu_s4\n .type cost_alu_s4,@function\ncost_alu_s4:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_alu_s4,r0)\n .p2align 5\n.Lc_alu_s4:\n"
        "    { v0.h = vadd(v0.h,v9.h); v1.h = vadd(v1.h,v9.h); v2.h = vadd(v2.h,v9.h); v3.h = vadd(v3.h,v9.h) }\n"
        "    { v0.h = vadd(v0.h,v9.h); v1.h = vadd(v1.h,v9.h); v2.h = vadd(v2.h,v9.h); v3.h = vadd(v3.h,v9.h) } :endloop0\n"
        " jumpr r31\n .size cost_alu_s4, .-cost_alu_s4\n");
void cost_alu_s4(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_alu_s8\n .type cost_alu_s8,@function\ncost_alu_s8:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_alu_s8,r0)\n .p2align 5\n.Lc_alu_s8:\n"
        "    { v0.h = vadd(v0.h,v9.h); v1.h = vadd(v1.h,v9.h); v2.h = vadd(v2.h,v9.h); v3.h = vadd(v3.h,v9.h) }\n"
        "    { v4.h = vadd(v4.h,v9.h); v5.h = vadd(v5.h,v9.h); v6.h = vadd(v6.h,v9.h); v7.h = vadd(v7.h,v9.h) } :endloop0\n"
        " jumpr r31\n .size cost_alu_s8, .-cost_alu_s8\n");
void cost_alu_s8(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_horner_qf16_s1\n .type cost_horner_qf16_s1,@function\ncost_horner_qf16_s1:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_horner_qf16_s1,r0)\n .p2align 5\n.Lc_horner_qf16_s1:\n"
        "    { v0.qf16 = vmpy(v0.qf16,v9.qf16) }\n"
        "    { v0.qf16 = vadd(v0.qf16,v10.qf16) } :endloop0\n"
        " jumpr r31\n .size cost_horner_qf16_s1, .-cost_horner_qf16_s1\n");
void cost_horner_qf16_s1(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_horner_qf16_s2\n .type cost_horner_qf16_s2,@function\ncost_horner_qf16_s2:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_horner_qf16_s2,r0)\n .p2align 5\n.Lc_horner_qf16_s2:\n"
        "    { v0.qf16 = vmpy(v0.qf16,v9.qf16); v1.qf16 = vadd(v1.qf16,v10.qf16) }\n"
        "    { v1.qf16 = vmpy(v1.qf16,v9.qf16); v0.qf16 = vadd(v0.qf16,v10.qf16) } :endloop0\n"
        " jumpr r31\n .size cost_horner_qf16_s2, .-cost_horner_qf16_s2\n");
void cost_horner_qf16_s2(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_horner_qf16_s4\n .type cost_horner_qf16_s4,@function\ncost_horner_qf16_s4:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_horner_qf16_s4,r0)\n .p2align 5\n.Lc_horner_qf16_s4:\n"
        "    { v0.qf16 = vmpy(v0.qf16,v9.qf16); v2.qf16 = vadd(v2.qf16,v10.qf16) }\n"
        "    { v1.qf16 = vmpy(v1.qf16,v9.qf16); v3.qf16 = vadd(v3.qf16,v10.qf16) }\n"
        "    { v2.qf16 = vmpy(v2.qf16,v9.qf16); v0.qf16 = vadd(v0.qf16,v10.qf16) }\n"
        "    { v3.qf16 = vmpy(v3.qf16,v9.qf16); v1.qf16 = vadd(v1.qf16,v10.qf16) } :endloop0\n"
        " jumpr r31\n .size cost_horner_qf16_s4, .-cost_horner_qf16_s4\n");
void cost_horner_qf16_s4(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_horner_i16_s1\n .type cost_horner_i16_s1,@function\ncost_horner_i16_s1:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_horner_i16_s1,r0)\n .p2align 5\n.Lc_horner_i16_s1:\n"
        "    { v0.h = vmpy(v0.h,v9.h):<<1:rnd:sat }\n"
        "    { v0.h = vadd(v0.h,v10.h) } :endloop0\n"
        " jumpr r31\n .size cost_horner_i16_s1, .-cost_horner_i16_s1\n");
void cost_horner_i16_s1(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_horner_i16_s2\n .type cost_horner_i16_s2,@function\ncost_horner_i16_s2:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_horner_i16_s2,r0)\n .p2align 5\n.Lc_horner_i16_s2:\n"
        "    { v0.h = vmpy(v0.h,v9.h):<<1:rnd:sat; v1.h = vadd(v1.h,v10.h) }\n"
        "    { v1.h = vmpy(v1.h,v9.h):<<1:rnd:sat; v0.h = vadd(v0.h,v10.h) } :endloop0\n"
        " jumpr r31\n .size cost_horner_i16_s2, .-cost_horner_i16_s2\n");
void cost_horner_i16_s2(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_horner_i16_s4\n .type cost_horner_i16_s4,@function\ncost_horner_i16_s4:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_horner_i16_s4,r0)\n .p2align 5\n.Lc_horner_i16_s4:\n"
        "    { v0.h = vmpy(v0.h,v9.h):<<1:rnd:sat; v1.h = vmpy(v1.h,v9.h):<<1:rnd:sat; v2.h = vadd(v2.h,v10.h); v3.h = vadd(v3.h,v10.h) }\n"
        "    { v2.h = vmpy(v2.h,v9.h):<<1:rnd:sat; v3.h = vmpy(v3.h,v9.h):<<1:rnd:sat; v0.h = vadd(v0.h,v10.h); v1.h = vadd(v1.h,v10.h) } :endloop0\n"
        " jumpr r31\n .size cost_horner_i16_s4, .-cost_horner_i16_s4\n");
void cost_horner_i16_s4(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_whf_s1\n .type cost_whf_s1,@function\ncost_whf_s1:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_whf_s1,r0)\n .p2align 5\n.Lc_whf_s1:\n"
        "    { v1:0.qf32 = vmpy(v2.hf,v9.hf) }\n"
        "    { v2.hf = v1:0.qf32 } :endloop0\n"
        " jumpr r31\n .size cost_whf_s1, .-cost_whf_s1\n");
void cost_whf_s1(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_whf_s2\n .type cost_whf_s2,@function\ncost_whf_s2:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_whf_s2,r0)\n .p2align 5\n.Lc_whf_s2:\n"
        "    { v1:0.qf32 = vmpy(v2.hf,v9.hf) }\n"
        "    { v5:4.qf32 = vmpy(v6.hf,v9.hf) }\n"
        "    { v2.hf = v1:0.qf32; v6.hf = v5:4.qf32 } :endloop0\n"
        " jumpr r31\n .size cost_whf_s2, .-cost_whf_s2\n");
void cost_whf_s2(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_whf_s4\n .type cost_whf_s4,@function\ncost_whf_s4:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_whf_s4,r0)\n .p2align 5\n.Lc_whf_s4:\n"
        "    { v1:0.qf32 = vmpy(v2.hf,v9.hf) }\n"
        "    { v5:4.qf32 = vmpy(v6.hf,v9.hf) }\n"
        "    { v13:12.qf32 = vmpy(v14.hf,v9.hf) }\n"
        "    { v17:16.qf32 = vmpy(v18.hf,v9.hf) }\n"
        "    { v2.hf = v1:0.qf32; v6.hf = v5:4.qf32 }\n"
        "    { v14.hf = v13:12.qf32; v18.hf = v17:16.qf32 } :endloop0\n"
        " jumpr r31\n .size cost_whf_s4, .-cost_whf_s4\n");
void cost_whf_s4(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_tap_acc\n .type cost_tap_acc,@function\ncost_tap_acc:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_tap_acc,r0)\n .p2align 5\n.Lc_tap_acc:\n"
        "    { v1:0.qf32 = vmpy(v2.hf,v9.hf) }\n"
        "    { v5:4.qf32 = vmpy(v6.hf,v9.hf); v20.qf32 = vadd(v20.qf32,v0.qf32); v21.qf32 = vadd(v21.qf32,v1.qf32) }\n"
        "    { v1:0.qf32 = vmpy(v3.hf,v9.hf); v20.qf32 = vadd(v20.qf32,v4.qf32); v21.qf32 = vadd(v21.qf32,v5.qf32) }\n"
        "    { v5:4.qf32 = vmpy(v7.hf,v9.hf); v20.qf32 = vadd(v20.qf32,v0.qf32); v21.qf32 = vadd(v21.qf32,v1.qf32) } :endloop0\n"
        " jumpr r31\n .size cost_tap_acc, .-cost_tap_acc\n");
void cost_tap_acc(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_conv_s1\n .type cost_conv_s1,@function\ncost_conv_s1:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_conv_s1,r0)\n .p2align 5\n.Lc_conv_s1:\n"
        "    { v0.qf32 = vadd(v0.sf,v8.sf) }\n"
        "    { v0.sf = v0.qf32 } :endloop0\n"
        " jumpr r31\n .size cost_conv_s1, .-cost_conv_s1\n");
void cost_conv_s1(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_conv_s2\n .type cost_conv_s2,@function\ncost_conv_s2:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_conv_s2,r0)\n .p2align 5\n.Lc_conv_s2:\n"
        "    { v0.qf32 = vadd(v0.sf,v8.sf); v1.qf32 = vadd(v1.sf,v8.sf) }\n"
        "    { v0.sf = v0.qf32; v1.sf = v1.qf32 } :endloop0\n"
        " jumpr r31\n .size cost_conv_s2, .-cost_conv_s2\n");
void cost_conv_s2(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_hf_i16_s1\n .type cost_hf_i16_s1,@function\ncost_hf_i16_s1:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_hf_i16_s1,r0)\n .p2align 5\n.Lc_hf_i16_s1:\n"
        "    { v0.h = v0.hf }\n"
        "    { v0.hf = v0.h } :endloop0\n"
        " jumpr r31\n .size cost_hf_i16_s1, .-cost_hf_i16_s1\n");
void cost_hf_i16_s1(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_hf_i16_s2\n .type cost_hf_i16_s2,@function\ncost_hf_i16_s2:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_hf_i16_s2,r0)\n .p2align 5\n.Lc_hf_i16_s2:\n"
        "    { v0.h = v0.hf; v1.h = v1.hf }\n"
        "    { v0.hf = v0.h; v1.hf = v1.h } :endloop0\n"
        " jumpr r31\n .size cost_hf_i16_s2, .-cost_hf_i16_s2\n");
void cost_hf_i16_s2(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_cmpmux_s1\n .type cost_cmpmux_s1,@function\ncost_cmpmux_s1:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_cmpmux_s1,r0)\n .p2align 5\n.Lc_cmpmux_s1:\n"
        "    { q0 = vcmp.gt(v0.h,v9.h) }\n"
        "    { v0 = vmux(q0,v0,v10) } :endloop0\n"
        " jumpr r31\n .size cost_cmpmux_s1, .-cost_cmpmux_s1\n");
void cost_cmpmux_s1(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_cmpmux_s4\n .type cost_cmpmux_s4,@function\ncost_cmpmux_s4:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_cmpmux_s4,r0)\n .p2align 5\n.Lc_cmpmux_s4:\n"
        "    { q0 = vcmp.gt(v0.h,v9.h); q1 = vcmp.gt(v1.h,v9.h); q2 = vcmp.gt(v2.h,v9.h); q3 = vcmp.gt(v3.h,v9.h) }\n"
        "    { v0 = vmux(q0,v0,v10); v1 = vmux(q1,v1,v10); v2 = vmux(q2,v2,v10); v3 = vmux(q3,v3,v10) } :endloop0\n"
        " jumpr r31\n .size cost_cmpmux_s4, .-cost_cmpmux_s4\n");
void cost_cmpmux_s4(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_load\n .type cost_load,@function\ncost_load:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_load,r0)\n .p2align 5\n.Lc_load:\n"
        "    { v0 = vmem(r1++#1) }\n"
        "    { v1 = vmem(r1++#1) } :endloop0\n"
        " jumpr r31\n .size cost_load, .-cost_load\n");
void cost_load(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_store\n .type cost_store,@function\ncost_store:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_store,r0)\n .p2align 5\n.Lc_store:\n"
        "    { vmem(r1++#1) = v8 }\n"
        "    { vmem(r1++#1) = v8 } :endloop0\n"
        " jumpr r31\n .size cost_store, .-cost_store\n");
void cost_store(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_copy\n .type cost_copy,@function\ncost_copy:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_copy,r0)\n .p2align 5\n.Lc_copy:\n"
        "    { v0 = vmem(r1++#1) }\n"
        "    { vmem(r2++#1) = v0 } :endloop0\n"
        " jumpr r31\n .size cost_copy, .-cost_copy\n");
void cost_copy(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_copy_1pk\n .type cost_copy_1pk,@function\ncost_copy_1pk:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_copy_1pk,r0)\n .p2align 5\n.Lc_copy_1pk:\n"
        "    { v0 = vmem(r1++#1); vmem(r2++#1) = v1 }\n"
        "    { v1 = vmem(r1++#1); vmem(r2++#1) = v0 } :endloop0\n"
        " jumpr r31\n .size cost_copy_1pk, .-cost_copy_1pk\n");
void cost_copy_1pk(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_load_mul\n .type cost_load_mul,@function\ncost_load_mul:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        " loop0(.Lc_load_mul,r0)\n .p2align 5\n.Lc_load_mul:\n"
        "    { v0 = vmem(r1++#1); v2.qf32 = vmpy(v1.sf,v8.sf) }\n"
        "    { v1 = vmem(r1++#1); v3.qf32 = vmpy(v0.sf,v8.sf) } :endloop0\n"
        " jumpr r31\n .size cost_load_mul, .-cost_load_mul\n");
void cost_load_mul(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_gather\n .type cost_gather,@function\ncost_gather:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        "  m0 = r4\n"
        "  r7 = #64\n"
        "  v11 = vsplat(r7)\n"
        " loop0(.Lc_gather,r0)\n .p2align 5\n.Lc_gather:\n"
        "    { vtmp.h = vgather(r2,m0,v11.h).h; vmem(r3+#0) = vtmp.new }\n"
        "    { v4 = vmem(r3+#0) } :endloop0\n"
        " jumpr r31\n .size cost_gather, .-cost_gather\n");
void cost_gather(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_gather_dep\n .type cost_gather_dep,@function\ncost_gather_dep:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        "  m0 = r4\n"
        "  r7 = #64\n"
        "  v11 = vsplat(r7)\n"
        "  r7 = #0x3e\n"
        "  v12.h = vsplat(r7)\n"
        " loop0(.Lc_gather_dep,r0)\n .p2align 5\n.Lc_gather_dep:\n"
        "    { vtmp.h = vgather(r2,m0,v11.h).h; vmem(r3+#0) = vtmp.new }\n"
        "    { v4 = vmem(r3+#0) }\n"
        "    { v11 = vand(v4,v12) } :endloop0\n"
        " jumpr r31\n .size cost_gather_dep, .-cost_gather_dep\n");
void cost_gather_dep(uint32_t n, void * a, void * b, void * c, uint32_t d);

__asm__(".text\n .p2align 5\n .globl cost_vlut16\n .type cost_vlut16,@function\ncost_vlut16:\n"
        " r5 = ##0x3f800000\n"
        "  v8 = vsplat(r5)\n"
        "  r6 = ##0x3c003c00\n"
        "  v9 = vsplat(r6)\n"
        "  v10 = vsplat(r6)\n"
        "  r7 = #0x0f\n"
        "  v12.b = vsplat(r7)\n"
        "  r5 = #0\n"
        " loop0(.Lc_vlut16,r0)\n .p2align 5\n.Lc_vlut16:\n"
        "    { v3:2.h = vlut16(v1.b,v9.h,r5) }\n"
        "    { v1 = vand(v2,v12) } :endloop0\n"
        " jumpr r31\n .size cost_vlut16, .-cost_vlut16\n");
void cost_vlut16(uint32_t n, void * a, void * b, void * c, uint32_t d);

struct cost_case {
    const char * name;
    cost_fn      fn;
    unsigned     packets;  // packets in one loop iteration
    unsigned     ops;      // HVX operations in one loop iteration
    const char * note;
};

static const struct cost_case cases[] = {
    { "qf32_mul_s1", cost_qf32_mul_s1, 2, 2, "qf32 multiply, 1 chain(s), 1 per packet" },
    { "qf32_mul_s2", cost_qf32_mul_s2, 2, 2, "qf32 multiply, 2 chain(s), 1 per packet" },
    { "qf32_mul_s4", cost_qf32_mul_s4, 4, 4, "qf32 multiply, 4 chain(s), 1 per packet" },
    { "qf32_mul_s8", cost_qf32_mul_s8, 8, 8, "qf32 multiply, 8 chain(s), 1 per packet" },
    { "qf16_mul_s1", cost_qf16_mul_s1, 2, 2, "qf16 multiply, 1 chain(s), 1 per packet" },
    { "qf16_mul_s2", cost_qf16_mul_s2, 2, 2, "qf16 multiply, 2 chain(s), 1 per packet" },
    { "qf16_mul_s4", cost_qf16_mul_s4, 4, 4, "qf16 multiply, 4 chain(s), 1 per packet" },
    { "qf32_add_s1", cost_qf32_add_s1, 2, 2, "qf32 add, 1 chain" },
    { "qf32_add_s2", cost_qf32_add_s2, 2, 4, "qf32 add, 2 chains in each packet" },
    { "qf32_add_s4", cost_qf32_add_s4, 2, 4, "qf32 add, 4 chains, 2 per packet" },
    { "qf32_add_s8", cost_qf32_add_s8, 4, 8, "qf32 add, 8 chains, 2 per packet" },
    { "i16_mul_s1", cost_i16_mul_s1, 2, 2, "int16 fractional multiply, 1 chain" },
    { "i16_mul_s2", cost_i16_mul_s2, 2, 4, "int16 multiply, 2 chains in each packet" },
    { "i16_mul_s4", cost_i16_mul_s4, 2, 4, "int16 multiply, 4 chains, 2 per packet" },
    { "i16_mul_s8", cost_i16_mul_s8, 4, 8, "int16 multiply, 8 chains, 2 per packet" },
    { "alu_s1", cost_alu_s1, 2, 2, "int16 add, 1 chain" },
    { "alu_s4", cost_alu_s4, 2, 8, "int16 add, 4 chains in each packet" },
    { "alu_s8", cost_alu_s8, 2, 8, "int16 add, 8 chains, 4 per packet" },
    { "horner_qf16_s1", cost_horner_qf16_s1, 2, 2, "qf16 multiply then add, 1 chain" },
    { "horner_qf16_s2", cost_horner_qf16_s2, 2, 4, "qf16 Horner, 2 chains" },
    { "horner_qf16_s4", cost_horner_qf16_s4, 4, 8, "qf16 Horner, 4 chains" },
    { "horner_i16_s1", cost_horner_i16_s1, 2, 2, "int16 multiply then add, 1 chain" },
    { "horner_i16_s2", cost_horner_i16_s2, 2, 4, "int16 Horner, 2 chains" },
    { "horner_i16_s4", cost_horner_i16_s4, 2, 8, "int16 Horner, 4 chains, 4 ops per packet" },
    { "whf_s1", cost_whf_s1, 2, 2, "widening f16 multiply then conversion, 1 chain" },
    { "whf_s2", cost_whf_s2, 3, 4, "widening multiply, 2 chains" },
    { "whf_s4", cost_whf_s4, 6, 8, "widening multiply, 4 chains" },
    { "tap_acc", cost_tap_acc, 4, 12, "4 widening multiplies with the 2 adds in the next packet" },
    { "conv_s1", cost_conv_s1, 2, 2, "sf to qf32 and back, 1 chain" },
    { "conv_s2", cost_conv_s2, 2, 4, "sf to qf32 and back, 2 chains" },
    { "hf_i16_s1", cost_hf_i16_s1, 2, 2, "f16 to int16 and back, 1 chain" },
    { "hf_i16_s2", cost_hf_i16_s2, 2, 4, "f16 to int16 and back, 2 chains" },
    { "cmpmux_s1", cost_cmpmux_s1, 2, 2, "compare then select, 1 chain" },
    { "cmpmux_s4", cost_cmpmux_s4, 2, 8, "compare then select, 4 chains" },
    { "load", cost_load, 2, 2, "sequential loads (r1)" },
    { "store", cost_store, 2, 2, "sequential stores (r1)" },
    { "copy", cost_copy, 2, 2, "load then store of the value" },
    { "copy_1pk", cost_copy_1pk, 2, 4, "a load and a store in each packet" },
    { "load_mul", cost_load_mul, 2, 4, "a load and a qf32 multiply of the load before" },
    { "gather", cost_gather, 2, 2, "gather of 64 int16 from the VTCM, then its load" },
    { "gather_dep", cost_gather_dep, 3, 3, "gather, load, and the next index from the value" },
    { "vlut16", cost_vlut16, 2, 2, "vlut16, then the next index from the value" },
};

int main(int argc, char ** argv) {
    const uint32_t iters = (uint32_t) lab_arg_long(argc, argv, "--iters", 2048);

    lab_init();

    // The memory cases read and write 2 vectors in each iteration. The gather table is in the VTCM.
    const size_t span = (size_t) iters * 2 * 128 + 256;
    uint8_t * vt_a  = lab_vtcm_alloc(span, 128);
    uint8_t * vt_b  = lab_vtcm_alloc(span, 128);
    uint8_t * table = lab_vtcm_alloc(4096, 128);
    uint8_t * slot  = lab_vtcm_alloc(128, 128);
    uint8_t * dd_a  = lab_ddr_alloc(span, 128);
    uint8_t * dd_b  = lab_ddr_alloc(span, 128);
    memset(vt_a, 0, span); memset(vt_b, 0, span); memset(table, 0, 4096);
    memset(dd_a, 0, span); memset(dd_b, 0, span);

    printf("lab: hvxcost %-16s %7s %5s %11s %9s  %s\n", "case", "packets", "ops", "cyc/packet", "cyc/op", "note");
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        const struct cost_case * c = &cases[i];
        for (int ddr = 0; ddr < 2; ddr++) {
            const int is_mem = !strncmp(c->name, "load", 4) || !strncmp(c->name, "store", 5) || !strncmp(c->name, "copy", 4);
            if (ddr && !is_mem) {
                break;
            }
            uint8_t * a = ddr ? dd_a : vt_a;
            uint8_t * b = ddr ? dd_b : vt_b;
            uint64_t best = UINT64_MAX;
            for (int rep = 0; rep < 3; rep++) {
                LAB_BARRIER();
                const uint64_t t0 = lab_cycles();
                c->fn(iters, a, !strncmp(c->name, "gather", 6) ? (void *) table : (void *) b, slot, 4095);
                const uint64_t t1 = lab_cycles();
                LAB_BARRIER();
                if (t1 - t0 < best) {
                    best = t1 - t0;
                }
            }
            char name[40];
            snprintf(name, sizeof(name), "%s%s", c->name, is_mem ? (ddr ? "_ddr" : "_vtcm") : "");
            printf("lab: hvxcost %-16s %7u %5u %11.2f %9.2f  %s\n", name, c->packets, c->ops,
                   (double) best / ((double) iters * c->packets), (double) best / ((double) iters * c->ops), c->note);
        }
    }
    return 0;
}
