# The Qwen3.5 vision encoder on the Adreno 830 OpenCL backend

Date: 2026-09-17. llama.cpp commit c6824a9. Static analysis and host builds only.
The test image is 681 x 1024 pixels. The preprocessor gives 672 x 1024 pixels, 42 x 64 = 2688 patches, 672 tokens.

## 1. Summary

The OpenCL backend accepts every op of the vision graph. No op goes to the CPU.
The 7 s are the time of the GPU kernels, plus the program builds of the first image.
The encoder does 2.4 TFLOP for 672 tokens, the same quantity as the LLM prefill of 700 tokens.
A prefill of 700 tokens in less than 1 s agrees with the Hexagon NPU context (`ctx_pf`, 1456 t/s). The same GPU does 700 LLM tokens in approximately 2.5 s (pp512 = 275 t/s).
Thus the GPU encoder is slow because of its kernels and its attention, not because of a fallback.

## 2. The graph and the backend decision for each op

The host run with `GGML_SCHED_DEBUG=2` gives 736 nodes and 1 split. The table gives the decision of `ggml_opencl_supports_op` (`ggml/src/ggml-opencl/ggml-opencl.cpp:8582`) for each op class.

| Node | Types and shapes (ne) | Decision | Kernel and reason |
|---|---|---|---|
| IM2COL x2 (nodes 0, 7) | src1 F32 [672, 1024, 3], dst F16 [768, 42, 64] | GPU | `kernel_im2col_f16`, `supports_op` returns true for all IM2COL (line 8910) |
| MUL_MAT x2 (patch, nodes 3, 10) | src0 F16 [768, 2688] (im2col), src1 F16 [768, 1024] (weight view), dst F32 [2688, 1024] | GPU, slow | src0 F16 is always accepted (line 8796). src1 is F16, thus no GEMM path applies and `kernel_mul_mat_f16_f16` (a GEMV loop, line 23336) does 1024 columns against 2688 rows |
| CONT x2 (nodes 6, 13) | F32 [42, 64, 1024, 1], contiguous copy | GPU | `kernel_cpy_f32_f32` |
| ADD (node 14) | F32 [42, 64, 1024, 1] + same | GPU | `kernel_add` |
| CONT (node 16) | permute(1, 2, 0, 3), nb0 = 10752 bytes, transposed | GPU | `kernel_cpy_f32_f32`, one strided read per element |
| CONT (node 19) | permute(0, 2, 1, 3) of [2048, 21, 2, 32] | GPU | `kernel_cpy_f32_f32` |
| ADD patch bias | [1024, 2688] + [1024] | GPU | `kernel_add_row` |
| UPSCALE (node 23) | src F32 permuted view of [1024, 48, 48], dst [42, 64, 1024], bilinear + align corners | GPU | `kernel_upscale_bilinear`, the flag ALIGN_CORNERS sets `pixel_offset` = 0 (line 15972), no antialias flag |
| CONT x3 (nodes 25, 26, 29) | one transposed copy, two contiguous copies | GPU | `kernel_cpy_f32_f32` |
| NORM + MUL + ADD (x49) | F32 [1024, 2688], weight and bias F32 [1024] | GPU, one kernel | `ggml_opencl_can_fuse` accepts the pattern (line 7930), `ggml_opencl_op_norm_fused` |
| MUL_MAT qkv, o, up, down (x96) | src0 F16 weight [1024, 3072], [1024, 1024], [1024, 4096], [4096, 1024], src1 F32 [K, 2688] | GPU | `ggml_cl_can_use_adreno_xmem_gemm_f16_f32` (line 18065) accepts K % 8 = 0, M >= 64, N = 2688 <= image width, thus `kernel_gemm_xmem_f16_f32_os8`. The kernel adds the products in F16 (`gemm_xmem_f16_f32_os8.cl:134`) |
| ADD bias (x96) | [M, 2688] + [M] | GPU | `kernel_add_row` |
| ROPE x48 | src0 F32 view [64, 16, 2688] with nb = (4, 256, 12288), mode VISION, positions I32 [10752] | GPU | `kernel_rope_vision_f32`, the kernel reads nb00..nb03 (line 27494) |
| CPY x48 (K and V to F16) | src F32 permuted [64, 2688, 16], dst F16 contiguous | GPU | `kernel_cpy_f32_f16` |
| FLASH_ATTN_EXT x24 | q F32 permuted [64, 2688, 16], k and v F16 [64, 2688, 16], no mask, dst F32 | GPU | dk = dv = 64 is in the table (line 8943), `is_f32_f16`, the A7X and E17 refusals do not apply to the Adreno 830 (A8X, compiler E031.47). Dispatch: `fa.f32_f16_split`, bm = 64, bn = 32, n_split = 2 (`fa_tune.h:17`), no KV pad because 2688 % 32 = 0 |
| ADD residual (x48) | [1024, 2688] + [1024, 2688] | GPU | `kernel_add` |
| GELU (x25) | F32 [4096, 2688] contiguous | GPU | `kernel_gelu_4` |
| MUL_MAT mm.0, mm.2 | F16 [4096, 4096] and [4096, 2048], src1 F32 [4096, 672] | GPU | xmem GEMM |

The copies between the host and the device are `inp_raw` (8.3 MB) and `positions` (43 KB) to the device, and the output (5.5 MB) to the host. There is no split boundary.

## 3. Where the 7 s go

Work of one encode of 2688 patches (script `vit_cost_model.py` in the session scratchpad):

| Op class | Work | Kernel | Estimate |
|---|---|---|---|
| Layer GEMMs (96) | 1624 GFLOP | xmem GEMM, F16 products | 1.4 to 2.0 s at 0.8 to 1.2 TFLOPS |
| Attention (24) | 710 GFLOP, 4 x 2688^2 x 64 x 16 per layer | fused FA tile, 128 work items per 64 queries, LDS-bound (comment at `flash_attn_f32_f16.cl:222`) | 1.4 to 3.5 s at 0.2 to 0.5 TFLOPS |
| Patch GEMV (2) | 8.5 GFLOP, but 10.6 GB of reads | `kernel_mul_mat_f16_f16` | 0.25 to 0.35 s |
| Elementwise (approximately 430 nodes) | 11.4 GB of F32 traffic | add, rope, cpy, gelu, fused norm | 0.3 to 0.4 s |
| Merger (2 GEMMs) | 34 GFLOP | xmem GEMM | 0.03 s |
| Kernel launches | approximately 600 | | 0.02 to 0.03 s |
| Sum | | | 3.4 to 6.4 s |

The first image after the app start is slower. `ggml_cl_flash_attn` builds the FA programs on first use (`ggml_opencl_ensure_fa_pre_kernels`, `ggml_opencl_ensure_fa_variant`, lines 5317 and 5426). For dk = 64 that is three program builds, two of them from the 2656-line source with all decode kernels included. The app sets `mp.warmup = false` (`llama_jni.cpp:341`), thus the first image also does the reserve.
Measure the second image, not the first. The app measures `mtmd_helper_eval_chunks`, which includes the LLM decode of the 672 image tokens on the prefill device.

The F32 activations cost 11 GB of traffic, approximately 0.3 s. They are not the primary cause. The attention is quadratic: 2688 patches give 4 times the attention work of the LLM prefill for the same token count, and the FA tile kernel is 2 to 5 times slower per FLOP than the xmem GEMM.

## 4. Patches

The patches are unified diffs against commit c6824a9, in `patches/opencl-vision/`. On the host CPU, the patched build gives bit-identical embeddings for the test image (maximum difference 0.0 over 672 x 2048 values).

- `0001-mtmd-qwen-patch-embedding-as-gemm.patch` (`tools/mtmd/models/qwen2vl.cpp`, `qwen3vl.cpp`). The patch embedding becomes `ggml_im2col` to F32 plus `ggml_mul_mat(weight, columns)`. The weight is `src0` and the columns are `src1`, thus the xmem GEMM does it, not the F16 x F16 GEMV. The result is already in the `[c, w, h]` order, thus the patch removes the transposed copy and two plain copies. The two temporal kernels use one im2col. The node count changes from 736 to 728. The estimate of the gain is 0.25 to 0.4 s.
- `0002-mtmd-clip-compute-warmup.patch` (`tools/mtmd/clip.cpp`). With `CLIP_WARMUP_COMPUTE=1`, `clip_init` encodes one 96 x 96 image after the load. The OpenCL backend then builds the FA programs at the load, not during the first image of the user. The app can do the same without a patch: encode a small bitmap after `mtmd_init_from_file`.
- `0003-mtmd-qwen3vl-zero-mask-test-aid.patch` (`qwen3vl.cpp`, `clip.cpp`). With `MTMD_VIT_ZERO_MASK=1`, the attention gets an all-zero mask. This is a test aid for section 6. The result does not change.

Options that need no patch, for a test on the phone:

- `GGML_OPENCL_FA_TUNE=64:64:64:64:2:64` changes bn from 32 to 64 for dk = 64 (`fa_tune.h:44`). Other values to try: `64:64:64:32:4:64` and `64:64:32:32:1:0`.
- `GGML_OPENCL_XMEM_SDPA=1` selects the image-based attention path (`ggml_cl_adreno_xmem_attn_can_use`, line 16738). The ViT shapes pass its checks. The path stores the full score matrix, 231 MB per buffer for 2688 tokens, thus it uses approximately 0.5 GB of GPU memory.
- `GGML_OPENCL_ADRENO_XMEM_GEMM=0` selects the local-memory GEMM with F32 sums. It is slower. Use it for an accuracy comparison of the image features.
- `GGML_OPENCL_PROFILING` (the `-Pprofiling=true` build of the app) gives the time of each kernel with the tensor name. Use it to replace the estimates of section 3 with measured values.

## 5. Fewer tokens

The Qwen preprocessor keeps the aspect ratio and aligns to 32 pixels (`mtmd-image.cpp:122`). `image_max_tokens` sets `image_max_pixels` = tokens x 1024 (`clip-model.h:190`). The app uses 1024 (`llama_jni.cpp:52`). For the test image:

| `image_max_tokens` | Pixels | Patches | Tokens | GEMM work | Attention work | Estimate of the time |
|---|---|---|---|---|---|---|
| 1024 (app) | 672 x 1024 | 2688 | 672 | 1.0 | 1.0 | 3.4 to 6.4 s |
| 576 | 608 x 928 | 2204 | 551 | 0.82 | 0.67 | 2.5 to 4.7 s |
| 400 | 512 x 768 | 1536 | 384 | 0.57 | 0.33 | 1.6 to 2.8 s |
| 256 | 416 x 608 | 988 | 247 | 0.37 | 0.14 | 0.9 to 1.5 s |

The resize is bicubic, thus a small image keeps its large structures. The model cannot read text and small objects when one token holds more than approximately 32 x 32 source pixels. The warning at `clip.cpp:1664` gives 1024 tokens as the minimum for grounding tasks. A cap of 400 tokens gives a 2 to 2.5 times faster encode for chat. Keep 1024 for OCR and grounding.

## 6. The Hexagon backend

The earlier bisect did not exclude the flash attention. `ggml_backend_hexagon_device_supports_op` uses `std::regex_match` (`ggml-hexagon.cpp:6644`), which accepts only a full match of the op name. The op name is `FLASH_ATTN_EXT` (`ggml.c:1070`), thus the regex `FLASH_ATTN|SOFT_MAX` matched nothing. The same applies to `CONV` and `CONV_2D`.

The ViT attention differs from the LLM attention on the NPU in three ways. It has no mask, the LLM always has one. It has 16 KV heads for 16 heads (G = 1), the LLM has G = 4. It has dk = dv = 64, the LLM has 256. `ggml_hexagon_flash_attn_is_hmx_eligible` (line 3980) sends dk = 64 to the HMX FA kernel. The LLM graphs did not put these three conditions on the NPU before.

Test sequence on the phone, one step at a time:

1. `GGML_HEXAGON_OPFILTER='FLASH_ATTN_EXT'`. If the features become correct, the op is the flash attention.
2. `GGML_HEXAGON_FA_SELECT=1` selects the HVX FA kernel instead of the HMX kernel (`opt_fa_select`, line 3991).
3. Patch 0003 with `MTMD_VIT_ZERO_MASK=1` gives the FA a mask. If this makes the features correct, the HTP FA kernel does not accept a NULL mask for these shapes.
4. `GGML_HEXAGON_OPFILTER='MUL_MAT'` moves all matmuls to the CPU. Patch 0001 already removes the only F16 x F16 matmul, which the HVX kernel `HTP_MM_KERNEL_HVX_F16_F16` did (`ggml_hexagon_matmul_is_hmx_eligible` accepts only src1 F32, line 4247).

The other ops are safe by inspection. UPSCALE has no Hexagon kernel and goes to the CPU. The CPY and CONT kernels read the strides (`htp/cpy-ops.c:220`, `:156`). The binary kernels have a row broadcast (`htp/binary-ops.c:384`). The rope kernel reads nb02 (`htp/rope-ops.c:598`).

If the flash attention is the op, an exclusion is not a solution: 710 GFLOP on the CPU takes more than 3 s. The fix goes into the HTP FA kernel. With that fix the NPU can encode the image in approximately 1 s, because its HMX GEMM does the LLM prefill at 1456 t/s.
