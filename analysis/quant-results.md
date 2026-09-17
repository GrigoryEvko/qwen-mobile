# Quantization results, Qwen3.5-2B

Metric: per-token KL divergence of the model against the F16 logits of the original
model on CUDA (WikiText-2 test, 16 chunks of 512 tokens, `llama-perplexity --kl-divergence`).
Lower KL and higher top-1 agreement are better. PPL is a side print.

Plan "targeted": bulk Q4_0 (MLP, GDN qkv/out, attention q/o), head Q4_0, embedding Q8_0,
k/v projections Q8_0, recurrence controls and norms F32. Transform: norms folded, full
Hadamard R1 on the residual stream, head untied.

| # | Configuration | Bytes/token | Mean KL | 99.9 % KL | Max KL | Top-1 | PPL |
|---|---|---|---|---|---|---|---|
| floor | F16 on the phone CPU vs CUDA | 3.89 GB | 0.00020 | 0.0016 | 0.0019 | 99.09 % | 14.00 |
| floor | F16 on the phone NPU vs CUDA | 3.89 GB | 0.00019 | 0.0015 | 0.0022 | 99.12 % | 14.00 |
| floor | F16 on the phone Adreno vs CUDA | 3.89 GB | 0.00065 | 0.0078 | 0.0112 | 98.53 % | 14.02 |
| 1 | Transformed F16 (fold + R1 + untie), no quantization, CUDA | 3.89 GB | 0.00037 | 0.0025 | 0.0052 | 98.78 % | 14.01 |
| 2 | Targeted plan, round-to-nearest Q4_0 with MSE scale search (worst case) | 1.62 GiB file | 0.0836 | 1.50 | 3.21 | 84.22 % | 14.73 |
| 3 | Targeted plan, GPTQ Q4_0 (128 x 2048 C4 tokens, damp 0.01), head round-to-nearest | 1.62 GiB file | 0.0373 | 0.629 | 1.01 | 88.48 % | 14.49 |
| 4 | Qronos refit (FP-flow reference across the model) + folded column scales + MLP permutation, head solved, Q4_0, solver only | 1.62 GiB file | 0.0299 | 0.727 | 2.97 | 90.47 % | 14.08 |
| 5 | Block optimization from the RTN start (8 epochs on 128 x 2048, all layer parameters, head on the KL), folds as row 4, Q4_0 | 1.62 GiB file | 0.0412 | 0.598 | 2.08 | 88.38 % | 14.69 |

| 6 | GPTQ start + folded column scales + MLP permutation, solver only | 1.62 GiB file | 0.0325 | | | 89.85 % | |
| 7 | Block optimization, 2 epochs, latent weights frozen (scales, norms, gates, head on the KL), GPTQ start, folds as row 6 | 1.62 GiB file | 0.0295 | | | 91.35 % | |
| 8 | As row 7 with the latent weights free at 3e-6 | 1.62 GiB file | 0.0314 | | | 90.12 % | |
| 9 | As row 7 from the round-to-nearest start | 1.62 GiB file | 0.0482 | | | 87.77 % | |

| 7c | Row 7 exported and evaluated on the laptop (RTX 1000 Ada, CUDA 13.1), the control of rows 10 and 11 | 1.62 GiB file | 0.0296 | 0.590 | 7.37 | 91.40 % | 14.43 |
| 10 | Row 7 with token_embd in Q4_0 (round-to-nearest with the scale search): the price of the 4-bit lookup, laptop | 1.39 GiB file | 0.0309 | 0.788 | 8.07 | 91.18 % | 14.47 |
| 11 | Tied head: token_embd is the head, Q4_0 by GPTQ on E' with the moments of M·norm(h), then the KL head optimization (frozen latents, 300 steps), layer packs of row 7, output_rot in F16, laptop | 1.13 GiB file | 0.0311 | 0.499 | 6.82 | 90.42 % | 14.46 |

Rows 7c, 10 and 11 ran on the laptop against the F16 base of the box, with llama.cpp c6824a9 plus
`patches/llama-tied-head`. The control reproduces row 7 (0.0296 / 91.40 % against 0.0295 / 91.35 %).
The motivation of the tie is RAM and file size, not the bytes of a decode token: the logits still
read the full Q4_0 head (0.286 GB) tied or not. The tie removes the separate Q8_0 `token_embd`
tensor (540 MB) from the file and from the RAM of the phone, and adds the dense map M in F16
(8.4 MB), one 2048 × 2048 matvec per token (+0.8 % of the MACs of the head). The file goes from
1,743,744,000 to 1,211,788,288 bytes.

The 4-bit lookup alone (row 10) costs 0.0013 of mean KL and 0.2 points of top-1. The tie (row 11)
costs 0.0003 more KL and 0.8 points of top-1 against row 10, and its 99.9 % KL is lower. A tied head
has no column scales, because its rows are the rows of the lookup, thus the head solve works on the
moments of M·norm(h) with no fold. The KL head optimization on the calibration tokens gives
0.02348 → 0.02145 for the tied head and 0.02189 → 0.01890 for row 7. The head-only path of
`quantize` reuses the layer packs of row 7: the residual stream and all the layer packs are those of
row 7, only the head and the lookup change. Its calibration on the laptop took 1364 s for the pass
of both copies through the layers, 675 s for the second pass of the working copy, 30 s per head
solve and about 6 min for the head optimization, with the copies on the CPU and one layer at a
time on the GPU.

The tie is exact. The tied F16 file of the converter (`transform --tie-head`, `convert --source t`)
and the tied F16 file of `export --tie-head` (from the untied F16 with the map of the tied
checkpoint) against the untied F16 file, on the CPU build, WikiText-2 16 × 512: mean KL −0.000004
and −0.000003 (the resolution of the 8-bit logit base), 99.9 % KL 0.000048, max 0.000052, same
top-1 99.975 % and 100.000 %. The HF model with the M wrapper against the original checkpoint on a
prompt (`verify`): max abs logit diff 1.8e-5, mean 1.9e-6, top-1 100 %. The local export of row 7
from the untied F16 is byte-identical to the phone file in 329 of 336 tensors, and the 7 `ssm_a`
tensors differ by 1 to 2 bytes of an `exp` in the converter.

Record for the 4B (d = 2560, no run): M is 13.1 MB in F16 (a Kronecker rotation, the same
identity), the Q8_0 `token_embd` is 675 MB, the Q4_0 head 358 MB, the matvec of M adds 1.0 % to
the MACs of the head. The commands of rows 10 and 11, from the project root with the `.venv`:

    python -m quant.run transform --model Qwen3.5-2B --device cpu --tie-head          # weights/Qwen3.5-2B-t, tied, with output_rot
    python -m quant.run transform --model Qwen3.5-2B --device cpu --suffix tu         # the untied reference of the exactness test
    python -m quant.run convert --model Qwen3.5-2B --source t
    python -m quant.run convert --model Qwen3.5-2B --source tu
    python -m quant.run quantize --model Qwen3.5-2B --device cuda --stream --head-only --packs quant-out/Qwen3.5-2B-bo-gptq-frozen-e2 \
        --embedding Q4_0 --init gptq --method blockopt --epochs 2 --freeze-weights --head-steps 300 --head-chunk 8192
    cp -al quant-out/Qwen3.5-2B quant-out/Qwen3.5-2B-tied-bo-gptq-frozen-e2
    python -m quant.run export --model Qwen3.5-2B --source tu --packs quant-out/Qwen3.5-2B-tied-bo-gptq-frozen-e2 --embedding Q4_0 --tie-head --tag tied-bo-gptq-frozen-e2
    python -m quant.run export --model Qwen3.5-2B --source tu --packs quant-out/Qwen3.5-2B-bo-gptq-frozen-e2 --embedding Q4_0 --tag row7-embd-q4
    python -m quant.run export --model Qwen3.5-2B --source tu --packs quant-out/Qwen3.5-2B-bo-gptq-frozen-e2 --tag local-row7
    python -m quant.run eval --model Qwen3.5-2B --gguf weights/gguf/Qwen3.5-2B-<tag>.gguf
    python -m quant.run export --model Qwen3.5-2B --source tu --packs quant-out/empty --only '^$' --tie-head --rot quant-out/Qwen3.5-2B-t-rot.npy --tag tu-F16-tied
    llama.cpp/build-host/bin/llama-perplexity -m weights/gguf/Qwen3.5-2B-tu-F16.gguf -f data/wiki.test.raw -c 512 --chunks 16 --kl-divergence-base eval/tu.kld
    llama.cpp/build-host/bin/llama-perplexity -m weights/gguf/Qwen3.5-2B-t-F16.gguf -f data/wiki.test.raw -c 512 --chunks 16 --kl-divergence-base eval/tu.kld --kl-divergence

Row 5 over-fits: the per-layer training loss halves, the held-out error grows at every layer (drift
KL 0.0421 against 0.0291 for row 3). The diagnostics say: the rounding must come from a Hessian
solver (row 9 against row 7), the latent weights must stay frozen on 128 x 2048 tokens (row 8
against row 7), and the block optimization then refines the scales, the norms and the head for a
gain of 10 % KL and 1.5 points of top-1 over its start (row 7 against row 6). Row 7 is the file on
the phone as of 2026-09-17 08:05 (`Qwen3.5-2B-gptq-Q4_0.gguf`). The drift KLs of rows 6 to 9:
0.0329, 0.0303, 0.0320, 0.0515.

Phone speed of row 3 (the GPTQ Q4_0 file, llama-bench, thermal status 0, 2026-09-17, op fusion off on
the NPU): GPU OpenCL pp512 574 t/s, tg64 30.1 t/s. NPU HTP0 pp512 968 t/s, tg64 31.9 t/s. The F16
file gives tg 13.6 (GPU) and 13.8 (NPU), thus Q4_0 more than doubles the decode speed.

Phone speed on the NPU with op fusion on (llama-bench -p 512 -n 32, 2026-09-17 23:58 and 2026-09-18
00:45): row 7 pp512 1017 t/s, tg32 32.05 t/s; row 11 (tied head, 0.53 GB less) pp512 1017 t/s, tg32
32.2 t/s; the 2B Q8_0 pp512 1004 t/s, tg32 21.5 t/s (2.1 GB per token, 45 GB/s); the 4B Q8_0 pp512
377 t/s, tg32 7.0 t/s (4.7 GB per token, the gate and up pairs stream at 56 GB/s, thus the floor). The
fused matvec add of the Hexagon backend (patches/hexagon-fusion/0001) makes the fusion correct: the
32-token greedy streams with fusion on and off are identical for the Q4_0 and the F16 file.

Grid test on Gaussian weights (256 x 512, block-32 F16 scales with the scale search, 4.5 bits per
weight, `python -m quant.trellis 12`): Q4_0 21.8 dB, IQ4_NL 22.5 dB, codebook per matrix 22.3 dB,
trellis (bit-shift, L = 12) 22.1 dB with a Gaussian table and 21.8 dB with the Ungerboeck cosets.
IQ4_NL is the best 4-bit grid at equal bytes on this test: the scale of a block carries the sign of
its maximum, thus the maximum lands on −127 and the asymmetric table (−127 … 113) spends its levels
where the values are. The numbers before 2026-09-17 came from a wrong positive half of the IQ4_NL
table (6 … 127 in place of 1 … 113) and a positive scale, which clipped every positive maximum.

The HF lockstep drift report (analysis/*.drift.md, 16 x 1024 WikiText tokens, fp32 on CUDA) gives
the same ranking with its own numbers: row 3 mean KL 0.0291, row 4 0.0314. The KL attribution
of row 3 is in analysis/quant-attribution.md: the round-to-nearest head alone is 0.011 of the
0.037, MLP gate/up 0.009, MLP down 0.007, GDN out 0.005, GDN qkv 0.0025, layers 0-2 0.0075.

## Q8_0 files of the original checkpoints, 2B and 4B (2026-09-18)

The packer of the pipeline (`q8_0_quantize` and `pack_q8_0` in `quant/grid.py`) wrote these files
from the F16 GGUF of the original checkpoint: no rotation, no folds, no calibration. Each 2-D weight
of the 24 (32) decoder layers and `token_embd` is round-to-nearest Q8_0 (the head is tied, thus the
lookup is the head). The norms, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, `ssm_alpha` and `ssm_beta`
are F32, as the plan of the export does today. The MTP block (`blk.24` of the 2B, `blk.32` of the
4B: 8 F16 tensors and 7 F32 norms) stays F16, because llama.cpp does not load it ("unused tensor").
The metadata is that of the F16 GGUF, with the file type Q8_0. The projectors are copies, byte for
byte, of the original F16 projectors, next to the files as `<name>.mmproj.gguf`.

| Model | File | Bytes | Mean KL | 99.9 % KL | 99.0 % KL | Max KL | Top-1 | PPL (Q / F16) |
|---|---|---|---|---|---|---|---|---|
| 2B | Qwen3.5-2B-Q8_0.gguf | 2,137,156,512 (1.99 GiB, 151 Q8_0 tensors) | 0.00115 | 0.0143 | 0.0057 | 0.0281 | 98.21 % | 14.00 / 14.00 |
| 4B | Qwen3.5-4B-Q8_0.gguf | 4,735,180,608 (4.41 GiB, 201 Q8_0 tensors) | 0.00202 | 0.175 | 0.0085 | 1.56 | 97.21 % | 10.78 / 10.77 |

The 2B ran against the F16 base of the box on the laptop CUDA build, with all layers on the GPU, as
rows 7c to 11. The F16 4B does not fit in the 6 GB of the laptop GPU, thus its base comes from
`llama-perplexity` with 15 decoder layers and the output on the GPU and 17 decoder layers on the
CPU (`-ngl 16`). The Q8_0 4B ran against that base with the same 16 layers on the GPU. The location
of the layers has no effect on this laptop. The F16 4B with all layers on the CPU gives a mean KL of
−0.000014 against that base (the resolution of the base), a maximum of 0.000002 and a top-1 of
100.000 %. The Q8_0 4B gives the same statistics with all layers on the GPU (`-ngl 99`), with 16
layers on the GPU and with all layers on the CPU (`-ngl 0`).

The 4B has a heavy tail. Through chunk 13 the mean KL is 0.0014 to 0.0018, and chunks 14 and 16
hold the tokens of the tail: the 99.9 % point is 0.17, thus eight tokens have a KL of more than
0.17, and the maximum is 1.56. The 2B has no such tail (maximum 0.028, 99.9 % point 0.014). The next
step isolates the class that causes the tail, with `--only` on the 4B export: `token_embd`, the MLP,
the GDN projections and the attention projections, one file each.

The checks: `llama-completion` of build-host (temperature 0, 48 tokens) on each of the two files
gives correct text about the laws of thermodynamics, and `llama-mtmd-cli` of build-host with the 2B
file and its projector gives one correct sentence about a 1000 × 1000 JPEG. The build-host
configuration has `LLAMA_BUILD_MTMD=ON` for `llama-mtmd-cli` (set for this check). The commands,
from the project root:

    .venv/bin/python -m quant.run export --model Qwen3.5-2B --device cpu --f16 weights/gguf/Qwen3.5-2B-F16.gguf --packs quant-out/empty --bulk Q8_0 --gdn-gate Q8_0 --tag Q8_0
    .venv/bin/python -m quant.run export --model Qwen3.5-4B --device cpu --f16 weights/gguf/Qwen3.5-4B-F16.gguf --packs quant-out/empty --bulk Q8_0 --gdn-gate Q8_0 --tag Q8_0
    cp weights/gguf/Qwen3.5-2B-mmproj-F16.gguf weights/gguf/Qwen3.5-2B-Q8_0.mmproj.gguf
    cp weights/gguf/Qwen3.5-4B-mmproj-F16.gguf weights/gguf/Qwen3.5-4B-Q8_0.mmproj.gguf
    .venv/bin/python -m quant.run eval --model Qwen3.5-2B --gguf weights/gguf/Qwen3.5-2B-Q8_0.gguf
    llama.cpp/build-cuda/bin/llama-perplexity -m weights/gguf/Qwen3.5-4B-F16.gguf -ngl 16 -f data/wiki.test.raw -c 512 --chunks 16 --kl-divergence-base eval/Qwen3.5-4B-F16.wiki.c512x16.kld
    .venv/bin/python -m quant.run eval --model Qwen3.5-4B --gguf weights/gguf/Qwen3.5-4B-Q8_0.gguf --ngl 16
    llama.cpp/build-cuda/bin/llama-perplexity -m weights/gguf/Qwen3.5-4B-F16.gguf -ngl 0 -t 12 -f data/wiki.test.raw -c 512 --chunks 16 --kl-divergence-base eval/Qwen3.5-4B-F16.wiki.c512x16.kld --kl-divergence
    llama.cpp/build-host/bin/llama-completion -m weights/gguf/Qwen3.5-2B-Q8_0.gguf -p "The three laws of thermodynamics are" -n 48 -no-cnv --temp 0 --simple-io
    llama.cpp/build-host/bin/llama-mtmd-cli -m weights/gguf/Qwen3.5-2B-Q8_0.gguf --mmproj weights/gguf/Qwen3.5-2B-Q8_0.mmproj.gguf --image <file.jpg> -p "Describe this image in one sentence." -n 64 --temp 0
