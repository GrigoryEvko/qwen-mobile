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
| 12 | Row 11 plus the MTP block in the rotated basis (F16, two F16 maps of 8.4 MB), the draft of `--spec-type draft-mtp`, laptop | 1.14 GiB file | 0.0311 | 0.499 | 6.82 | 90.42 % | 14.46 |
| 13 | Row 12 with the eight matrices of the MTP block in round-to-nearest Q8_0 | 1.09 GiB file | as row 12 | | | | |

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

## The MTP block of the rotated files (2026-09-18)

The multi-token prediction (MTP) block of Qwen3.5 (`blk.24` of the 2B) drafts the next token
for the speculative mode `--spec-type draft-mtp` of llama.cpp. The block reads the final-norm
output x = γ_f ⊙ norm(h) of the main model through `hnorm`, and the embedding of the last token
through `enorm`. The transformed main model supplies y = norm(h') = Qᵀ·norm(h) in place of x, and
rms(x) is not rms(y). Thus the block of the rows before row 12 reads the wrong vector. The transform
(`quant/transform.py`, on by default when the checkpoint has the block) puts the block into the
rotated basis with two dense maps of d × d in F16, 8.4 MB each for the 2B:

- `blk.24.nextn.hnorm_rot` = diag(γ_f)·Q before `hnorm`, which makes x from y.
- `blk.24.nextn.shared_head_rot` = Qᵀ·diag(γ_f)⁻¹·diag(γ_s)·Q after the head norm of the block,
  which becomes the identity. The map puts the normed state of the block into the convention of the
  main model. Thus the tied head reads it through `output_rot`, as the main graph does. And the next
  draft step reads it through `hnorm_rot` again, because llama.cpp chains the draft steps on the
  `h_nextn` of the block.

The embedding half of `fc` folds `enorm` and Q, `fc` writes into the rotated stream, and the layer of
the block takes the transform of a main layer. The head map is the same for the tied and the untied
file. The calibration of row 11 moved the output norm by 0.6 %. Thus the export scales the columns
of `hnorm_rot` and the rows of `shared_head_rot` by the exported norm. The patch
`patches/llama-tied-head/0002-qwen35-mtp-rotation.patch` (on top of 0001) maps the two tensors in the
converter, loads them as optional, and applies them in the MTP graph. The main graph does not load
the block, thus rows 12 and 13 have the KL of row 11. The 321 trunk tensors of row 12 are byte for
byte those of row 11. The KL run gives 0.031126 / 0.4989 / 6.82 / 90.417 % (row 11: 0.0311 / 0.499 /
6.82 / 90.42 %).

The exactness test is the draft acceptance of `llama-speculative-simple` on the CPU build, one
prompt, temperature 0, 256 tokens, 12 threads, draft 3 (the preset). The original F16 file and the
transformed tied F16 file (`weights/Qwen3.5-2B-tm`) give identical counts and identical text. The
tied F16 file of row 11 (`-t`, the block untouched) is the control: the target accepts no draft
token, and the speculation makes the decode slower than no speculation. The checkpoint `-tm` has the
trunk of `-t` byte for byte (618 tensors), only the block and the two maps are different.

| File | Bytes | Decode, no draft, t/s | Decode, draft-mtp, t/s | Drafted | Accepted | Acceptance |
|---|---|---|---|---|---|---|
| Original F16 (`Qwen3.5-2B-F16.gguf`) | 3,897,387,936 | 9.1, 8.0 | 11.3, 14.6 | 278 | 164 | 58.99 % |
| Transformed tied F16 with the MTP block (`Qwen3.5-2B-tm-F16.gguf`) | 3,922,553,984 | | 11.2, 8.9 | 278 | 164 | 58.99 % |
| Transformed tied F16, the block untouched (`Qwen3.5-2B-t-F16.gguf`), the control | 3,905,776,640 | | 3.9 | 768 | 0 | 0.00 % |
| Row 12: row 11 plus the block in F16 | 1,228,565,632 | 16.2, 23.6 | 22.5, 22.7 | 281 | 166 | 59.08 % |
| Row 13: row 11 plus the block in Q8_0 | 1,171,549,312 | | 21.0, 24.1 | 282 | 165 | 58.51 % |

The counts are the same in the two passes of each file. The speed has two values, one per pass,
because another tenant used the CPU of the laptop during the runs (load average 9 to 12 on 28
threads). Thus the speed is an indication only. The rows 12 and 13 give the same text, thus the Q8_0
block drafts the same tokens on this prompt with 57 MB less. The block adds no cost to the main
model: llama.cpp loads it only with `--spec-type draft-mtp`. The 4B has the same block structure
(`blk.32`, the shared embedding, the tied head), thus the same commands apply to it. The commands,
from the project root with the `.venv`:

    python -m quant.run transform --model Qwen3.5-2B --device cpu --tie-head --suffix tm
    python -m quant.run convert --model Qwen3.5-2B --source tm
    python -m quant.run export --model Qwen3.5-2B --device cpu --source tm --packs quant-out/Qwen3.5-2B-tied-bo-gptq-frozen-e2 --embedding Q4_0 --tie-head --tag tied-bo-gptq-frozen-e2-mtp
    python -m quant.run export --model Qwen3.5-2B --device cpu --source tm --packs quant-out/Qwen3.5-2B-tied-bo-gptq-frozen-e2 --embedding Q4_0 --tie-head --mtp Q8_0 --tag tied-bo-gptq-frozen-e2-mtp-q8
    python -m quant.run eval --model Qwen3.5-2B --gguf weights/gguf/Qwen3.5-2B-tied-bo-gptq-frozen-e2-mtp.gguf
    llama.cpp/build-host/bin/llama-speculative-simple -m weights/gguf/<file>.gguf --spec-type draft-mtp -p "<prompt>" -n 256 --temp 0 -t 12 -c 2048
    llama.cpp/build-host/bin/llama-completion -m weights/gguf/<file>.gguf -p "<prompt>" -n 256 --temp 0 -t 12 -c 2048 -no-cnv --simple-io

The prompt: "Explain how the TCP three-way handshake works, why it is needed, and what happens when
a packet of the handshake is lost."

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
0.17, and the maximum is 1.56. The 2B has no such tail (maximum 0.028, 99.9 % point 0.014). The
section "The class that makes the tail of the 4B Q8_0 file" isolates the class.

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

Decode path on the NPU with the fused recurrent state step (patches/hexagon-fusion/0002 to 0004, 2026-09-18):
the one-token decode of the Q4_0 file (llama-perplexity -b 1 -ub 1, 2 chunks, against the CUDA F16
base) gives mean ln(PPL(Q)/PPL(base)) 0.0380 ± 0.0216 with the fusion on and 0.0405 ± 0.0217 with it
off, same top-1 (90.98 % against 90.78 %), maximum KLD 7.35 against 7.49: the fused path is at the
floor of the unfused path. The decode batch on the DSP falls from 31.0 ms to 24.8 ms per token (513
ops instead of 790). The 512-token prefill profile: GATED_DELTA_NET takes 235 ms of the 501 ms batch
(44 %), the matmuls 152 ms, SSM_CONV plus CONCAT 54 ms; the prefill floor is about 160 ms.

MTP on the phone (NPU, llama-server --spec-type draft-mtp, draft 3, one prompt, 96 tokens, temperature
0, 2026-09-18): row 12 (tied Q4_0 with the rotated MTP block) 20.0 t/s with 51 of 131 drafts accepted
(39 %), against 33.1 t/s plain; the untransformed 2B Q8_0 17.3 t/s with 49 of 135 accepted, against
21.1 plain; the 4B Q8_0 18.6 t/s with 34 of 38 accepted (89 %), against 7 to 9.6 plain. The rotated
block drafts as the original (the acceptance matches), but the 2B head is weak, and on the HVX matvec
a 4-row verify step costs 1.9x a single row, thus MTP pays only where the acceptance is high (the 4B)
or once the multi-row matvec streams the weights one time for 1 to 4 rows.

MTP on the 4B Q8_0 (NPU, llama-server, 2026-09-18, the fused state step on): a free-form answer of
160 tokens gives plain 10.4 t/s, draft 3 11.3 t/s (89 of 210 drafts accepted, 42 %), draft 5 8.1 t/s
(27 %); the morning number of 18.6 t/s with 89 % accepted was a formulaic thinking block. A draft-3
step costs 2.1 plain tokens (three draft passes through the 0.67 GB head, a four-row verify at 1.9x a
single row, four host round trips) and yields 2.27 tokens. The gain needs the multi-row matvec and the
host gap fix; the draft length must follow the recent acceptance.

## The class that makes the tail of the 4B Q8_0 file (2026-09-18)

The tail is the round-to-nearest Q8_0 of `ssm_out`, the output projection of the GDN block. With
`ssm_out` in F16 and every other 2-D weight in Q8_0, the maximum KL falls from 1.56 to 0.553 and the
99.9 % point falls from 0.175 to 0.0369, for 236 MB more. No other class comes near that trade.

Each "only" row comes from one `--only` export on the F16 GGUF of the original 4B: the class takes
round-to-nearest Q8_0 and every other 2-D weight stays F16. The norms, `ssm_a`, `ssm_dt.bias`,
`ssm_conv1d`, `ssm_alpha` and `ssm_beta` stay F32, as they do in the full file, thus the row holds
the error of that class alone. Each "all but" row uses `--invert`: the class stays F16 and all the
rest takes Q8_0. All the rows ran against the F16 base of the 4B with 16 layers on the laptop GPU.

| Variant | Q8_0 tensors | Bytes | Mean KL | 99.9 % KL | 99.0 % KL | Max KL | Top-1 | PPL (Q) |
|---|---|---|---|---|---|---|---|---|
| only `token_embd` | 1 | 8,077,516,608 | 0.000801 | 0.0131 | 0.00306 | 0.455 | 98.04 % | 10.770 |
| only the MLP (gate, up, down) | 96 | 6,550,118,208 | 0.001228 | 0.0217 | 0.00538 | 0.726 | 97.72 % | 10.779 |
| only the attention (q, k, v, o) | 32 | 8,398,233,408 | 0.000784 | 0.00922 | 0.00297 | 0.506 | 98.31 % | 10.778 |
| only the GDN qkv and gate | 48 | 7,965,695,808 | 0.000703 | 0.0124 | 0.00352 | 0.193 | 98.33 % | 10.775 |
| only `ssm_out` | 24 | 8,437,555,008 | 0.001084 | 0.0442 | 0.00499 | 0.517 | 98.16 % | 10.754 |
| only `token_embd` and the MLP | 97 | 5,954,150,208 | 0.001008 | 0.0183 | 0.00493 | 0.112 | 97.79 % | 10.775 |
| all but `ssm_out` (F16 `ssm_out`) | 177 | 4,971,110,208 | 0.001392 | 0.0369 | 0.00593 | 0.553 | 97.77 % | 10.775 |
| all but `token_embd` | 200 | 5,331,148,608 | 0.001993 | 0.0772 | 0.00877 | 1.272 | 97.28 % | 10.766 |
| all but the MLP | 105 | 6,858,547,008 | 0.001686 | 0.0907 | 0.00603 | 1.439 | 97.67 % | 10.760 |
| the full Q8_0 file | 201 | 4,735,180,608 | 0.00202 | 0.175 | 0.0085 | 1.56 | 97.21 % | 10.778 |

The KL does not add over the classes. The five single classes give 0.00465 of mean KL together, and
the full file gives 0.00202. The pair of `token_embd` and the MLP gives 0.00101, and the two alone
give 0.00203. Thus an "only" row over-states the share of its class, and the "all but" rows are the
rows that a recipe must read.

The "all but" rows rank the classes by the tail that each one removes from the full file, per
megabyte that it adds:

| Class in F16 | Added MiB | 99.9 % KL removed | 99.9 % KL per added MiB | Max KL removed | Mean KL removed |
|---|---|---|---|---|---|
| `ssm_out` | 225 | 79 % (0.175 → 0.0369) | 6.1e-4 | 65 % | 31 % |
| `token_embd` | 568 | 56 % (0.175 → 0.0772) | 1.7e-4 | 18 % | 1 % |
| the MLP | 2025 | 48 % (0.175 → 0.0907) | 4.2e-5 | 8 % | 17 % |

`ssm_out` removes the tail 3.5 times more efficiently than the embedding and 14 times more
efficiently than the MLP, per megabyte. The chunk record agrees: the running mean KL of the
`ssm_out` file rises by 0.00017 at chunk 14, which is the largest rise of any single class, and the
GDN qkv and gate file rises by 0.00001 at the same chunk. The unsloth per-tensor sweep of the
35B-A3B ranks `ssm_out` as by far the most sensitive class, and this measurement of the 4B agrees
with it on a different model and at a different bit width.

The 2B needs no such guard. Its Q8_0 file has a maximum KL of 0.028 and a 99.9 % point of 0.014,
thus the 4B tail comes from the model and not from the packer. The commands, from the project root:

    for R in '^token_embd\.weight$' 'ffn_(gate|up|down)\.weight' 'attn_(q|k|v|output)\.weight' \
             'attn_qkv\.weight|attn_gate\.weight' 'ssm_out\.weight'; do
      .venv/bin/python -m quant.run export --model Qwen3.5-4B --device cpu \
          --f16 weights/gguf/Qwen3.5-4B-F16.gguf --packs quant-out/empty --bulk Q8_0 --gdn-gate Q8_0 \
          --only "$R" --tag only
      llama.cpp/build-cuda/bin/llama-perplexity -m weights/gguf/Qwen3.5-4B-only.gguf -ngl 16 \
          -f data/wiki.test.raw -c 512 --chunks 16 \
          --kl-divergence-base eval/Qwen3.5-4B-F16.wiki.c512x16.kld --kl-divergence
    done
    .venv/bin/python -m quant.run export --model Qwen3.5-4B --device cpu \
        --f16 weights/gguf/Qwen3.5-4B-F16.gguf --packs quant-out/empty --bulk Q8_0 --gdn-gate Q8_0 \
        --only 'ssm_out\.weight' --invert --tag Q8_0-f16-ssmout

## The 3-bit question (2026-09-18)

The verdict is no. No 3-bit type of ggml runs on the Hexagon backend or on the Adreno OpenCL
backend of llama.cpp c6824a9. A 3-bit file of this pipeline would run on the phone CPU only. Thus
the pipeline stays at 4 bits, and this section records the survey and the cost.

| Type | Block | Bytes | Bits per weight | Hexagon | Adreno OpenCL |
|---|---|---|---|---|---|
| Q8_0 | 32 | 34 | 8.500 | yes | yes |
| Q6_K | 256 | 210 | 6.562 | yes | yes |
| Q4_0 | 32 | 18 | 4.500 | yes | yes |
| IQ4_NL | 32 | 18 | 4.500 | yes | yes |
| Q4_K | 256 | 144 | 4.500 | yes | yes |
| MXFP4 | 32 | 17 | 4.250 | yes | yes |
| Q3_K | 256 | 110 | 3.438 | no | no |
| IQ3_S | 256 | 110 | 3.438 | no | no |
| IQ3_XXS | 256 | 98 | 3.062 | no | no |
| Q2_K | 256 | 84 | 2.625 | no | no |

The Hexagon backend has no reference to `Q3_K`, `IQ3_S` or `IQ3_XXS` in `ggml/src/ggml-hexagon/`.
Its matmul takes the types of `ggml_hexagon_is_repack_type`: Q4_0, Q4_1, Q8_0, IQ4_NL, MXFP4, Q4_K
and Q6_K. The OpenCL backend names no 3-bit type in `ggml_backend_opencl_supports_op` for
`GGML_OP_MUL_MAT`, thus the scheduler puts such a matmul on the CPU. Its one code path that names
`Q3_K` selects the Q4_K kernel and then stops at `GGML_ASSERT(false && "not implemented")`.

A Q3 block grid in `quant/grids.py` is cheap, and it is not the blocker. The `Grid` class takes any
number of levels, and the solver and the block optimization read `grid.levels` only. Two places
assume sixteen levels: `pack_nibbles` (the 4-bit byte layout of ggml) and `fit_codebook`. Thus a
grid of eight levels with its packer is approximately 40 lines, and it needs less than two hours.

The blocker is the file format and the two kernels. ggml has no 3-bit type with a block of 32 and
one F16 scale, which is the contract of `Grid`. Its 3-bit types are superblocks of 256 with a
second scale level: `Q3_K` holds a 6-bit scale for each 16 weights, an F16 super-scale and a mask
of the high bit, and `IQ3_S` and `IQ3_XXS` add codebook tables. Thus a 3-bit file needs one of two
paths, and neither is two hours:

- A new ggml type: the enum value, the traits, the CPU kernels, the gguf-py support, an HVX kernel
  for the Hexagon backend and an OpenCL kernel. An upstream build then does not load the file.
- The Q3_K layout: a second quantizer in the pipeline for its two scale levels, plus the same two
  kernels. This is the shorter path of the two, because the two backends hold Q4_K and Q6_K
  superblock kernels that a Q3_K kernel can follow.

MXFP4 is the one step below Q4_0 that the two backends run today: 4.25 bits per weight, 5.6 % less
than Q4_0. But its block scale is a power of two (E8M0, one byte) and its levels are the fixed FP4
table. The gain of this pipeline comes from the fitted F16 scale of each block, from the rotation
and from the folded column scales. Thus MXFP4 gives up the scale search for 5.6 % of the bytes.
Measure it before any use.

## The per-class plan on the row-7 packs (2026-09-18)

Which class earns its bytes at 8 bits? Each row takes the solved Q4_0 packs of row 7 and moves one
class to round-to-nearest Q8_0. The other classes keep their packs, thus the rows differ in one
class only. Q8_0 is near-lossless, thus round-to-nearest needs no calibration for a moved class.

`ssm_out` wins by a wide margin: 36 MiB removes 15 % of the mean KL and 45 % of the maximum KL. It
is 2.6 times better per megabyte than `ffn_down`, 3.4 times better than the attention q and o
pair, and 5.2 times better than the GDN qkv.

| # | Class at Q8_0 | Bytes | Added MiB | Mean KL | 99.9 % KL | Max KL | Top-1 | PPL | Mean KL per added MiB |
|---|---|---|---|---|---|---|---|---|---|
| a | none, the row-7 file (control) | 1,743,744,000 | 0 | 0.029588 | 0.590 | 7.370 | 91.397 % | 14.430 | |
| b | `ssm_out` (GDN out), 18 tensors | 1,781,492,736 | 36 | 0.025223 | 0.461 | 4.079 | 91.250 % | 14.382 | 1.21e-4 |
| c | `attn_q` and `attn_output`, 12 tensors | 1,781,492,736 | 36 | 0.028311 | 0.583 | 7.504 | 91.250 % | 14.410 | 3.55e-5 |
| d | b and c together, 30 tensors | 1,819,241,472 | 72 | 0.024027 | 0.441 | 4.333 | 91.716 % | 14.378 | 7.72e-5 |
| e | `ffn_down` (MLP down), 24 tensors | 1,894,738,944 | 144 | 0.022919 | 0.404 | 7.177 | 91.716 % | 14.347 | 4.63e-5 |
| f | `attn_qkv` (GDN qkv), 18 tensors | 1,856,990,208 | 108 | 0.027082 | 0.493 | 7.027 | 91.275 % | 14.383 | 2.32e-5 |
| g | b, c, e and f together, 72 tensors | 2,083,482,624 | 324 | 0.015025 | 0.199 | 2.765 | 93.260 % | 14.220 | 4.49e-5 |
| h | `token_embd` at Q4_0, not Q8_0 (row 10) | 1,489,464,320 | −242.5 | 0.030870 | 0.788 | 8.073 | 91.176 % | 14.471 | 5.29e-6 |

Row a reproduces row 7c and row h reproduces row 10, thus the harness is the harness of those rows.
Row h reads in the other direction: the embedding at Q8_0 costs 242.5 MiB and gives 0.00128 of mean
KL, which is 23 times less KL per megabyte than `ssm_out`. Thus the embedding is the first place to
take bytes away, and `ssm_out` is the first place to give them.

The rows on the Pareto front, from small to large: h (1.39 GiB), a (1.62 GiB), b (1.66 GiB),
d (1.69 GiB), e (1.76 GiB), g (1.94 GiB). Row f is the one row that another row dominates: d is
smaller and its KL is lower. Row g gives a little less than the sum of its parts: it removes
0.01456 of mean KL, and rows b, c, e and f remove 0.01482 together. It adds 1.9 points of top-1.

`ssm_out` is also the one class that moves the maximum KL. It takes the maximum from 7.37 to 4.08,
and `ffn_down`, `attn_qkv` and the attention pair leave it at 7.0 to 7.5. The 4B Q8_0 measurement
of the section above names the same class for its tail. The unsloth per-tensor sweep of the
35B-A3B ranks `ssm_out` first as well. Three measurements on three models agree.

### How a class moves to Q8_0 without a new calibration

A moved class needs its weight in the coordinates of the calibration, because the folds moved the
columns of the solved classes, the channel order of the MLP and the norms. The folded reference
`weights/Qwen3.5-2B-tf` of row 7 is on the box and not on the laptop. Thus `--promote` takes the
unfolded source (`--source tu`) and `quant/refold.py` makes the folded weight from three sources:

- The norm ratio. The column scales of a class sit in the norm that feeds it, thus the source norm
  divided by the folded norm gives them. This is exact for the mixer inputs, for `ssm_out`
  (`ssm_norm`), for the MLP gate and up (`post_attention_norm`) and for the head.
- The row ratio of the folded `attn_v`, which holds the column scales of `attn_output`.
- The packs of `ffn_gate` and `ffn_up`, which give the channel permutation by the cosine of each
  row, and the `ffn_down` column scales by the least-squares ratio of the rows.

The strongest check compares each refolded tensor with its own pack, over all the 138 layer packs of
row 7. The pack holds the 4-bit rounding of the folded weight, thus the difference must be the
4-bit rounding error and nothing more. It is, for every class:

| Class | Packs | Relative difference against the pack (min, mean, max) | The recovery |
|---|---|---|---|
| `attn_qkv` | 18 | 0.0947, 0.0966, 0.0984 | the norm ratio, exact |
| `attn_gate` | 18 | 0.0946, 0.0967, 0.0985 | the norm ratio, exact |
| `attn_q` | 6 | 0.0942, 0.0967, 0.0995 | the norm ratio, exact |
| `ffn_gate` | 24 | 0.0935, 0.0963, 0.1027 | the norm ratio and the row match |
| `ffn_up` | 24 | 0.0935, 0.0963, 0.1027 | the same, and the `ffn_down` scales |
| `ssm_out` | 18 | 0.1012, 0.1105, 0.1254 | the norm ratio, exact |
| `attn_output` | 6 | 0.1043, 0.1102, 0.1196 | the rows of the folded `attn_v` |
| `ffn_down` | 24 | 0.0930, 0.0998, 0.1191 | the row ratio of the `ffn_up` packs |

A wrong coordinate would show at once, because the column scales have an rms log of 0.04 to 0.31,
which is 4 % to 31 % per column. Every class stays at 9.3 % to 12.5 %, which is the 4-bit error of
Q4_0 on these weights. `ssm_out` and `attn_output` hold the two highest values, and that is the
property of those matrices and not of the recovery: `ssm_out` is the class that this whole section
ranks as the hardest. The `ffn_up` row is not an independent check, because the `ffn_down` scales
come from the `ffn_up` packs.

The one estimated quantity is the `ffn_down` column scale. A row of `ffn_up` has 2048 columns, and
the least-squares ratio averages the 4-bit error over them. Thus its noise is approximately
0.096 / sqrt(2048), which is 0.2 %, of the order of the Q8_0 rounding itself.

Three more numbers check the result, because `search_column_scales` makes scales of geometric mean one.
The recovered `ssm_norm` scales have a geometric mean of 0.9989 to 1.0003, the `attn_v` row scales
1.0000 to 1.0001, and the `ffn_down` scales 0.9967 to 0.9998 before the correction of their mean.
The row match of the MLP has a minimum cosine of 0.9935 and a minimum margin of 0.022 over the 24
layers, and the two packs of a layer always give the same permutation. A moved tensor then holds
the FP weight of the original checkpoint in the coordinates of the file, which is what 8 bits must
hold. The tests of `quant/tests/test_refold.py` give the exact classes and the approximate ones.

The numbers of rows b to g are a lower bound of a calibration that knows the plan. The solver chose
the column scales of the moved class for a 16-level grid, and a calibration with `ssm_out` at Q8_0
would choose them for 256 levels, and the block optimization would then refine the rest around it.
The commands, from the project root with the `.venv`:

    .venv/bin/python -m quant.run export --model Qwen3.5-2B --device cuda --source tu \
        --packs quant-out/Qwen3.5-2B-bo-gptq-frozen-e2 --promote 'ssm_out\.weight' --tag q8-ssm-out
    .venv/bin/python -m quant.run export --model Qwen3.5-2B --device cuda --source tu \
        --packs quant-out/Qwen3.5-2B-bo-gptq-frozen-e2 --promote 'attn_q\.weight|attn_output\.weight' --tag q8-attn-qo
    .venv/bin/python -m quant.run export --model Qwen3.5-2B --device cuda --source tu \
        --packs quant-out/Qwen3.5-2B-bo-gptq-frozen-e2 --promote 'ffn_down\.weight' --tag q8-ffn-down
    .venv/bin/python -m quant.run export --model Qwen3.5-2B --device cuda --source tu \
        --packs quant-out/Qwen3.5-2B-bo-gptq-frozen-e2 --promote 'attn_qkv\.weight' --tag q8-gdn-qkv
    .venv/bin/python -m quant.run export --model Qwen3.5-2B --device cuda --source tu \
        --packs quant-out/Qwen3.5-2B-bo-gptq-frozen-e2 \
        --promote 'ssm_out\.weight|attn_q\.weight|attn_output\.weight|attn_qkv\.weight|ffn_down\.weight' \
        --tag q8-all-but-mlp-gate-up
    .venv/bin/python -m quant.run eval --model Qwen3.5-2B --gguf weights/gguf/Qwen3.5-2B-<tag>.gguf

## The plan for the 4B tied Q4_0 run (2026-09-18)

The plan: the bulk Q4_0, `ssm_out` and `ffn_down` at Q8_0, the k and v projections Q8_0, the head
tied and Q4_0, the recurrence controls and the norms F32, the MTP block Q8_0. The two guards come
from the 2B measurement of the section above, which ranks them first and second in KL per byte, and
from the 4B Q8_0 measurement, which names `ssm_out` as the class of the tail.

| Class | Tensors | Weights | Type | MiB |
|---|---|---|---|---|
| `token_embd` (the tied head) | 1 | 635.7 M | Q4_0 | 341.0 |
| `ffn_gate`, `ffn_up` | 64 | 1510.0 M | Q4_0 | 810.0 |
| `ffn_down` | 32 | 755.0 M | **Q8_0** | 765.0 |
| `attn_qkv` | 24 | 503.3 M | Q4_0 | 270.0 |
| `attn_gate` | 24 | 251.7 M | Q4_0 | 135.0 |
| `ssm_out` | 24 | 251.7 M | **Q8_0** | 255.0 |
| `attn_q` | 8 | 167.8 M | Q4_0 | 90.0 |
| `attn_output` | 8 | 83.9 M | Q4_0 | 45.0 |
| `attn_k`, `attn_v` | 16 | 41.9 M | Q8_0 | 42.5 |
| the F32 tails and the norms | 217 | 4.9 M | F32 | 18.7 |
| `output_rot` | 1 | 6.6 M | F16 | 12.5 |
| the MTP block (`blk.32`) | 15 | 120.6 M | Q8_0 + 2 F16 maps | 147.3 |
| the metadata and the tokenizer | | | | 10.5 |

Three sizes, by the guards that the run takes:

| Guards | Size | Expected mean KL | Note |
|---|---|---|---|
| none, the bulk at Q4_0 | 2.41 GiB | 0.030 | the plan of row 11 on the 4B |
| `ssm_out` Q8_0 | 2.52 GiB (+120 MiB) | 0.025 | the best value per byte, take it always |
| `ssm_out` and `ffn_down` Q8_0 | 2.87 GiB (+480 MiB) | 0.019 | take it when 2.9 GiB fits the phone |

The KL column is an extrapolation of the 2B ratios (−15 % for `ssm_out`, −37 % for the pair), and
not a measurement of the 4B. Do not assume that the 4B is easier than the 2B: its Q8_0 file has a
mean KL of 0.00202 against 0.00115 for the 2B. The first run must measure the baseline and the two
guards, and it must record the maximum KL, because `ssm_out` is the class that moves it.

Three classes stay at Q4_0 on purpose. `attn_qkv` costs 270 MiB more at Q8_0 and gives the worst
KL per byte of the five classes on the 2B. `attn_q` and `attn_output` give 0.0013 of mean KL on the
2B, and the 4B has 8 attention layers of 32, the same share. The embedding stays the tied Q4_0
head: the 4-bit lookup costs 0.0013 of mean KL on the 2B and saves 341 MiB, which is the best trade
of the file in the other direction.

`ssm_out` at F16 costs 225 MiB more than Q8_0. The 4B Q8_0 file says that the Q8 rounding of
`ssm_out` still makes a maximum KL of 1.56. But a Q4_0 file has a maximum KL of about 7 of its own
(7.37 for the 2B row 7), thus Q8_0 is enough for `ssm_out` here. Take F16 for `ssm_out` only in a
Q8_0 file, where it takes the maximum from 1.56 to 0.553.

The plan needs no new machinery. `--ssm-out` and `--ffn-down` are switches of `quant/plan.py`, and
a Q8_0 class already takes the whole path: the solver gives it the round trip, the block
optimization trains it and rounds it again at the end of each layer, `save_folds` writes it, and
the export reads it from the folds. The k and v projections have gone that way since row 3. The
commands, from the project root with the `.venv`:

    python -m quant.run transform --model Qwen3.5-4B --device cpu --tie-head
    python -m quant.run convert   --model Qwen3.5-4B --source t
    python -m quant.run quantize  --model Qwen3.5-4B --device cuda --stream \
        --bulk Q4_0 --embedding Q4_0 --kv-proj Q8_0 --gdn-gate Q4_0 --ssm-out Q8_0 --ffn-down Q8_0 \
        --init gptq --method blockopt --epochs 2 --freeze-weights --head-steps 300 --head-chunk 8192
    python -m quant.run convert   --model Qwen3.5-4B --source tf
    python -m quant.run export    --model Qwen3.5-4B --source tf \
        --bulk Q4_0 --embedding Q4_0 --kv-proj Q8_0 --gdn-gate Q4_0 --ssm-out Q8_0 --ffn-down Q8_0 \
        --tie-head --mtp Q8_0 --tag tied-q4-guards
    python -m quant.run eval      --model Qwen3.5-4B --gguf weights/gguf/Qwen3.5-4B-tied-q4-guards.gguf --ngl 16

Two notes for the run. The 4B has 16 key heads and 32 value heads, thus the converter tiles the
value heads of `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta` and the columns of `ssm_out`, and
`LinearAttentionLayout` of the export permutes the packs the same way. The 2B has one value head
per key head, thus the 2B rows never took that path, and the first 4B file must pass the
`llama-completion` check before any KL run. And the export above reads the folded reference, thus
it needs no `--promote` and no `quant/refold.py`.

The two product models on the NPU, 2026-09-18, thermal status 0, the series of 18 patches
(llama-bench -p 512 -n 64, and llama-server --spec-type draft-mtp --spec-draft-n-max 3 on a
free-form prompt of 96 tokens):

| File | Prefill t/s | Decode t/s | Decode with MTP t/s | Accepted |
|---|---|---|---|---|
| Qwen3.5-2B-Q8_0 | 1558 | 23.6 | 27.8 to 28.8 | 77 to 82 % |
| Qwen3.5-4B-Q8_0 | 755 | 10.4 | 15.4 | 71 % |

The morning of the same day gave 981 / 21.5 for the 2B and 382 / 7.0 for the 4B, thus the prefill
of both models is twice as fast. The chunked gated delta rule carries that gain, and the host
patches make the speculative decoding pay. The multi-row matmul of a verify step is neutral end to
end: the weight stream of a step already hides its compute, thus the kernel gives a margin for a
longer draft and not a time today.
