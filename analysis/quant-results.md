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

Grid test on Gaussian weights (256 x 512, block-32 F16 scales with the scale search, 4.5 bits per
weight, `python -m quant.trellis`): Q4_0 21.8 dB, IQ4_NL 22.2 dB, codebook per matrix 22.3 dB,
trellis (bit-shift, L = 12) 22.0 dB with a Gaussian table and 21.7 dB with the Ungerboeck cosets.
The codebook grid is the best 4-bit grid at equal bytes on this test.

The HF lockstep drift report (analysis/*.drift.md, 16 x 1024 WikiText tokens, fp32 on CUDA) gives
the same ranking with its own numbers: row 3 mean KL 0.0291, row 4 0.0314. The KL attribution
of row 3 is in analysis/quant-attribution.md: the round-to-nearest head alone is 0.011 of the
0.037, MLP gate/up 0.009, MLP down 0.007, GDN out 0.005, GDN qkv 0.0025, layers 0-2 0.0075.
