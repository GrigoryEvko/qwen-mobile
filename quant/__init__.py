"""Quantization pipeline for Qwen3.5 into the Q4_0 GGUF grid.

The stages are: transform (fold the norms, apply the residual rotation,
permute the free dimensions), convert (the llama.cpp converter writes the
F16 GGUF), quantize (calibrate and solve each matrix into the grid), and
export (replace the big matrices in the F16 GGUF with the packed blocks).
"""
