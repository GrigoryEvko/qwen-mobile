"""Quantization pipeline for Qwen3.5 into a GGUF grid that the plan selects.

The stages are: transform (fold the norms, apply the residual rotation,
permute the free dimensions), convert (the llama.cpp converter writes the
F16 GGUF), quantize (calibrate and solve each matrix into the grid), and
export (replace the big matrices in the F16 GGUF with the packed blocks).

The grid of each tensor class comes from the plan of the run, not from this
package: ``--bulk``, ``--head``, ``--embedding`` and the other plan flags of
``quant.run`` give it. Refer to analysis/quant-results.md for the rows.
"""
