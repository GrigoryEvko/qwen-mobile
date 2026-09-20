# A quantization recipe for a 2B to 4B hybrid model on an integer NPU

Target: Qwen3.5 2B and 4B, a hybrid of gated delta net layers and full attention
layers, on the Hexagon v79 HMX matrix engine of a Snapdragon 8 Elite.

Format: **per-channel int8 weights** (one scale for each output row) and
**per-token u8 activations**. Written 2026-09-20.

This document is a procedure, not a survey. Every number in it is either a
measurement of our own model, with the command that produces it, or a published
number with a link. Where the two disagree, our measurement wins and the
document says so.

## How to read this

Sections 1 and 2 give the hardware contract and the error budget. Sections 3
to 8 are the recipe, in the order that a build must run them. Section 9 gives
the evaluation, and it is not optional: three of the failure modes in this
document are invisible to the metric we use today. Section 10 answers the
4-bit question. Section 11 gives the build order with effort. Section 12 lists
what not to do, with the number that rules each item out.

A reader who wants only the decisions can read section 11 and section 12.

---

## 1. The hardware contract

### 1.1 The one rule that sorts the whole literature

The HMX matrix engine accumulates int32 across the **whole** inner dimension K.
A scale is legal only when it is constant along K. Two scale shapes satisfy
this:

- One scale for each token, which is constant along K by construction
- One scale for each output channel, which is also constant along K

Every other shape fails. A scale for each block of 32 along K, a scale for
each group of 128 along K, and a per-16-element microscale are all illegal,
because the engine cannot break the sum and rescale in the middle of it.

This rule is not a preference. It removes the following from consideration,
whatever their accuracy: Atom, QUIK, NVFP4 and MXFP4 microscaling, the
per-state-group scales of Quamba2, and the key-axis mixed precision of DAMP.
It also removes every method that keeps outlier channels in a separate
higher-precision kernel: LLM.int8, SVDQuant, QUAD, and the shadow outlier path
of llm.npu.

Two independent measurements give the price of breaking the rule on this
silicon. llm.npu measured a per-group matmul at **8.1 to 10.7 times** the cost
of a per-tensor matmul on a Hexagon NPU with QNN. Quant.npu measured
per-tensor activations with per-channel weights at **1.2 times** the speed of
per-block weights.

### 1.2 What the engine gives back

Measured on the device, in `tools/hmx-bench`: the int8 mode gives **2.05 times**
the throughput of the f16 mode. The block geometry is 64 x 32 x 32 for int8
against 32 x 32 x 32 for f16, thus one int8 issue does 65536 multiply-accumulate
operations where one f16 issue does 32768.

The activation operand is unsigned 8-bit (`activation.ub`) and the weight
operand is signed 8-bit (`weight.b`). A zero point on the activation is
available and a zero point on the weight is not, which is the standard
arrangement and the correct one. Refer to section 5.2.

**The 2.05 figure is the multiply alone, and the accumulator read is not free.**
The int32 accumulator has no f16 store and no f32 store. It leaves only as u8 or
as u16, thus a full int32 read is four byte-plane stores of
`mxmem(out, rN):after:retain:cm.ub = acc`, each with its own scale in the bias
register. The intercepts of the same measurement price it. The f16 read costs
nothing that the fit can see, and the int8 read costs about **711 processor
cycles** for one 64 x 32 tile. One read covers the whole k range of one output
tile, thus at K = 2560 the arithmetic is:

| | f16, 64 rows | int8, 64 rows |
|---|---|---|
| multiply | 160 issues, 5558 cyc | 80 issues, 2707 cyc |
| accumulator read | about 0 | about 711 cyc |
| total | 5558 | 3418 |

That is **1.63 times, not 2.05**. There is a worse case. Our own kernel reaches
about 9 cycles for one f16 tile where the HexKL call costs 34, thus most of the
per-issue figure is call overhead and not engine time. If the 711 cycles is
mostly engine time, the true ratio falls toward 1.04. Do not plan on 2.05 until
a device run settles it. `tools/hmx-bench` now times the read on its own.

**The instruction set is reachable without HexKL, and that matters because the
HexKL archive cannot ship.** The macros are in the tree at
`ggml/src/ggml-hexagon/htp/hmx-utils.h`, behind patch `hexagon-hmx-i8/0001`.
They compile with the Hexagon compiler and emit the same encodings as the vendor
library. The multiply packet is
`{ activation.ub = mxmem(act, 0x1f):cm ; weight.b = mxmem(wt, 0x380) }`, and the
k tiles chain with `:deep:cm` in that order.

### 1.3 Where the time goes

Decode is bandwidth bound. The 4B streams 4610 MB for each token at about
51 GB/s. Prefill is compute bound above 120 rows, at 12.5 TFLOPS measured, and
matmul is 44 % of it.

Thus the two levers are different for the two phases:

- **Decode**: only the bytes of each token matter. A per-row int8 weight is
  8.031 bits against 8.5 for Q8_0, thus about 1.14 times the decode speed once
  the per-block scale multiply leaves the matvec.
- **Prefill**: the int8 mode of the HMX matters, up to 1.19 times end to end.

**Compute at run time is nearly free relative to bandwidth.** A fast Hadamard
transform on a vector of length K costs `K·log2(K)` add and subtract operations
and no multiply. Against a matmul of `2·K·M` operations the share is
`log2(K)/(2M)`. At K = 8192 and M = 2048 that is 0.32 %. Thus an online
transform that adds flops and no bytes is the cheapest thing this machine can
buy, and the recipe uses several.

### 1.4 The byte budget

From `tools/prof/bytes.py budget 4b`, at 8 bits:

| Class | MB for each token | Share |
|---|---|---|
| MLP | 2406.5 | 52.5 % |
| GDN projections | 1069.5 | 23.4 % |
| output head | 675.4 | 14.7 % |
| attention projections | 312.0 | 6.8 % |
| GDN state read and write | 100.7 | 2.2 % |
| GDN per-head maps | 15.7 | 0.3 % |
| total | 4579.8 | |

Two numbers in that table decide two sections of this recipe. The MLP is more
than half the bytes, thus section 10 is about the MLP and nothing else. The
recurrent state is 2.2 %, thus section 6.1 says to leave it alone.

### 1.5 What the vector unit permits, and it constrains every format

The matrix engine sets the scale rule of section 1.1. The vector unit sets a
second rule that decides which **grids** are affordable, and it is easy to miss.

**HVX gives one 32-entry byte lookup for each instruction.** Everything else is
a chain:

| Table size | Lookup passes for each vector | Example |
|---|---|---|
| 16 entries | about 1 | `IQ4_NL`, one instruction, free |
| 256 entries | about 8 | an 8-bit trellis or an E8 source table |
| 1024 entries | about 32 | a 10-bit trellis |
| 4096 entries | about 128 | a 12-bit trellis |

The unpack budget is tight. At 4.6 GB for each token and 51 GB/s the decode has
90 ms, thus about 91 G weight-decodes for each second. `Q4_0` and `IQ4_NL` cost
3 to 4 HVX operations for each 128 weights. A 32-pass gather is more than an
order of magnitude above that.

This single fact, and not accuracy, rules out every large codebook on this
part. Section 12.10 gives the measured curve.

**One exception, and it is an instruction and not a grid.** The matrix engine
has a native 4-bit weight load, `weight.n`, and HexKL ships `mm_u8i4` for it.
The geometry is the same 64 x 32 x 32 and the issue count is the same. Only the
weight tile changes, from 1024 bytes to 512, and the range operand from 0x380
to 0x180. The engine unpacks the nibbles itself, thus the lookup budget of this
section does not reach a 4-bit weight on the matmul path. That buys bytes and
not issues, which is exactly the decode lever of section 1.3. It does not touch
the accuracy question, and the scale rule of section 1.1 still binds: such a
weight needs one scale for each output channel, not one for each block of 32.
Refer to section 10.

Two related facts, both worth a check in our tree.

**Does `IQ4_NL` even reach the matrix engine today?** The Hexagon backend lists
`HTP_TYPE_Q4_0_TILED`, `Q4_1_TILED`, `Q8_0_TILED` and `MXFP4_TILED`. There is
**no** `IQ4_NL_TILED`. If `IQ4_NL` falls to the HVX path while `Q4_0` reaches
HMX, then every grid comparison we have made compares two different kernels.
Verify this before reading any speed number.

**The three Cornell repositories are GPLv3.** QTIP, QuIP# and YAQA are all
GPL-3.0. Our tree is MIT and Apache. Their code cannot be reused, only their
papers. Plan any adoption as a clean re-implementation.

---

## 2. The error budget, measured on our model

### 2.1 What a per-row weight scale costs

For a weight row the int8 step comes from the maximum of the scale group. Thus
one scale for each row against one scale for each block of 32 multiplies the
rounding error by

    ratio = max(|row|)^2 / mean_b(max_b(|row|)^2)

Measured on `weights/Qwen3.5-2B` (original) and `weights/Qwen3.5-2B-t` (norms
folded, Hadamard R1, head untied), against a Gaussian reference of the same
shape:

| Class | Original | After R1 | Gaussian minimum | Above the minimum |
|---|---|---|---|---|
| embedding | 2.99 | 2.32 | 2.32 | 0 % |
| GDN gate z | 2.81 | 2.31 | 2.32 | 0 % |
| MLP gate | 2.64 | 2.34 | 2.32 | +1 % |
| MLP up | 2.56 | 2.34 | 2.32 | +1 % |
| attn v | 3.73 | 2.39 | 2.32 | +3 % |
| attn q | 3.01 | 2.43 | 2.32 | +5 % |
| MLP down | 3.92 | 2.90 | 2.68 | +8 % |
| attn k | 3.98 | 2.62 | 2.32 | +13 % |
| GDN out | 3.48 | 2.63 | 2.32 | +13 % |
| attn o | 3.64 | 3.11 | 2.32 | **+34 %** |

Three readings follow.

**The per-row scale costs a factor of 2.3 to 3.1, and not more.** The 2B Q8_0
file has a mean KL of 0.00115 against an F16 floor of 0.00020, thus the
rounding share is about 0.00095. At 2.5 times, a per-row int8 file lands near
**0.0025 to 0.003 of mean KL**. The shipped Q4_0 file is 0.0295. The 8-bit
decision buys about an order of magnitude, and the per-row form costs a factor
of 2.5 of a number that is already small. This is an estimate from the ratio
and not a measurement of the file.

**The existing Hadamard R1 already does almost all the available work on the
weight side.** Every class whose reduction axis R1 rotates sits at the Gaussian
minimum, to within 5 %. That is the incoherence property, measured on our
model. A learned rotation cannot beat a value that is already at the minimum,
and section 12.3 gives the theory that says the same.

**The distance that remains sits on the three output projections.** `attn_o`
at +34 %, `ssm_out` at +13 % and `ffn_down` at +8 % read an internal activation
after an elementwise gate, thus R1 never touches their reduction axis.
`attn_k` at +13 % is an anomaly, because its input is the rotated residual. Do
a check of it.

### 2.2 What a per-row clipping search costs, and why not to build one

I swept the clip limit as a factor of the row maximum, at 8 bits, on the
rotated checkpoint, and took the factor that minimizes the row error:

| Tensor, layer 5 of `Qwen3.5-2B-t` | Best factor | Error gain against absmax |
|---|---|---|
| `in_proj_qkv` | 0.990 | 1.020 |
| `in_proj_z` | 0.990 | 1.020 |
| `in_proj_a` and `in_proj_b` | 0.989 and 0.991 | 1.020 |
| `linear_attn.out_proj` | 0.990 | 1.021 |
| `mlp.gate_proj` and `mlp.up_proj` | 0.990 | 1.021 |
| `mlp.down_proj` | 0.973 | **1.037** |

The optimum sits at 0.97 to 0.99 of absmax and the gain is **2 % to 4 %**.
Against a per-row cost of 2.3 to 3.1 times, that is nothing.

The theory agrees. The ACIQ condition puts the MSE-optimal Laplace clip at
about 9.9b at 8 bits, while a rotated row of 2048 has a maximum near 4.6b.
Thus absmax already sits inside the optimum and a search wants to widen, not
narrow. OCTAV measured the same at 8 bits for each output channel, where plain
max-scaling wins on two of six models. Production agrees: `llm-compressor`
issue 2094 reports the MSE observer **worse** than min-max, 0.7767 against
0.7171 on GSM8K Platinum, closed with no fix.

Note the ordering in the table. `mlp.down_proj`, the one class whose reduction
axis R1 does not rotate, is the only one with a factor below 0.99. Clipping and
incoherence attack the same quantity, and the rotation already took it.

### 2.3 What the activation costs, and this is the part that matters

Per-site activation statistics of the unrotated 2B, over 2 sequences of 512
tokens of the calibration set. `u8 SNR` is the signal-to-noise ratio of a
symmetric per-token int8 quantizer. `+had128` applies a block-128 Hadamard to
the input first. Sites marked `*` sit after an elementwise gate and thus cannot
take a folded rotation.

| Site | max/rms | u8 SNR dB | +had128 dB | gain dB |
|---|---|---|---|---|
| GDN qkv | 17.0 | 28.5 | 41.0 | +12.4 |
| GDN gate z | 17.0 | 28.5 | 41.0 | +12.4 |
| GDN out `*` | 16.8 | 28.5 | 38.7 | +10.2 |
| MLP gate and up | 16.3 | 28.7 | 41.1 | +12.4 |
| MLP down `*` | 24.2 | 24.7 | 39.3 | **+14.6** |
| attn qkv | 18.6 | 27.5 | 40.7 | +13.2 |
| attn o `*` | 12.9 | 30.6 | 40.3 | +9.7 |

The same measurement on `weights/Qwen3.5-2B-t`, which carries the folded norms
and the Hadamard R1. **This table is the most important one in the document.**

| Site | max/rms | u8 SNR dB | +had128 dB | gain dB |
|---|---|---|---|---|
| GDN qkv | **3.5** | **42.0** | 41.7 | −0.3 |
| GDN gate z | **3.5** | **42.0** | 41.7 | −0.3 |
| GDN out `*` | 16.8 | 28.5 | **38.7** | **+10.2** |
| MLP gate and up | **3.4** | **42.2** | 41.7 | −0.4 |
| MLP down `*` | 24.2 | 24.7 | **39.3** | **+14.6** |
| attn qkv | **3.5** | **41.9** | 41.6 | −0.3 |
| attn o `*` | 12.9 | 30.6 | **40.3** | **+9.7** |

Four readings, and together they are the recipe.

**R1 is worth 13.5 dB on every site it reaches.** The four residual-stream
readers go from 27.5 to 28.7 dB up to 41.9 to 42.2 dB. That is a factor of 22
in mean squared error, from a transform we already ship.

**R1 does nothing at the three gated sites.** Their numbers are identical to
the unrotated model, to the last digit. This is not an approximation. The
elementwise gate sits between the rotated residual and the projection input,
thus the rotation cannot reach through it.

**A second Hadamard on an already-rotated site is worth −0.3 dB.** Nothing.
Do not apply an online transform where R1 already applies a folded one. This
kills the QuaRot R3 arrangement for our model, and it agrees with SpinQuant,
which states that at W4A8 "adding additional online Hadamard rotation yields
marginal benefit".

**An online Hadamard at the three gated sites is worth 9.7 to 14.6 dB**, and
it takes them to 38.7 to 40.3 dB, level with the rest of the model. That is
the entire activation-side plan, and it is three sites and nothing else.

One residual note. The three gated sites land 2 to 3 dB behind the R1 sites,
because the measurement used a **block-128** Hadamard and R1 is a full one. A
full Hadamard on 2048, 4096 or 6144 should close that gap, and section 3.4
shows all six dimensions factor. Thus the numbers above are a lower bound.

**Correction of a published expectation.** The Minima paper measured the
`out_proj` input of a 27B gated delta net at a max-to-RMS ratio of 298.1 and
the `down_proj` input at 368.3. Our 2B measures 16.8 and 24.2. **Our model is
more than ten times better behaved than its 27B relative.** Any plan built on
the 27B numbers over-states the risk. Plan against the table above.

**The activation is nevertheless the dominant error.** A per-row int8 weight
of a rotated row gives about 42 dB. The activation gives 24.7 to 30.6 dB. Thus
the activation error is 12 to 17 dB larger than the weight error, which is a
factor of 16 to 50 in mean squared error. **At 8 bits, weight quantization is
a rounding detail and activation quantization is the whole problem.**

A block-128 Hadamard moves every site to 38.7 to 41.1 dB, which is level with
the weight side. That is the single largest change available anywhere in this
recipe.

The ranking of the sites matches the literature exactly. `MLP down` is the
worst and gains the most, because its input is the SwiGLU product, which is
bilinear in the residual and stays heavy-tailed even when both factors are
tame. `attn o` is the best behaved. The two GDN sites that read the normed
residual give identical numbers, which is the sanity check that the harness
works.

### 2.4 The summary of the budget

At the target format, for each matmul, in order of size:

1. **Activation quantization**, 24.7 to 30.6 dB without a transform. Reducible
   to about 40 dB by a block-128 Hadamard.
2. **Weight rounding under a per-row scale**, about 42 dB. Reducible by
   nothing worth building, refer to section 2.2.
3. **The recurrent state**, if quantized. Do not quantize it, refer to
   section 6.1.

Everything else is below these three.

---

## 3. Stage 1: the transform

Nothing in this stage changes the function of the model. Every step is exact
up to floating point, thus every step can be verified against the original
checkpoint before any quantization runs.

### 3.1 Fold the norms

Fold each RMSNorm weight into the linear layers that consume it. Every
residual norm becomes a plain RMSNorm, which commutes with a rotation. This is
already in `quant/transform.py` and it is the precondition for step 3.2.

### 3.2 Rotate the residual stream, R1

Apply one orthogonal Q to the residual stream: the embedding rows, the input
side of every input projection, the output side of every output projection,
and the head. Use `Q = H·D` with H a Hadamard and D a random sign diagonal.
`RMSNorm(Q^T h) = Q^T RMSNorm(h)` for orthogonal Q, thus the function does not
change.

This is already in the pipeline and section 2.1 measures that it works: every
class whose reduction axis it rotates sits at the Gaussian minimum.

Use a **random sign diagonal**, not a bare Hadamard. QuIP# proves that a
randomized Hadamard transform gives incoherence `mu_W = 2 log(4mn/delta)`, a
logarithmic dependence on the matrix size, where a Kronecker construction gives
a log-squared dependence.

### 3.3 Rotate the three output projections, and this is the new work

The three reductions that R1 cannot reach:

| Site | 2B reduction | 4B reduction | Why R1 misses it |
|---|---|---|---|
| `linear_attn.out_proj` | 2048 | 4096 | reads the gated norm output |
| `self_attn.o_proj` | 2048 | 4096 | reads the attention output in head space |
| `mlp.down_proj` | 6144 | 9216 | reads the SwiGLU product |

Each site takes a Hadamard H on its reduction axis. One half folds into the
weight offline as `W ← W·H^T`. The other half runs at run time on the
activation. The elementwise gate that sits in front is exactly why the
activation half cannot fold.

One transform buys two things at once. It flattens the activation, which
section 2.3 measures at +9.7 to +14.6 dB. And it rotates the reduction axis of
the weight, which section 2.1 measures as +34 %, +13 % and +8 % of distance
above the Gaussian minimum.

**Order the three sites by value, not by convenience.** OSTQuant gives the only
clean per-site ablation, at W4A4 on Llama-2-7B, nine-task average: the down
projection site is worth **+8.01 points** and the value and output site is
worth **+0.62 points**. Our own table agrees for the activation: MLP down gains
14.6 dB and attn o gains 9.7 dB. And the byte budget agrees: the MLP is 52.5 %
of the token and the attention projections are 6.8 %.

Thus: **`ffn_down` first, `ssm_out` second, `attn_output` last.**

`ssm_out` ranks second and not third because three independent measurements
name it the most sensitive weight class of this architecture. Our own 4B Q8_0
attribution puts its removal at 79 % of the 99.9 % KL for 225 MiB. Our 2B
per-class table puts it first in KL for each megabyte. The Minima per-projection
replay puts it at 12.7 % of output error against 10.4 % for qkv. The unsloth
per-tensor sweep of the 35B-A3B ranks it first.

### 3.4 The non-power-of-two problem, and our dimensions

A Hadamard of order n needs n to be 1, 2, or a multiple of 4, and the fast
butterfly needs a power of two. AMD Quark cannot apply its R4 to Qwen2-7B at
all, because the intermediate size 18944 factors as 148 x 2^7 and 148 has no
supported Hadamard order.

Our six reduction dimensions, from `analysis/*.structure.txt`:

| Site | 2B | Factors | 4B | Factors |
|---|---|---|---|---|
| `linear_attn.out_proj` | 2048 | 2^11 | 4096 | 2^12 |
| `self_attn.o_proj` | 2048 | 2^11 | 4096 | 2^12 |
| `mlp.down_proj` | 6144 | 12 x 2^9 | 9216 | 36 x 2^8 |

Four of the six are powers of two and take a plain fast Hadamard. The two MLP
dimensions take a Kronecker product: `H_12 ⊗ H_512` for the 2B and
`H_36 ⊗ H_256` for the 4B. Both 12 and 36 are multiples of 4, thus both orders
exist. The pipeline already builds a Kronecker Hadamard for the 4B hidden size
of 2560, thus the machinery exists.

A block-128 Hadamard is the fallback that always divides and always costs
`K·7` operations. Section 2.3 measures the block-128 form, thus the numbers in
this document are the fallback numbers and a full transform can only do
better.

**But do not shrink the block to save operations.** InfoQuant measured, on
Llama-3-8B, 1 block at 7.13 perplexity, 2 blocks at 7.18, 4 blocks at 7.17 and
8 blocks at 7.39. More blocks is worse, and the operations saved are 0.1 % of
a matmul.

### 3.5 Compose the rotation with a KLT, for the gated delta net layers

A Hadamard equalizes channel maxima. It does **not** equalize channel
variances. MambaQuant proves this: the diagonal of the post-rotation covariance
keeps a per-channel index, thus a fixed orthogonal transform cannot adjust the
variance uniformly in all directions.

The fix is `H_K = K·H`, with K the eigenvector matrix of the calibration
covariance at that site. Every post-rotation channel variance then collapses to
the same value. The matrix stays orthogonal, thus it folds offline into the
weight at zero cost at run time.

The measured value at **8 bits**, which is the bit width of this recipe:

| Model | FP16 | RTN | SmoothQuant | QuaRot | MambaQuant |
|---|---|---|---|---|---|
| Mamba-LLM 1.4b | 58.6 | 53.9 | 54.2 | 56.9 | **58.3** |
| Mamba-LLM 2.8b | 62.2 | 58.4 | 58.7 | 59.3 | **62.1** |
| Vim-T | 76.1 | 37.4 | 37.7 | 59.3 | **75.6** |

The cumulative ablation on Vim-T† at W8A8 isolates the step: RTN 32.4, plain
Hadamard 33.9, **KLT-composed 47.7**, plus the online Hadamard 69.7, plus
smooth-fused scaling 77.8. The plain Hadamard is worth 1.5 points and the
variance alignment is worth 13.8.

Cost: 329 k extra parameters on Mamba-2.8b, which is 0.01 % of the model, and
0.91 % of compute at 1024 tokens.

**Caveat, stated plainly.** Every number above is a Mamba model or a vision
Mamba. No published work applies a KLT-composed rotation to a gated delta net.
Our own activation table already sits at 38.7 to 41.1 dB after a plain
block-128 Hadamard, thus the distance left for a KLT is smaller here than the
Vim-T numbers suggest. Measure the step, do not assume it.

### 3.6 Fold the column scales

A linear that reads `x·W^T` can read `(x/t)·(W·diag(t))^T` when its producer
supplies `x/t`. The producers in this model are exact: the zero-centered
RMSNorm weight, the rows of `v_proj`, the rows of `up_proj`, and the shared
weight of the gated norm. This is already `quant/scale.py`.

**Change the objective.** The current search minimizes the weighted block error
of the **weight**. At 8 bits the weight side is nearly free and the activation
side is everything, refer to section 2.4. Search the exponents against the
activation quantization error instead.

**Search the exponent, do not fix it.** The SmoothQuant exponent is
model-specific: 0.5 for OPT and BLOOM, 0.75 for GLM-130B, **0.8 to 0.9 for the
Llama family**, 0.8 for Mistral. A Qwen model at this size is the hard case and
0.5 will be wrong.

**Order the scale before the rotation, at the gated sites only.** Czakó et al.
measured, on Llama2-7B `down_proj` layers 1 and 30, that a rotation alone is
**worse than no transform at all**, because a rotated outlier token clusters
around `2^(|O|-1)` centroids with a worst case of `sum|o_i|/sqrt(d) + |eps|`.
A scale migration first fixes it. Our own table shows no such failure at 8
bits, thus treat this as a check and not as a necessity.

**One warning.** A per-column scale is not a substitute for a rotation on the
gated delta net output projection. MambaQuant measured SmoothQuant at 58.7
against plain RTN at 58.4 on Mamba-2.8b at W8A8, and **worse than RTN** at
790M. Scale migration alone does not work on that tensor.

### 3.7 Permute the channels, where it is free

A channel permutation commutes through every elementwise operation, thus it
folds into the surrounding weights at zero cost at run time. The pipeline
already permutes the MLP intermediate dimension.

At 8 bits with a per-row weight scale and a per-token activation scale, a
permutation buys **nothing on its own**, because both scales take a maximum
over the whole vector and a permutation does not change a maximum. It buys
something only in front of a **block** rotation, where it equalizes the block
masses. PeRQ measured its MassDiff permutation at 37.5 % to 40.5 % of mean
error at 4 bits with block 32, in under two minutes of calibration.

Keep the existing MLP permutation, because it costs nothing and the pipeline
depends on it for the refold path. Do not build a new one.

### 3.8 The permutation must travel through the whole block

If any permutation of a gated delta net channel order is introduced, it must
travel consistently through:

- the columns of `in_proj_qkv` and `in_proj_z`
- the depthwise `conv1d`, which has one weight for each channel
- the gated RMSNorm weight
- the head grouping that `ssm_alpha` and `ssm_beta` index
- the rows of `out_proj`

Quamba2 does exactly this and calls it cluster-aware weight reordering. Get one
of the five wrong and the block computes the wrong thing with no error.

### 3.9 Verify the transform before anything else

The whole stage is function-preserving, thus it has an exact test. The
pipeline already runs it: the transformed F16 GGUF against the original F16
GGUF on WikiText-2, 16 x 512, gives a mean KL of −0.000004 and a top-1
agreement of 99.975 %. Run that test after every new transform. A transform
that fails it is a defect, not a quality trade.

---

## 4. Stage 2: the weights

### 4.1 The format

One signed int8 value for each weight. One f32 scale for each output row.
8.031 bits for each weight at a row length of 2048, against 8.5 for Q8_0.

Symmetric, with no zero point on the weight. The asymmetric weight term is the
one term of the four-term expansion that depends on the data and thus cannot be
precomputed. The Qualcomm white paper states the standard arrangement:
asymmetric activations and symmetric weights. Keep it.

### 4.2 The scale: absmax

Use `d_row = max(|row|) / 127`. Do not build a clipping search. Section 2.2
measures the available gain at 2 % to 4 % and section 12.2 gives the theory and
the production counter-evidence.

### 4.3 The rounding: round to nearest is enough

QuaRot ran our exact ablation, with per-column symmetric weights, a Hadamard,
and a clipping search:

| Llama-2 | INT8 RTN | INT8 GPTQ | INT4 RTN | INT4 GPTQ |
|---|---|---|---|---|
| 7B | 5.50 | **5.50** | 8.37 | 6.10 |
| 13B | 4.90 | **4.90** | 6.09 | 5.40 |
| 70B | 3.33 | **3.33** | 4.14 | 3.79 |

Identical to two decimals at 8 bits, and 2.27 apart at 4 bits. The complete
headroom at 8 bits with a per-channel scale is 0.03 perplexity on the 7B model.

A second, stronger piece of evidence: Red Hat ships a production W8A8 model
whose GPTQ calibration is 256 sequences of **uniformly random token
identifiers**, and it recovers 100.31 % of the baseline. A Hessian built from
random tokens is not the model's Hessian. If the error feedback were doing
work, that model would be broken.

**So do not run the Cholesky rounding on the 8-bit path.** This is the largest
simplification in the recipe. It removes the calibration pass, the Hessian, the
damping parameter and the column ordering from the weight stage.

### 4.4 What to keep of the solver, and how to test it

Keep the Qronos refit, and measure it alone. It is the one part that targets
`‖(X − X̃)W‖`, the input mismatch, which does **not** shrink when the weight
width rises. Its measured gain scales with activation quantization: on
Llama3-8B it removes 0.1 of a 0.4 excess at W4 weight-only and 0.6 of a 2.1
excess at W4A4.

At W8A8 the total excess is small, but our model is a hybrid and not a Llama,
and the refit is one linear solve. Run it, measure it against nothing, and
delete it if it measures nothing.

**Before you trust a null result, confirm the solver ran.** `llm-compressor`
issue 2952 records Hessian inversion failing on 154 of 154 linear modules,
every one falling back to round to nearest, with the run still looking
successful.

### 4.5 Bias correction

With `eps = W̃ − W`, the output mean shifts by `eps·E[x]`. That vector has the
shape of the output, thus it folds into the per-output-channel constant that
the accumulator already adds. Cost at run time: zero.

Nagel et al. measured it at **weights per channel, INT8**, on ImageNet: 70.65
to 70.80 plain, and 70.93 to 71.30 with cross-layer equalization. The total
quantization error falls from 1.07 to 0.39 points. The analytic and the
empirical variants land in the same place, 71.19 against 71.15.

This is the only measured 8-bit per-channel weight-side gain in the whole
literature sweep, and it is free. Take it.

### 4.6 Which classes stay above 8 bits

Keep at F32: the recurrence controls and the norms. That is `ssm_a`,
`ssm_conv1d.weight`, `ssm_dt.bias`, `ssm_norm.weight`, `attn_q_norm.weight`,
`attn_k_norm.weight`, `attn_norm.weight` and `post_attention_norm.weight`.
Together they are 4.9 M weights on the 4B, which is 18.7 MiB. The cost of
keeping them exact is under 0.5 % of the file.

`ssm_alpha.weight` and `ssm_beta.weight` are the two to re-examine. They are
projections into the **log-space** path, where softplus and the exponential
compress the error. Minima measured the decay gate projection at 2.1 % of
output error and the write gate projection at 2.6 %, against 12.7 % for
`out_proj`, and put both at 4 bits with no measurable cost. QUASAR reached the
same conclusion by quantization-aware training on the same architecture family.
Our plan keeps them at F32 on the unsloth recommendation, which is a 4-bit
recommendation. Measure the move with `quant/attribute.py` before you trust it.

`ssm_out` needs no special treatment at 8 bits with the rotation of section
3.3. It needed it at 4 bits, and at Q8_0 with no rotation the 4B still showed a
maximum KL of 1.56 from this class alone. The rotation is the fix, not the bit
width.

---

## 5. Stage 3: the activations

### 5.1 The format

One unsigned 8-bit value for each activation element. One scale and one zero
point for each token, computed at run time from the row.

Section 2.3 measures this format at 41.9 to 42.2 dB on every site that R1
reaches, and at 24.7 to 30.6 dB on the three gated sites before the transform
of section 3.3.

### 5.2 The accumulator epilogue

With an asymmetric activation and a symmetric weight the product expands into
two terms:

    sum_k (a_k - z) * w_k  =  sum_k a_k w_k  -  z * sum_k w_k

The first term is the int32 accumulation the engine does. The second term is
one scalar for each token times one precomputed constant for each output
channel. Precompute `colsum(W)` once, at export time, and store one f32 vector
for each linear.

**Plan this deliberately.** With a per-tensor zero point the correction is a
scalar and folds away. With a **per-token** zero point it is a rank-one outer
product `z_t ⊗ colsum(W)`, which must be materialized in the epilogue. The
cost is `M·N` additions against `M·N·K` multiply-accumulate operations, thus
under 0.1 % at K = 2048. It is cheap, but it does not appear for free. vLLM
declined to implement it and supports dynamic per-token **symmetric** only.

Note one consequence of the rotation. After a Hadamard the activation becomes
close to zero-mean and symmetric, thus the zero point sits near the middle of
the range, which is correct and loses nothing. But `colsum(W)` becomes
non-trivial for rows where it was near zero before. Do a check of that term.

### 5.3 The range: absmax, with one exception

Use absmax of the token for the sites that R1 or a folded rotation reaches.
At 42 dB there is nothing to recover and a clipping search adds a reduction
pass to a machine that dislikes reductions.

Consider a percentile clip at the **two gated sites** only, and only if the
transform of section 3.3 leaves them behind. Quamba measured a percentile clip
at 2.6 points of zero-shot average at W8A8 on Mamba-2.8B, and measured that
the correct percentile moves with model size: 99.999 at 130M and 370M, 99.99
at 1.4B, **99.9 at 2.8B**. A fixed value costs about 1 point at 2.8B. Sweep
it, do not fix it.

InfoQuant swept the clip ratio directly against absmax on LLaMA-2-7B and found
the optimum near **0.90** of absmax, with a sharp cliff at 0.80: 1.00 gives
7.21, 0.95 gives 7.06, 0.90 gives 6.98, 0.85 gives 7.00, 0.80 gives 8.02.
That is a W4A4 measurement, thus take the shape and not the magnitude.

### 5.4 The per-channel bias, BASE-Q

A per-token asymmetric scale fits one zero point for each token. It does
**not** remove the spread of per-channel means inside that token. A rotation
makes the distribution Gaussian but does not center it. That residual term is
what a per-channel bias removes.

Subtract a calibrated per-channel bias `b` before the activation quantizer and
add `W·b` back after the projection. The correction has the shape of the
output, thus it lands in the same accumulator constant as the zero-point term
of section 5.2. Cost: one f32 vector for each linear, zero online matmul.

BASE-Q measured the bias term alone at −0.21 perplexity on Llama-3-8B and
−40.79 on Qwen2.5-3B, and measured the mean-spread term at "up to 85 % layer
rounding error in Qwen2.5-3B". The Qwen result is the relevant one for us, and
it is also the least reproducible: the paper has no code and the large number
is the rescue of a collapse case. Build it anyway, because it is one vector
and it lands in a slot that already exists.

### 5.5 Static against dynamic

Our engine gives a per-token dynamic scale, thus this section is a note and
not a decision. But the numbers are worth knowing, because a dynamic scale
costs a reduction for each row and each layer.

The measured difference between a per-token dynamic scale and a per-tensor
static scale, after a transform, is small:

| Comparison | Model | Difference |
|---|---|---|
| SmoothQuant O1 against O3 | OPT-175B | 0.1 points |
| SmoothQuant O1 against O3 | BLOOM-176B | 0.9 points |
| SmoothQuant O1 against O3 | GLM-130B | 0.9 points |
| PrefixQuant O1 against O2, W4A8KV4 | Llama-2-7B | 0.01 perplexity |

PrefixQuant measured static as **faster and more accurate** than per-tensor
dynamic at W8A8: Llama-2-7B 5.48 against 5.75, with FP16 at 5.47. Prefill 178
ms static against 183 ms dynamic.

Two warnings if static is ever considered. Quant.npu measured a **15.61 %**
accuracy loss when a dynamically optimized model is converted to static, and
SpinQuant converted the same way reached a perplexity of 1187.69. Optimize for
static from the first step or do not do it at all. And a static scale is a
calibration artifact that drifts, which a dynamic scale cannot.

**Recommendation: keep per-token dynamic.** It is free on this engine, it is
worth 0 to 1 point, and it is immune to calibration drift. Revisit only if the
row reduction measures as a real cost in the kernel.

### 5.6 Prefixed outlier tokens

The token-wise half of the outlier problem is separate from the channel-wise
half, and a rotation does not fix it. A few high-frequency tokens, typically
`[BOS]`, `.` and `\n`, carry activations three to four orders of magnitude
above the median. Massive Activations measured LLaMA2-7B at a top value of
2622.0 against a median of 0.2, with fewer than ten such values in 40000, at
fixed feature dimensions and fixed token positions.

PrefixQuant puts those tokens at the start of every sequence and computes their
K and V once offline in full precision, thus they never enter the quantized
forward pass. Detection takes 0.2 minutes for Llama-3-8B. Llama-3-8B needs one
token, Llama-2-7B needs three, Mistral-v0.3-7B needs four.

For us this is cheap insurance and it composes with everything. The prefix is
the same for every request, thus it is a one-time constant. Our calibration
harness already collects the statistics necessary to detect it. **But note
that our own per-token numbers do not show a crisis**: the max-to-RMS ratio
after R1 is 3.4 to 3.5, and a token-wise outlier would show there. Measure
before you build.

### 5.7 What must not go to 8 bits

- **The recurrent state.** Section 6.1.
- **The materialized decay.** Section 6.2.
- **The output of the final norm before the head**, if the head is tied. The
  head reads it through the dense map M, and the KL of the logits is the
  metric. Keep the map in F16, as the pipeline already does.

---

## 6. Stage 4: the gated delta net block

24 of 32 layers on the 4B, and 23.4 % of the decode bytes plus the state. This
is the half of the model that the literature does not cover, and the half
where a wrong decision is expensive.

### 6.1 The recurrent state stays in high precision. Do not quantize it.

This is a decision, not a trade, and two measurements make it.

**The decay spectrum of our model.** The per-head decay at the natural centre,
`alpha = exp(-exp(A_log) * softplus(dt_bias))`, over the 288 heads of the 2B:

| Quantile | p10 | p50 | p90 | p99 |
|---|---|---|---|---|
| alpha | 0.462 | 0.962 | **0.999** | 0.9997 |

The error retention factor `1/(1-alpha^2)` has a median of 13 and a 90th
percentile of **543**. **11.5 % of our heads have alpha above 0.999.**

DAMP measured Qwen3.6-35B at p10 0.376, p50 0.967, p90 0.9996. We are the same
distribution. That is the distribution in which DAMP measured a plain INT8
state as AIME 85.46 to 18.48 and LiveCodeBench 86.95 to 32.23, while MMLU-Pro
moved only 84.66 to 83.88.

**The byte budget.** The state read and write is 100.7 MB of 4579.8 MB, which
is **2.2 %** of a decode token at 8 bits. An 8-bit state saves at most 1.7 %
of the bandwidth.

1.7 % of bandwidth against a measured 67-point collapse on reasoning, invisible
to every metric we run, is not a trade. **Keep the state in F32 or F16.**

This deletes a large body of work from the plan: DAMP, `carrykernel`,
stochastic rounding of the state, Q-Mamba decoupled scale quantization, and the
whole error-feedback thread. Each is a real technique and none of them is worth
1.7 %.

Record what would be necessary if the decision ever reverses. The rounding
rule, not the bit width, is the variable. Nemotron 3 Ultra measured INT8 block
128 with round to nearest at −1.42 % and the same format with **stochastic
rounding** at +0.15 %. Pimba took Mamba-2 from perplexity 62 to 11.9 with
stochastic rounding. `carrykernel`, which is one unreviewed repository with
n = 100 on our exact model, restored Qwen3.5-4B GSM8K from 41 % to 81 % with
error feedback. Try the rounding rule before the bit width, and never use an
FP8 format, because three sources agree that mantissa bits beat exponent range
for an accumulator.

### 6.2 The gates: quantize the pre-activation, never the decay

The parameterization is `g = -exp(A_log) * softplus(a + dt_bias)`,
`alpha = exp(g)`, `beta = sigmoid(b)`. Softplus and the exponential **compress**
an error on the pre-activation and **expand** an error on the result.

Both sides are measured:

- Minima quantized the two gate projections to 4 bits and measured an 11.0 %
  GEMM error becoming a 2.1 % output error, and 8.5 % becoming 2.6 %. They are
  the **least** sensitive of the five projections of the block.
- Minima measured 0.1 % of multiplicative noise placed **directly on alpha** as
  a 22 % state error, and 1 % as 42.8 %. The same noise on beta gives 0.4 %.
- QMamba measured the discretized decay at 4 bits as the worst activation of a
  Mamba block, at −66.8 Top-1, and fixed it with a log-domain quantizer
  `q = clip(round(-log2(1 - A)), 0, 2^b - 1)`.

Thus: `ssm_alpha.weight` and `ssm_beta.weight` are ordinary projections and can
take 8 bits or less. `ssm_a` (which is `A_log`) and `ssm_dt.bias` stay F32.
The runtime alpha never enters an integer format.

One algebra rule follows. **Never fold a scale past the decay exponential as a
multiplication.** MambaQuant gives the correct pattern: a smoothing factor `s`
that must cross `exp` becomes `-ln(s)` added to the pre-activation, and applied
to the first token only, as `addcmul(-ln(s), delta(1), A)`.

### 6.3 The depthwise causal convolution

Leave it at F32 or F16.

The literature has no isolated measurement of this tensor anywhere. Five papers
touch it and none ablates it. Minima excludes it from quantization. SSDi8 uses
the unmodified CUDA operator. Quamba and Quamba2 quantize it to int8 with SiLU
fused into the kernel and give no number.

Our own data is better evidence than any of them: the unsloth per-tensor sweep
puts `ssm_conv1d` at a KL of 0. And SSDi8 measured the convolution at 2.526 ms
against 41.297 ms for the input projection at batch 32, which is about 1.5 % of
the block.

There is nothing to win and there is bookkeeping to lose, because the
convolution is depthwise and thus carries one weight for each channel, which
must follow any channel permutation. Leave it.

### 6.4 The gated RMSNorm

Our block applies `y = RMSNorm(v) * silu(z)` elementwise before `out_proj`.
This is the gate that blocks a folded rotation at that site, and it is the
reason section 3.3 exists.

No published work measures a **gated** norm under quantization. Every source
treats the block norm as a plain RMSNorm. This is a genuine gap and we are
the ones who will find the answer.

Two treatments exist for the plain case and they disagree. Quamba keeps the
norm weights unquantized. SSDi8 migrates the norm weight into the in-projection
to remove the norm-induced outliers, and measures the norm latency falling from
3.086 ms to 2.291 ms at batch 32. We already fold norms, thus the SSDi8 route
costs us nothing.

### 6.5 The state matmuls

The recurrence itself has two matrix products: the write `beta * k v^T` into
the state, and the read `S^T q`. Both contract the state dimension.

**Do not quantize along the state dimension.** SSDi8 states the rule and
measures the price of breaking it: on Mamba-2 8B, per-token quantization of X
and the state gives 11.97 perplexity against 7.42 for the head-and-channel
axis. The state dimension participates directly in the following matrix
product, where the error cannot be recovered.

If the read and write ever move to integer, SSDi8 is the template. It
reformulates `State = X x (B ⊙ LUT)` as `State_INT32 = Q(LUT ⊙ X) x Q(B)`,
which is valid because the decay table lives on the chunk axis that X and B
share. It converts the int32 state to int8 in registers and never writes an
F16 state to memory. It makes the state update a bit shift by fixing the decay
at `S = 2^7`. Its W8A8 results are within 0.6 points of FP16 on Mamba-2 at
1.3B, 2.7B and 8B.

Given section 6.1, this is recorded and not scheduled.

### 6.6 The output projection is the one to protect

Three independent measurements name `ssm_out` the most sensitive tensor of
this architecture:

- Our own 4B Q8_0 attribution: F16 on `ssm_out` alone takes the 99.9 % KL from
  0.175 to 0.0369 and the maximum from 1.56 to 0.553, for 225 MiB. That is
  3.5 times the efficiency of the embedding and 14 times the MLP, for each
  megabyte.
- Our own 2B per-class table: 36 MiB removes 15 % of the mean KL and 45 % of
  the maximum KL. It is the one class that moves the maximum.
- Minima per-projection replay: 12.7 % of output error, against 10.4 % for
  qkv and 9.9 % for z.
- The unsloth per-tensor sweep of the 35B-A3B ranks it first.

At 8 bits with the rotation of section 3.3 this class needs no extra bits.
Verify that with `quant/attribute.py` and be ready to be wrong.

---

## 7. Stage 5: the multi-token prediction block

### 7.1 What we know

The MTP block is `blk.24` of the 2B and `blk.32` of the 4B. llama.cpp loads it
only for `--spec-type draft-mtp`, thus it costs nothing when speculation is
off. The transform already puts it into the rotated basis with two dense F16
maps of d x d, and the pipeline verifies the transform by draft acceptance:
the original F16 file and the transformed file give identical counts, 278
drafted and 164 accepted.

Our measured acceptance is **prompt-dependent**: 42 % on a free-form answer
and 89 % on a formulaic thinking block, on the 4B. Row 13 of the results shows
the eight matrices of the block at round-to-nearest Q8_0 giving the same text
as F16 on the test prompt, with 58.51 % acceptance against 58.99 %.

### 7.2 What the literature does not know

**There is no published measurement of a quantized draft block against the
acceptance rate. None.** The whole field rests on one sentence in QSpec, which
says that GPTQ on an EAGLE draft "resulted in substantial degradation of the
acceptance rate" and gives no number. SpecMQuant cites that sentence as its
reason to keep its draft at FP16, and measures only the target, where it finds
"minimal impact on average accepted length" at W8A8 and W4A16.

### 7.3 Why the risk is structurally worse than a task score suggests

The acceptance rate is a function of the divergence between the draft
distribution and the target distribution. That is second order in the
perturbation. A task score is often first order and forgiving. Two further
points are specific to us. The draft reads the same recurrent state as the
target, thus a state error is common-mode and partly cancels in the accept
test. A draft-block **weight** error is not common-mode at all.

### 7.4 The recipe

Put the MTP block at the same 8 bits as the trunk, with the same transform.
Then **measure the acceptance rate directly, split by prompt class**, before
and after. Do not infer it from KL and do not infer it from a task score.

Record the economics so the measurement can be judged. A draft-3 step on the
4B costs about 2.1 plain tokens and yields 2.27, thus the margin is thin and a
few points of acceptance decide whether speculation pays at all.

---

## 8. Stage 6: the vision tower

Leave the projector at F16.

The projector is a separate GGUF and it runs once for each image, not once for
each token. It does not appear in the decode byte budget. The pipeline already
rotates it to match the rotated trunk, in `quant/rotate_mmproj.py`.

No published work quantizes a vision tower attached to a linear-attention
trunk. There is nothing to copy and nothing to gain. Revisit only if image
turns become a latency problem.

---

## 9. Evaluation: what to measure, and what will lie to you

Our metric today is the per-token KL against the F16 logits on WikiText-2, 16
chunks of 512 tokens, plus top-1 agreement. It is a good metric and it is blind
to three of the failure modes in this document.

### 9.1 Keep

- **Mean KL** against the CUDA F16 base. The primary number.
- **The 99.9 % point and the maximum KL.** The maximum is the number that
  `ssm_out` moves, and the 4B Q8_0 file has a maximum of 1.56 where its mean is
  0.00202. A mean-only report hides a class failure.
- **Top-1 agreement.**
- **The F16 floors.** CPU 0.00020, NPU 0.00019, Adreno 0.00065. A result inside
  the floor is not a result.

### 9.2 Add, because the current metric cannot see these

**A reasoning-length metric.** TASA measured the Kendall correlation between
the perplexity-ranked layers and the reasoning-critical layers at **−0.08 with
p = 0.53** on LLaMA-3-8B, and named it the Perplexity Illusion. Meta measured
aggressive PTQ inflating the chain of thought by **4.5 times** on MATH-500,
with the damage concentrated on the hesitation tokens "Wait", "But",
"Alternatively". The cheapest version is the mean generated token count on a
fixed prompt set against F16. A model that grows the chain by more than about
20 % is damaged even when the KL looks correct.

**A long generative reasoning task.** DAMP's own tables make the case: RULER
from 4K to 128K separates the methods by at most **0.13 points** where AIME
separates them by **67**. Retrieval is blind to state error. Do not score the
recurrent path on needle-in-a-haystack.

**The MTP acceptance rate, split by prompt class.** Section 7.

**A tool-call test and IFEval, because they break first.** Two independent
measurements say the aggregate score hides this. Meta measured BFCL V2 tool
calling falling **67.0 to 63.5** and QUASAR measured IFEval falling **74.4 to
68.5**, both while the headline numbers looked acceptable. A third paper,
"Flat Score, Amplified Failures", measures 4-bit weights amplifying agent
failure modes by **up to 2.5 times** on a two-agent benchmark while the score
itself reads as lossless. https://arxiv.org/abs/2607.27275

Our product is a chat application with tool use. This is the metric that
matches the product, and we do not run it today.

**Task accuracy on a small chat and math set.** Our KL is measured on
WikiText-2 and our product is a chat application. COLA measured a whole-spread
of 0.45 perplexity and 1 to 2 accuracy points across calibration sources, and
measured that domain-matched calibration moves math by +5.92 points and code
by −4.89 on the same model.

### 9.3 Three defects to guard against explicitly

**The fused-GEMM scale mismatch.** Minima found that vLLM fuses `qkv+z` and
`b+a` into shared GEMMs and takes the maximum of the per-module scales. The
paired scales differed by 1.82x and 2.75x in **every one of 48 layers**. AIME
fell from 86.7 to 80.8 and the perplexity at 32K **improved** to 6.86 against
a true 10.84, because a broken forget gate makes the state hold everything.
Our export fuses GDN projections. Do a check of the paired scale ratios and do
not trust perplexity to catch it.

**The silent solver fallback.** `llm-compressor` issue 2952 records Hessian
inversion failing on 154 of 154 modules with the run still looking successful.
If a solver measures as worth nothing, first confirm that it ran.

**The vendor toolchain that ignores scales.** "Is INT8 Portable?" held an ONNX
artifact and its scales fixed across seven accelerators and found the Hexagon
HTP **silently ignores external QDQ scales**: the file compiles, profiles and
runs with no error while top-1 falls from 0.753 to 0.005. We bypass QNN through
the llama.cpp Hexagon backend, thus this may not touch us. Gate the release on
an on-device accuracy number and never on a clean build.

### 9.4 The calibration set

Keep C4 English at 128 sequences of 2048 tokens. The evidence says this is
close to the right answer and that curation is worth little:

- COLA measured the whole spread across WikiText, C4, SlimPajama,
  self-generated and curated sources at **0.45 perplexity** and about 2 points
  of commonsense accuracy, for AWQ 4-bit on LLaMA3-8B.
- COLA found "diminishing returns beyond 64 to 128 samples".
- AMD measured that a method with one parameter for each channel reaches
  near-asymptotic quality from **a single sequence of 2048 tokens**, while
  GPTQ with absmax needs 64 or more. Our 8-bit path fits one parameter for
  each row, thus it is in the first regime.
- RSQ found "no clear trend that longer calibration sequences help long-context
  tasks".
- TASA measured that **pure** task data is worse than a mix: on LLaMA-3-8B at
  3.5 bits, a mix ratio of 0.00 gives 44.6 GSM8K, 0.50 gives 46.2 and 1.00
  gives 33.7. Pure task data degenerates the Hessian eigenspectrum.

One change is worth an hour: add a chat-like and long-context slice, because
our product is a chat application and the unsloth caveat applies to us as well.
Their calibration is long-context chat and tool calls, and they note that
WikiText-512 metrics favour quants whose calibration is WikiText-like.

---

## 10. The 4-bit question

The owner's decision is 8 bits. This section records why, and what would have
to be true for 4 bits to come back. The user asked directly whether any 4-bit
method can close the gap with 8 bits, so the answer is here in full.

One hardware fact arrived after this section, and it removes the kernel
objection without touching the accuracy argument. Refer to section 1.5: the
matrix engine has a native 4-bit weight load. Thus a 4-bit MLP would pay no HVX
unpack and no extra issues, and it would halve the bytes of the class that is
52.5 % of the token. The argument below is about accuracy alone, and it stands.

### 10.1 The measurement that decides it, from our own table

Take two rows of `analysis/quant-results.md` on the 2B:

| File | Bytes | Mean KL |
|---|---|---|
| Row g: Q4_0 bulk with `ssm_out`, `attn_q`, `attn_output`, `ffn_down` and `attn_qkv` at Q8_0 | 2,083,482,624 | **0.015025** |
| Plain Q8_0, round to nearest, no rotation, no calibration | 2,137,156,512 | **0.00115** |

The plain 8-bit file is **2.6 % larger and 13 times better**. It uses no
rotation, no calibration, no solver and no block optimization. The best mixed
4-and-8 file we have built, with all of that machinery, is dominated by the
crudest possible 8-bit file at the same size.

**That is the whole argument.** The 4-and-8 Pareto front does not exist at this
model size. It is a set of points under the 8-bit point.

### 10.2 What 4 bits would have to reach

To be on a new front, a 4-bit file must land near **0.003 of mean KL at about
1.2 GiB** for the 2B. Our best 4-bit file today is 0.0295 at 1.62 GiB, and the
tied-head variant is 0.0311 at 1.13 GiB. The distance is a factor of ten.

**The published curve says that target is out of reach, and shows why.** Three
external reference points, all mean KL against the unquantized model:

The Unsloth ladder on Qwen3.5-35B-A3B, measured with the same tool and the
same protocol we use, `llama-perplexity --kl-divergence` against F16:

| Quant | disk | mean KL |
|---|---|---|
| `Q8_K_XL` | 36.04 GB | **0.0026** |
| `Q6_K_XL` | 28.22 GB | 0.0041 |
| `Q4_K_M`, AesSedai | 20.62 GB | 0.0096 |
| `Q4_K_XL` | 19.17 GB | 0.0137 |
| `Q4_K_M`, Unsloth | 18.49 GB | 0.0192 |
| `IQ4_XS` | 16.40 GB | 0.0235 |

And two file-size-matched comparisons from a different harness, gemma-4-12B:
EXL3 at 4.006 bits measures 0.02479 where `Q5_K_M` at the same 7.8 GiB
measures **0.01719**, and EXL3 at 5.00 measures 0.00946 where `Q6_K` at the
same 9.1 GiB measures 0.00918.

Three shapes matter.

**Even 8 bits is not zero.** The best public 8-bit file measures 0.0026, which
is 13 times our own F16 floor of 2e-4.

**The decay is 1.7 to 2 times for each bit, not the 4 times that high-rate
theory predicts.** That gap is the signature of a small set of hard tensors
that dominate the error, which is exactly what our own `ssm_out` attribution
found. It is also why allocation beats every grid change: the same nominal
format moves KL by **2 times** on allocation alone, 0.0192 at 18.49 GB against
0.0096 at 20.62 GB.

**The curve is flat below about 4.25 bits and steep from 5 upward.** Buying
quality with bits works above 4.5 and stops working below it.

A realistic 4-bit target for our size is **0.005 to 0.008**, not 0.003, and
nothing published reaches even that on a plain integer grid.

The same tables give the one PTQ lever that is clearly worth its bytes:
**per-tensor allocation**. The same `Q4_K_M` family at 18.49 GB measures KLD
0.0192 and at 20.62 GB measures **0.0096**. Half the KL for 11 % more disk,
with no training. Unsloth's own March 2026 revision took `UD-Q4_K_XL` from
5.894 to 2.877 maximum KLD for 7.8 % more disk. Our `quant/attribute.py`
already does this measurement and our results table already ranks the classes.

### 10.3 What post-training quantization can do, and it is not enough

Everything in the PTQ literature has been tried or costed:

| Method | Measured on our model or published | Verdict |
|---|---|---|
| GPTQ | 0.0373 | the baseline |
| Qronos refit plus folded column scales plus MLP permutation | 0.0299 | −20 % |
| Block optimization with frozen latent weights | 0.0295 | −1 % more |
| Trellis, QTIP style, L = 12 | 22.1 dB against IQ4_NL 22.5 dB on the grid test | loses |
| Per-matrix Lloyd-Max codebook | 22.3 dB against 22.5 | loses |
| Low-rank correction | LRC's own weight-only result: "low-rank terms do not provide any additional improvement" | redundant |
| Per-class 8-bit guards | 0.0295 to 0.015025 for 324 MiB | real, and dominated by 10.1 |
| Learned rotation | OptRot at 8 % of KL over a random Hadamard at W4A8 | small |
| Fisher reweighting, GuidedQuant | 0.01 perplexity at 4 bits on QTIP | noise |

Sum the honest best case. Start at 0.0295, take 20 % from a better rotation
and a better objective, take 50 % from per-class guards, and land near 0.012 at
about 1.9 GiB. That is still ten times the 8-bit file at a smaller size.

**Post-training quantization cannot close this at 2B.** The reason is not the
method. It is the model size. Quantization damage scales inversely with
parameter count, and every published 4-bit success is on a model five to
twenty times larger: Minima at 27B, unsloth's task-lossless result at 9B,
QUASAR at 27B.

### 10.4 What quantization-aware training can do, and it is enough

This is the one path with evidence on our architecture.

**QUASAR** trains the same model family, Qwen3.8-27B with 48 gated delta net
layers, to NVFP4 W4A4 on **496 of 496** linear tensors, attention and gated
delta net included. One epoch of distillation against the frozen BF16 teacher,
global batch 32, learning rate 1e-6, **2446 steps**. Result: GPQA-Diamond 90.91
against BF16 91.41, AIME'26 100.0 for both, 19.7 GB against 55.6 GB. Its
general result is 10 % or more lower held-out KL at 3 and 4 bits against the
best PTQ, and 29 % at 2 bits.
https://arxiv.org/abs/2608.13966

**EfficientQAT** (ACL 2025 Main) runs the same two stages our pipeline already
has in a smaller form: **Block-AP**, which trains all parameters of a block
with the quantizer in the loop, then **E2E-QP**, which tunes only the step
sizes across the whole network. Its headline is a **2-bit** Llama-2-70B on a
single A100-80GB in **41 hours**, at less than 3 points of degradation, 69.48
against 72.41. The two-stage shape is the important part for us: our
`blockopt.py` is Block-AP, and E2E-QP is the stage we do not have.
https://arxiv.org/abs/2407.11062

Read the two papers together. EfficientQAT says the training loop is affordable
on one GPU even at 70B. QUASAR says the loop closes the 4-bit gap on our
architecture. Neither needs the cluster that the owner's note assumed.

**The token budget is the whole difference, and we can compute it.** QUASAR
used 2446 steps at a global batch of 32. At a sequence length of 2048 that is
**160 M tokens**, and at 4096 it is 320 M. Our block optimization uses
128 x 2048 = **262 144 tokens**. The ratio is 600 to 1200 times.

Our own results already show this exact signature. Row 5 of the results table
trains all layer parameters freely for 8 epochs on 128 x 2048 and lands at
0.0412, **worse** than the 0.0325 solver start. Row 8 frees the latent weights
at 3e-6 and lands at 0.0314, worse than the 0.0295 of the frozen regime. The
pipeline note records the diagnosis: "the per-layer training loss halves, the
held-out error grows at every layer". **That is not a defect of the optimizer.
It is data hunger, and it is the signal that the regime is right and the token
count is 1000 times too small.**

**The cost, computed for our model.** A 2B student with a forward and a
backward pass is about `6·N·T` floating-point operations, and the frozen
teacher forward is `2·N·T`. At N = 2e9 and T = 2e8 tokens that is about
`3.2e18` operations. An RTX 6000 Ada at 91 TFLOPS bf16, at a realistic 35 %
utilization, gives about 32 TFLOPS, thus about **28 hours** for the 2B and two
to three days for the 4B.

**One day on the box is the price of a 4-bit model that may match 8 bits on
tasks.** The data is not the constraint: 160 M tokens of C4 is a trivial
download. The constraint is GPU time and a training loop, and the block
optimizer already holds most of the loop.

### 10.4b The honest limit of quantization-aware training, and it is not KL

The paragraph above states the task result. The KL result is different and it
must not be over-sold. Three numbers set the limit:

- QUASAR measures held-out forward KL against the BF16 teacher on **Qwen3.5-9B
  at NVFP4**: standard quantization-aware training 0.009, QUASAR **0.006**.
- On INT4 group 128 with Llama-3.1-8B, GPTQ gives 0.015 and full
  quantization-aware training gives **0.014**. That is 7 %, not a factor.
  The same comparison on Qwen3-4B gives 41 %.
- Our 2B Q8_0 file, round to nearest, no rotation, no calibration, is
  **0.00115**.

Read those together. A 4-bit quantization-aware training run on a **9B** model
lands at five times the KL of an 8-bit round-to-nearest file on our **2B**.
Training does not close the KL gap between 4 bits and 8 bits. It closes the
**task** gap.

That is not a contradiction of section 10.4, because task accuracy is what the
product sells. The unsloth harness measures UD-Q4_K_XL on the 9B at 71.4
against an original of 71.0 while its mean KL is 0.0137. A 4-bit model can be
task-lossless at a KL that looks bad.

**But treat that with suspicion here**, for the reason section 9.2 gives. The
task suites that stay flat are multiple choice. The suites that break first are
instruction following and tool use, which is what a chat product does. QUASAR
measures IFEval falling 74.4 to 68.5 and Meta measures BFCL V2 tool calling
falling 67.0 to 63.5, both with the mean KL looking acceptable. If the 4-bit
run happens, gate it on IFEval and a tool-call test, not on KL and not on MMLU.

**A free experiment exists, and it should run before any training does.**
QUASAR published a quantization-aware checkpoint of **our exact model**:
`huggingface.co/QUASAR-QAT/Qwen3.5-4B-QUASAR-Q4_0-GGUF`, trained directly on
the `Q4_0` lattice. Their own comparison, on their harness:

| Build | disk | mean KL | median KL |
|---|---|---|---|
| QUASAR `Q4_0`, trained on the lattice | 2.70 GB | **0.235** | **0.0053** |
| plain `Q4_0` rounding | 2.61 GB | 0.372 | 0.0108 |
| a training run then re-rounded to `Q4_0` | 2.54 GB | 0.363 | 0.0147 |

Their absolute numbers are not comparable with ours, because the harness and
the sequence length differ. The **ratio** is the useful part: training on the
lattice halves the median KL against plain rounding. And the file is a GGUF of
the 4B, thus `quant.run eval` can score it against our own F16 base in an
afternoon, with no training and no box time. **Do that before committing a
day of GPU time to reproduce it.**

**The size of the training gain, on a Qwen model of our size.** Asymmetric
INT4 at group 128, held-out forward KL against the teacher:

| Method | Qwen3-4B-Thinking | Llama-3.1-8B-Instruct |
|---|---|---|
| round to nearest | 0.108 | 0.044 |
| GPTQ | 0.027 | 0.015 |
| standard quantization-aware training | 0.019 | 0.014 |
| BitDistiller | 0.018 | 0.013 |
| QUASAR | **0.016** | **0.012** |

Training over GPTQ is worth **41 %** on the Qwen 4B and 20 % on the Llama 8B.
That is a real gain and it is not a factor of ten. Our 8-bit file will be near
0.0025. Nothing in that column comes close.

**Correct the cost estimate of section 10.4 to a range.** The published costs
span two orders of magnitude, because they buy different things:

| Run | Cost |
|---|---|
| YAQA, reduced sketch, post-training only | about 1 GPU-hour |
| EfficientQAT, 2-bit Llama-2-70B | 41 hours on one A100 |
| QUASAR, 4096 steps at 4096 context, about 246 M tokens | 10 to 50 H100-hours for a 4B to 9B |
| Meta Llama 3.2 QAT plus LoRA, 3B | **1600 GPU-hours** |

Our own arithmetic in section 10.4 gives about 28 hours for the 2B, which sits
inside the QUASAR band. Budget one to three days on the box, not one day, and
treat the Meta figure as the ceiling of what a full recipe can cost.

**One published run argues the other way, and it deserves a hearing.**
ParetoQ (Meta, NeurIPS 2025) concludes that 4 bits is **not** Pareto optimal,
and its own 4-bit LLaMA-8B measures 6.8 perplexity against 6.42 for KronQ,
which is pure post-training quantization. A full training run losing to a
one-hour post-training run is the clearest evidence that training is not
automatically better at 4 bits. https://arxiv.org/abs/2502.02631

**Two traps, both recorded before anyone starts.**

**Train on the lattice the kernel decodes, and export the trained integer
codes.** The third row of the table above is the proof: a training run that is
re-quantized afterwards measured **0.363 mean KL, worse than plain
post-training quantization at 0.372**. A re-rounded run is wasted compute.

**Do not compare against a vendor quantization-aware checkpoint.** Those
optimize KL against their own training parent, not against the original BF16.
Measured on `google/gemma-3-12b-it-qat-q4_0-gguf`: the Google checkpoint sits
at **KL 0.089** from the original BF16 while YAQA, a pure post-training method
at 4 bits, sits at **0.058**. If low KL against the original is the goal, run
the training with the original as the teacher.

One correction to section 10.3 while we are here. YAQA is cheaper than the
first estimate in our sweep suggested. Its reduced setting is about **one
GPU-hour** and the authors state it still reaches the top results. On plain
INT4 with Llama-3.1-8B-Instruct it measures LDLQ 0.038, DiscQuant 0.061 and
**YAQA-A 0.028** of KL, with no inference change at all. It is worth one
afternoon on the 4-bit path, and nothing on the 8-bit path.
https://arxiv.org/abs/2505.22988

### 10.5 The format that would carry it

A 4-bit model must still run on the per-channel int8 kernel, because that is
the only fast path. The format that does this is **LPBQ**, Qualcomm's own
low-power blockwise quantization, shipped in AIMET and in the ONNX Runtime QNN
provider.

The mechanism: write a block scale as a per-channel float scale times a small
integer, so that the 4-bit value times that integer is an int8 value with one
scale for each row. With a 4-bit value in −8 to 7 and a multiplier in 1 to 15,
the product stays inside −127 to 127.

    w  =  s_row  x  ( m_block  x  q4 )        with  m_block x q4  in  int8

Properties:

- **4.125 bits for each weight.** 32 nibbles is 16 bytes, plus 4 bits for the
  block multiplier, which is 16.5 bytes for 32 weights. Q4_0 is 18.
- **Block-32 adaptivity is preserved**, which is what pure per-channel 4-bit
  throws away. The Quant.npu tables show why that matters: ExecuTorch W4A8
  per-channel collapses to 26.39 perplexity on Llama-3.2-3B where per-block
  W4A16 gives 18.19 and FP32 gives 16.85.
- **It runs on the int8 kernel unchanged.** The expand step multiplies a nibble
  by a small integer instead of by an F16 scale, which is cheaper than what the
  kernel does today.

Our existing scale search maps onto it directly. `grid.scale_search` already
evaluates 41 candidate scales for each block against a weighted error. Change
the candidate set from a continuous range to `s_row x {1..15}` and fit `s_row`
so the ladder covers the block scale distribution. That is a small change to
`quant/grid.py`.

**The one alternative worth costing against LPBQ is GPTVQ**, because it is the
only method in the whole field designed for exactly our constraint. Qualcomm
AI Research built it around a mobile lookup instruction that maps a **6-bit
index to an 8-bit value**, which is the same shape as the HVX byte lookup of
section 1.5.

The deployed setting is 2-D vector quantization, 6-bit indices, **3 bits for
each weight**, group 8192, and an **INT8 codebook**. The rounding is
GPTQ-style, but each step quantizes a 2-D vector against a small codebook fit
by Hessian-weighted expectation maximization. There is **no Hadamard transform
at all**.

- Decode: **one lookup for each weight**, fully parallel, output is a signed
  8-bit integer **by construction**, table 128 bytes for each 8192 weights.
- Quality, Llama-2-7B at 2048 context, FP16 5.47: at 4.125 bits GPTVQ-2D 5.59
  against AWQ 5.62 and GPTQ 5.61, which their own 10-seed study puts inside
  ±0.01 and thus is noise. **At 3.125 bits GPTVQ-2D 5.83 against GPTQ 6.29.**
- On a Snapdragon X Elite, Llama-3-8B: their INT4 at 4.33 GB gives 23.81
  tokens for each second, and **VQ 2D at 3.125 bits and 3.52 GB gives 26.15**.
- Encoder: 30 to 60 minutes on one H100 for a 7B model, thus 20 to 40 minutes
  for ours on one 24 GB card.

https://arxiv.org/abs/2402.15319 · https://github.com/Qualcomm-AI-research/gptvq

The warning is the usual one. The inference kernels were never released, the
repository has 2 commits, and issue 4 disputes the bits-for-each-value
arithmetic with no author reply since 2024.

**How to choose between the two.** LPBQ is a scale trick on a format we
already produce, thus it is days of work and it keeps the block-32 adaptivity
we have measured. GPTVQ is a different grid with a different encoder, thus it
is weeks, and it buys its advantage below 4 bits where LPBQ does not reach. If
the 4-bit run happens, build LPBQ first and hold GPTVQ for the case where 3
bits becomes interesting.

**Two kernel ideas worth reading if that day comes.** CodeGEMM precomputes the
inner products of the centroids with the activation once into a table, then
each index gathers a partial sum, which amortizes over the whole weight stream
at batch 1 (https://arxiv.org/abs/2512.17970). UniSVQ makes each codeword an
affine map `A·w_int + B` and folds `A` into the activations, thus the weight
stays an integer and no lookup happens at all, for 40 bytes for each matrix
(https://arxiv.org/abs/2606.10520).

### 10.6 The prize

From `tools/prof/bytes.py`, the 4B decode ceiling from bytes alone: **11.1 t/s
at 8 bits** and **20.6 t/s at 4 bits**. Decode is bandwidth bound, thus that
ratio is real and it is nearly a factor of two.

### 10.7 The verdict

Ship 8 bits. Schedule one quantization-aware training run at 4 bits on the 2B,
with the LPBQ format, after the 8-bit path is on the phone. It is a day of
box time against a factor of two in decode speed, and it is the only
experiment in this document with a chance of changing the product.

Do not spend another week on 4-bit post-training quantization. Section 10.3 is
the ceiling and we are already near it.

---

## 11. The build order

Each row gives the effort and what it is expected to buy. The gains are from
the measurements of section 2 where they exist, and from the literature where
they do not, and the table says which.

| # | Step | Effort | Expected gain | Source of the estimate |
|---|---|---|---|---|
| 1 | Per-row int8 export of the folded reference, plus the KL harness. No solver, no calibration. | 1 day | Sets the baseline. Predicted mean KL 0.0025 to 0.003 | our section 2.1 |
| 2 | Online Hadamard at `ffn_down`, `ssm_out`, `attn_output`, fused into the producing kernel | 1 to 2 weeks, most of it kernel | +9.7 to +14.6 dB of activation SNR at the three sites | our section 2.3 |
| 3 | The zero-point epilogue and `colsum(W)` | 3 days | Necessary for u8 asymmetric. Not a gain, a correctness item | section 5.2 |
| 4 | Bias correction, weight side and activation side | 4 days | The only measured 8-bit per-channel weight gain, plus the mean-spread term | Nagel, BASE-Q |
| 5 | Column scales re-searched against an activation objective, exponent 0.8 to 0.9 | 3 days | We own the machinery. The exponent is wrong today | SmoothQuant |
| 6 | Acceptance-rate, chain-length, IFEval and a tool-call test in the harness | 3 days | Closes three blind spots. Tool use and instruction following break first | section 9.2 |
| 7 | Percentile clip at the two gated sites, with a sweep | 1 day | Up to 2.6 points at W8A8, if the rotation leaves anything | Quamba |
| 8 | KLT composition of the foldable rotations, gated delta net layers first | 3 days | +13.8 points at W8A8 on a Mamba, unknown here | MambaQuant |
| 9 | Qronos refit alone, measured against nothing | 1 day | Probably zero. Cheap to settle | section 4.4 |
| 10 | Move `ssm_alpha` and `ssm_beta` off F32 | 1 day | Bytes back | Minima, QUASAR |
| 11 | Prefixed outlier tokens, if the token statistics justify them | 3 days | Removes the token-wise half | PrefixQuant |
| 12a | **Score the public QUASAR Qwen3.5-4B `Q4_0` checkpoint on our own harness.** No training, no box, one download | half a day | Tells us what a 4-bit trained model of our exact model measures, before we spend a day building one | section 10.4b |
| 12b | One 4-bit quantization-aware training run on the 2B, LPBQ format, **only if 12a is encouraging** | 1 to 3 days of box time plus a week of code | Up to 1.9x decode | section 10 |

Steps 1 and 2 are the recipe. Steps 3 to 5 are cheap and additive. Steps 6 to
11 are measurements and refinements. Step 12 is the next product decision.

Three items sit outside the 8-bit path and should not block it, but they are
one line each and they stop a wrong number from misleading somebody later.

| Item | Effort | Why |
|---|---|---|
| Anchor the trellis scale on the block root mean square in `quant/grids.py` and re-run the grid test | 1 hour | The recorded trellis result in `quant-results.md` is wrong by about 1 dB. Section 12.10 |
| Verify whether `IQ4_NL` reaches the tiled path that feeds the HMX | 1 hour | There is no `IQ4_NL_TILED` in the backend. If it falls to HVX, every grid speed comparison we hold is between two different kernels. Section 1.5 |
| Add `IQ4_XS` to the Hexagon backend, **only if 4 bits returns** | 3 days | 0.25 bits for each weight, 0.02 dB, both halves of the machinery are in the tree. Section 12.10c |

---

## 12. What not to do, and the number that rules it out

### 12.1 Do not build a rounding solver for the 8-bit path

QuaRot Table 9, per-column symmetric weights with a Hadamard: INT8 round to
nearest and INT8 GPTQ give **identical perplexity to two decimals** on
Llama-2 7B, 13B and 70B. The whole headroom is 0.03 perplexity on the 7B.
Red Hat ships a W8A8 model calibrated on **random token identifiers** and
recovers 100.31 %.

### 12.2 Do not build a per-row clipping search

Our own measurement: 2 % to 4 %, with the optimum at 0.97 to 0.99 of absmax.
The ACIQ condition puts the 8-bit Laplace optimum at 9.9b where our rotated
rows sit at 4.6b, thus absmax is already inside it. OCTAV measured max-scaling
winning on two of six models at 8 bits. `llm-compressor` measured its MSE
observer as a production regression, 0.7767 to 0.7171.

### 12.3 Do not learn the rotation

Three lines of evidence.

Our own measurement: every class whose reduction axis R1 rotates already sits
at the Gaussian minimum, to within 5 %. There is nothing to learn.

The theory: ButterflyQuant measured that a Hadamard reaches `mu = 1/sqrt(n)`,
which is the Welch bound and thus **optimal** worst-case coherence, while a
**learned** butterfly lands 1.2 to 2 times worse. A learned rotation trades
worst-case incoherence for average-case fit. At 8 bits the worst case is what
matters.

The bit-width trend: HARP against a randomized Hadamard on Llama-2-7B gives
−0.99 perplexity at 2 bits, −0.10 at 3 bits and −0.05 at 4 bits. Each bit cuts
the gain by roughly ten. One more octave puts it at zero, and the entire 8-bit
headroom is smaller than the 4-bit step.

One exception is worth reading and not building: OptRot learns on the **weight**
distribution with the data-free objective `‖vec(W̃)‖_4^4`, and measures 8 % of
KL over a random Hadamard at W4A8. It is twenty lines and it needs no data. If
step 1 of section 11 comes in above the prediction, try it.

### 12.4 Do not apply an online transform where R1 already applies a folded one

Our own measurement: **−0.3 dB**. SpinQuant states the same at W4A8.

### 12.5 Do not quantize the recurrent state

2.2 % of the decode bytes against a measured 67-point AIME collapse, on a decay
spectrum we have confirmed matches the one where it was measured. Section 6.1.

### 12.6 Do not keep the low-rank correction

Microsoft measured its own weight-only result and wrote that low-rank terms "do
not provide any additional improvement". SPEAR measured the gain falling from
0.64 to 0.15 perplexity when the group goes from per-channel to 128, and our
block is finer than 128. The controlled study measured GPTQ plus low-rank at
11.73 against GPTQ alone at 11.80, which is noise. At 8 bits it is dead. Give
the bytes back.

### 12.7 Do not use FlatQuant

198.1 M online multiply-accumulate operations for each token on an 8B model,
against 15.4 M for SpinQuant, and a measured **0.71x decode throughput at batch
1**. Our decode is batch 1. Two independent evaluations also report it
collapsing on Qwen2.5-3B.

### 12.8 Do not adopt any method that needs a scale along the reduction axis

Atom, QUIK, NVFP4, MXFP4, the per-state-group scales of Quamba2, the key-axis
mixed precision of DAMP. The engine cannot express them. llm.npu measured the
price at 8.1 to 10.7 times on this silicon.

### 12.9 Do not adopt any method with a higher-precision side channel

LLM.int8, SVDQuant, QUAD, the CPU shadow path of llm.npu. They need a second
kernel at a different precision. llm.npu measured its own CPU and NPU
synchronization at **29.7 %** of end-to-end latency on Qwen1.5-1.8B, which is
the empirical reason not to.

### 12.10 Do not spend time on trellis codes, vector codebooks or per-block bits

A 256-level grid has no shaping gain worth a kernel. At 8 bits with one scale
for each row there is no block structure left for a codebook to exploit.

**But correct the record on our own 4-bit measurement first, because it is
wrong.** `analysis/quant-results.md` records a QTIP-style trellis at 22.1 dB
against IQ4_NL at 22.5 dB, and the pipeline note calls the trellis "parked" on
that basis. An independent rebuild of the experiment reproduced our numbers and
then found the cause, and it is a defect in `quant/trellis.py` and
`quant/grids.py` and not a property of the trellis.

`Grid.scale_rtn` maps the signed maximum of a block onto the level of largest
magnitude. That is correct for a bounded 16-level table. It is wrong for a
trellis table of 2^L Gaussian samples. `Trellis.gaussian` builds
`randn(1<<L) * 40`, and the maximum of 32 Gaussian weights is about 2.2 sigma,
thus mapping it onto 127 makes `x/s` about 1.44 times too wide for the table
and the outer levels overload. Worse, the correct factor **moves with L**,
because the extreme order statistic of 2^L samples grows with L, thus a scale
search range tuned for 16 levels never contains the trellis optimum.

Anchored on the block root mean square instead, on real Qwen3.5-2B tensors at
the same 4.500 bits for each weight: L=12 goes from 21.51 dB to **22.98 dB**
and L=16 reaches **23.70 dB**, against IQ4_NL at 21.97 dB. The trellis moves
from 0.4 dB behind to 1.0 to 1.7 dB ahead.

The conclusion does not change and the reason changes completely. **Do not
build a trellis decoder for this part**, because the smallest table that beats
IQ4_NL needs about 32 chained HVX lookup passes for each vector, against one
instruction for a 16-entry table, and that buys 0.41 dB, which is about 0.005
of perplexity. The computed form emits a 10-bit value and cannot feed an int8
matrix engine at all.

Fix the scale reference anyway. It is one line, and leaving a wrong number in
`quant-results.md` costs somebody a week later.

**The diagnostic signature, for the record.** With the absmax anchor and a
sweep of 0.65 to 1.15, L=12 gave 21.03 dB and L=16 gave **19.75 dB**. A larger
trellis measuring worse than a smaller one is the signature of this defect,
because the extreme order statistic of 2^L samples grows with L and the sweep
range never contains the correct factor.

### 12.10a Why the two mechanisms never added, which is the real lesson

This part generalizes past the trellis and is worth keeping.

For a Gaussian source at a high rate, a fixed-rate scalar quantizer sits
**4.35 dB** from the rate-distortion bound. That distance has two named parts:
**1.53 dB of space-filling loss**, which is the penalty of a one-dimensional
lattice, and **2.82 dB of shaping loss**, which is the penalty of a fixed
codebook against a source whose local energy varies.

Now measure what each mechanism recovers, on synthetic Gaussian data at 4 bits
against a Lloyd-Max grid with one scale for the whole matrix at 20.17 dB:

| Mechanism | Cost | Recovers |
|---|---|---|
| a block-32 F16 scale | 0.5 bits for each weight | **+2.15 dB** |
| an L=16 trellis, one scale | 0 bits | **+2.83 dB** |

Both numbers are close to the 2.82 dB shaping term, because **both mechanisms
attack the same term**. A block scale is side information about block energy.
A Gaussian-valued trellis codes that energy into the path. Stacking them pays
twice for one thing, and then the 0.5 bits taken from the code makes the
trellis worse than the scalar grid it should beat.

The rate-distortion bound for reference: **24.08 dB at 4.000 bits** and
**27.09 dB at 4.500 bits** for each weight. Our IQ4_NL baseline at 22.23 dB is
already within 4.9 dB of the 4.5-bit bound, and most of what is left is the
space-filling term that only a genuine vector code can reach.

This is also why QTIP's published gain looked larger than ours. Its scalar
reference is Lloyd-Max with one scale, at 20.17 dB. Our baseline is 22.23 dB.
The paper never grants the baseline the 2.15 dB that a block scale already
gives, and it tabulates no GPTQ and no AWQ row at all.

Two more asymmetries in the published comparisons, both measured on our real
tensors. A random Hadamard transform is worth **+2.19 dB** to a scale-free
trellis and only **+0.32 dB** to the block-scaled IQ4_NL grid, and every
published comparison gives the rotation to the trellis and not to the
baseline. And our Ungerboeck-coset result at 21.8 dB against the Gaussian
table at 22.1 dB is the **correct** ordering, not a defect: classical set
partitioning keeps a uniform value set, thus it collects the 1.53 dB
space-filling term and not the 2.82 dB shaping term.

### 12.10b Do not fit a codebook to our own weights either

Before anyone proposes a learned table, it has been measured. A 16-value table
fit to the block-32-normalized values of three real Qwen3.5-2B tensors, by
alternating a scale solve with a Lloyd step:

| Tensor | Fixed `IQ4_NL` | Table fit to that tensor |
|---|---|---|
| `attn out_proj` layer 10 | 22.04 dB | 22.13 dB |
| `mlp down_proj` layer 10 | 22.03 dB | 22.09 dB |
| `mlp gate_proj` layer 3 | 22.15 dB | 22.20 dB |

**0.05 to 0.09 dB.** And the fitted tables come out almost identical to the
fixed one:

    IQ4_NL             : -127 -104 -83 -65 -49 -35 -22 -10  1 13 25 38 53 69 89 113
    fit, L10 down_proj : -127 -100 -80 -63 -48 -34 -22 -10  1 12 24 37 52 68 87 113

Kawrakow's table is already near optimal for block-32 normalized weights of a
language model. Our own `fit_codebook` in `quant/grids.py` therefore has almost
nothing to find, which matches our recorded CB4 result of 22.3 dB against
IQ4_NL at 22.5.

One trap if anyone retries it. Fit the table to the **block-normalized**
values, never to the raw weight distribution. Fitting to the raw distribution
measures **0.4 to 0.8 dB worse** than the fixed table.

The transferable part of the learned-table line (any4, SqueezeLLM) is the
**activation weighting** of the fit, not the per-row table. And even that is
worth 0.12 perplexity in its own paper.

### 12.10c One 4-bit item to remember, if 4 bits ever returns

Our block-32 F16 scale costs 0.500 bits for each weight. A 6-bit block scale
inside a 256 super-block costs 0.195. Measured on real Qwen3.5-2B tensors:
F16 per 32 gives 22.05 / 22.03 / 22.15 dB on `out_proj`, `down_proj` and
`gate_proj`, and 6-bit per 32 gives **22.02 / 22.01 / 22.13** at 4.250 bits
instead of 4.500.

**0.25 bits for each weight, for 0.02 dB.** That is 5.6 % of the weight stream
on a bandwidth-bound decode. It is the IQ4_XS layout, and the Hexagon backend
supports `IQ4_NL` and `Q4_K` but does not list `IQ4_XS`. Both halves of the
machinery are in the tree.

This is irrelevant to the 8-bit path, where the per-row scale is already 0.031
bits for each weight and there is nothing to shrink. It belongs to section 10.

### 12.11 Do not curate the calibration set

COLA measured the whole spread at 0.45 perplexity and 1 to 2 accuracy points.
AMD measured that a per-channel method converges from a single sequence. C4
English at 128 x 2048 is correct. The one worthwhile change is a chat-like
slice for the task metrics, not for the calibration.

### 12.12 Do not trust a paper's baseline

Every number in this area is comparable only inside one paper. The GPTAQ
authors reported a GPTQ baseline of 6.88 for W3g128 Llama-2-7B where OmniQuant
reported 6.29 for the same nominal configuration, and the difference was
symmetric against asymmetric quantization. At 8 bits, where the whole effect we
hunt is 0.03 perplexity, that kind of setup difference is an order of magnitude
larger than the thing being measured.

---

## 13. Appendix A: the measurements in this document

All three ran on the laptop, read only, with no job on the box and no phone.
The scripts are in the session scratchpad and are reproduced here in summary.

### A.1 The per-row scale cost, section 2.1

For each 2-D weight, `mean_rows( max(|row|)^2 / mean_blocks(max_block(|row|)^2) )`,
over all layers, for `weights/Qwen3.5-2B` and `weights/Qwen3.5-2B-t`. The
Gaussian reference comes from the same statistic on `torch.randn(4000, n)`:
n = 2048 gives 2.32, n = 4096 gives 2.54, n = 6144 gives 2.68, n = 9216 gives
2.82.

### A.2 The per-row clipping optimum, section 2.2

For each row, sweep the clip limit over 41 factors in [0.60, 1.00] of the row
maximum, quantize to 127 levels, and take the factor that minimizes the mean
squared error of the row. Report the mean best factor and the mean error ratio
against the factor 1.00.

### A.3 The per-site activation statistics, section 2.3

Hook every `nn.Linear` of interest, record the input of each token, and report
three numbers: the mean of `max(|x|)/rms(x)` over tokens, the signal-to-noise
ratio of a symmetric per-token int8 quantizer, and the same after a block-128
Hadamard. 2 sequences of 512 tokens from `data/calib-128x2048.pt`, float32 on
the CPU, 64 tokens sampled for each forward pass.

### A.4 The decay spectrum, section 6.1

`alpha = exp(-exp(A_log) * softplus(dt_bias))` over the 288 head entries of the
2B, from the 24 `A_log` and 24 `dt_bias` tensors of the original checkpoint.

### A.5 What is not measured, and should be

- The same activation table on the **4B**. Our conclusions assume the 2B
  transfers, and the 4B has 16 key heads against 32 value heads, thus a
  different channel structure.
- The activation table with a **full** Hadamard in place of block-128. Section
  2.3 predicts 2 to 3 dB more.
- The actual per-row int8 KL. Step 1 of section 11.
- The gated RMSNorm output under quantization. No published measurement exists.
- The MTP acceptance rate under an 8-bit block. No published measurement
  exists.

---

## 14. Appendix B: sources

Grouped by the section that uses them.

**The hardware contract and the format**
- Quant.npu, fully static W8A8 on SM8650 and SM8750 — https://arxiv.org/abs/2605.20295
- llm.npu, Fast On-device LLM Inference with NPUs, ASPLOS 2025 — https://arxiv.org/abs/2407.05858
- MobileQuant, EMNLP 2024 Findings, measured on a Hexagon HTP — https://arxiv.org/abs/2408.13933
- AIMET low-power blockwise quantization — https://quic.github.io/aimet-pages/releases/latest/techniques/lpbq.html
- Is INT8 Portable? — https://arxiv.org/abs/2609.16085
- Qualcomm white paper on neural network quantization — https://arxiv.org/abs/2106.08295

**Rotation and incoherence**
- QuaRot, NeurIPS 2024 — https://arxiv.org/abs/2404.00456
- SpinQuant, ICLR 2025 — https://arxiv.org/abs/2405.16406
- QuIP#, the incoherence bound — https://arxiv.org/abs/2402.04396
- OSTQuant, the per-site ablation, ICLR 2025 — https://arxiv.org/abs/2501.13987
- OptRot, data-free weight-side rotation — https://arxiv.org/abs/2512.24124
- ButterflyQuant, the Welch bound result — https://arxiv.org/abs/2509.09679
- HARP, the bit-width trend — https://arxiv.org/abs/2605.29843
- PeRQ, block rotations and the non-power-of-two cost — https://arxiv.org/html/2601.22347
- InfoQuant, the block-count ablation — https://arxiv.org/abs/2605.26175
- Turning LLM Activations Quantization-Friendly — https://arxiv.org/abs/2506.01967
- FlatQuant, ICML 2025, rejected on decode cost — https://github.com/ruikangliu/FlatQuant

**Weights and rounding**
- Qronos — the refit our pipeline implements
- PiSO, exact per-channel scales — https://arxiv.org/abs/2606.10890
- OCTAV, MSE-optimal clipping, ICML 2022 — https://arxiv.org/abs/2206.06501
- ACIQ, the closed-form clip — https://arxiv.org/abs/1810.05723
- Nagel et al., bias correction, ICCV 2019 — https://arxiv.org/abs/1906.04721
- MagR, weight magnitude reduction, NeurIPS 2024 — https://arxiv.org/abs/2406.00800
- llm-compressor MSE observer regression — https://github.com/vllm-project/llm-compressor/issues/2094
- llm-compressor silent GPTQ fallback — https://github.com/vllm-project/llm-compressor/issues/2952

**Activations**
- SmoothQuant, ICML 2023, the granularity ablation — https://arxiv.org/abs/2211.10438
- PrefixQuant — https://arxiv.org/abs/2410.05265
- BASE-Q, the per-channel bias — https://arxiv.org/abs/2506.15689
- Massive Activations, COLM 2024 — https://arxiv.org/abs/2402.17762
- Depth Registers, why down_proj resists rotation — https://arxiv.org/abs/2604.18128
- Give Me BF16 or Give Me Death, the production W8A8 recipe — https://arxiv.org/abs/2411.02355
- Red Hat W8A8 study, 2026-09-14 — https://developers.redhat.com/articles/2026/09/14/understanding-w8a8-int8-llm-quantization-accuracy-and-performance-results
- SageAttention, the K mean subtraction, ICLR 2025 — https://arxiv.org/abs/2410.02367
- INT-FlashAttention, the axis rules — https://arxiv.org/abs/2409.16997

**Linear attention, state space and hybrids**
- Quamba, the output Hadamard ablation at W8A8 — https://arxiv.org/abs/2410.13229
- Quamba2, ICML 2025 — https://arxiv.org/abs/2503.22879
- MambaQuant, the KLT-composed rotation, ICLR 2025 — https://arxiv.org/abs/2501.13484
- SSDi8, the persistent int8 path, ICLR 2026 — https://arxiv.org/abs/2608.21952
- Q-Mamba, the outer-product state structure, ACL 2025 Findings — https://aclanthology.org/2025.findings-acl.551/
- QMamba, the log-domain decay quantizer — https://arxiv.org/abs/2501.13624
- DAMP, the INT8 state collapse — https://arxiv.org/abs/2608.27513
- Minima, Why Gated DeltaNet Survives 4-Bit Quantization — https://arxiv.org/abs/2609.04098
- QUASAR, quantization-aware training on the same family — https://arxiv.org/abs/2608.13966
- Pimba, the swamping effect, MICRO 2025 — https://doi.org/10.1145/3725843.3756121
- NVIDIA Nemotron 3 Ultra, stochastic rounding of the state — https://arxiv.org/abs/2606.15007
- carrykernel, error feedback on a Qwen3.5 state, unreviewed — https://github.com/tomalmog/carrykernel
- Mamba-PTQ, the dt projection is clean — https://arxiv.org/abs/2407.12397

**Evaluation, calibration and mixed precision**
- TASA, the Perplexity Illusion — https://arxiv.org/abs/2607.00908
- Quantized Reasoning Models Think They Need to Think Longer — https://arxiv.org/abs/2606.00206
- Quantization Hurts Reasoning? — https://arxiv.org/abs/2504.04823
- Does quantization affect long-context tasks?, EMNLP 2025 — https://arxiv.org/abs/2505.20276
- COLA, calibration data curation — https://arxiv.org/abs/2510.10618
- RSQ, token-weighted reconstruction — https://arxiv.org/abs/2503.01820
- A Comprehensive Evaluation on Quantization Techniques — https://arxiv.org/abs/2507.17417
- Low-Rank Correction for Quantized LLMs, the negative result — https://arxiv.org/abs/2412.07902
- SPEAR, the granularity dependence of low rank — https://arxiv.org/abs/2606.11244
- Optimal Formats for Weight Quantisation, the Fisher rule — https://arxiv.org/abs/2505.12988
- EfficientQAT, ACL 2025 — https://arxiv.org/abs/2407.11062
- ParetoQ, which argues against 4 bits, NeurIPS 2025 — https://arxiv.org/abs/2502.02631
- BitDistiller — https://arxiv.org/abs/2402.10631
- Flat Score, Amplified Failures, the agent failure amplification — https://arxiv.org/abs/2607.27275
- QUASAR Qwen3.5-4B `Q4_0` checkpoint, our exact model — https://huggingface.co/QUASAR-QAT/Qwen3.5-4B-QUASAR-Q4_0-GGUF
- Meta Llama quantized lightweight models, the 1600 GPU-hour figure — https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/
- Google Gemma 3 quantization-aware training, the vendor-parent trap — https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/
- QSpec, the one sentence on draft acceptance — https://arxiv.org/abs/2410.11305
- SpecMQuant, the target-side result — https://arxiv.org/abs/2505.22179
- unsloth Qwen3.5 GGUF benchmarks — https://unsloth.ai/docs/models/qwen3.5/gguf-benchmarks

**Grids, codebooks and trellis codes, for section 10 and section 12.10**
- QTIP, trellis-coded quantization, NeurIPS 2024 — https://arxiv.org/abs/2406.11235
- QuIP#, the E8 lattice, ICML 2024 — https://arxiv.org/abs/2402.04396
- GPTVQ, built for a mobile lookup instruction — https://arxiv.org/abs/2402.15319
- any4, learned scalar tables, ICML 2025 — https://arxiv.org/abs/2507.04610
- SqueezeLLM — https://arxiv.org/abs/2306.07629
- HIGGS, data-free Gaussian grids — https://arxiv.org/abs/2411.17525
- Q-Palette, the allocator beats the grid, NeurIPS 2025 — https://arxiv.org/abs/2509.20214
- Grid Games, two grids with a free selector — https://arxiv.org/abs/2605.12327
- CodeGEMM, the precomputed partial-sum table, NeurIPS 2025 — https://arxiv.org/abs/2512.17970
- UniSVQ, affine codewords that keep the weight integer — https://arxiv.org/abs/2606.10520
- AQLM, rejected on codebook size — https://arxiv.org/abs/2401.06118
- VPTQ, rejected on reproducibility — https://arxiv.org/abs/2409.17066
- MatQuant, rejected on 4-bit accuracy — https://arxiv.org/abs/2502.06786
- ParetoQ, whose own 4-bit result loses to post-training quantization — https://arxiv.org/abs/2502.02631
- Leech lattice vector quantization, rejected on served bit rate — https://arxiv.org/abs/2603.11021
- GPTAQ / GPTQv2, about 20 lines over GPTQ, 4-bit only — https://arxiv.org/abs/2504.02692
- llama.cpp i-quant design and the IQ4_NL measurements — https://github.com/ggml-org/llama.cpp/pull/5590 and https://github.com/ggml-org/llama.cpp/discussions/5063
- ik_llama.cpp, the trellis verdict from the author of the i-quants — https://github.com/ikawrakow/ik_llama.cpp/discussions/2213 and https://github.com/ikawrakow/ik_llama.cpp/pull/113
- ik_llama.cpp, the NEON cost of a trellis, 5.73 against 27.48 tokens for each second — https://github.com/ikawrakow/ik_llama.cpp/pull/471
- EXL3 against GGUF at matched file size — https://atomic.chat/blog/guides/exl3-vs-gguf
- ExLlamaV3, the living trellis implementation — https://github.com/turboderp-org/exllamav3

**Our own documents**
- `analysis/quant-results.md` — every row this recipe cites
- `analysis/quant-attribution.md` — the per-class KL attribution
- `tools/prof/bytes.py` — the byte budget
- `tools/hmx-bench/src/hmx_rate.c` — the 2.05x measurement
