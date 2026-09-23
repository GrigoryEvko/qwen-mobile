# The profiling kit

Four tools, each answering a different question. Start from the question.

| Question | Tool | Needs the phone |
|---|---|---|
| Is this change worth writing at all? | `prof/bytes.py` | no |
| Why is this kernel slow? | `htp-lab/run.sh` | no, it runs the simulator |
| What did the phone actually do? | `prof/run.py` + `prof/pmu.py` | yes |
| When did each thing run? | `trace/htp_trace.py`, `trace/perfetto/capture.sh` | yes |

Decode on this phone is memory-bandwidth bound at 51 to 56 GB/s. Almost every
question about decode speed is a question about bytes, thus `prof/bytes.py`
comes first and the phone comes last.

## prof/bytes.py — the byte budget, and whether a change can pay

Decode reads the whole weight set once per token. At a fixed weight format the
only two levers are the bytes one step reads and the tokens one step yields.
This tool computes both and needs no hardware.

```
tools/prof/bytes.py budget 4b            # where the bytes go
tools/prof/bytes.py plan 4b --accept 0.65  # the ladder of Q8-preserving changes
tools/prof/bytes.py spec 4b --draft 3 --accept 0.65 --draft-head 3072 --hmx-verify
```

The rates it prints are ceilings from bytes alone. They ignore host overhead and
throttling, thus a measured rate is always lower. Use the tool to rank changes,
never to predict a number. The model tracks measurement to within about 15 %:
the 2B ceiling is 27.1 t/s against a measured 23.6, and the 4B 13.2 against 10.8.

## prof/pmu.py — named counter sets

The Hexagon has eight PMU counters. The backend takes exactly eight event ids in
`GGML_HEXAGON_PROFILE` and then reports one counter delta per op. A question
that needs more than eight events needs more than one run.

```
tools/prof/pmu.py list           # every set
tools/prof/pmu.py show stalls    # the events of one set
tools/prof/pmu.py env stalls     # the assignment the backend parses
```

The set the backend ships by default holds no stall-attribution event, thus a
run can say that an op is slow and never say why. Use `stalls` first. It is the
Hexagon analogue of the Nsight Compute stall-reason breakdown.

`pmu.py todo` lists the coprocessor events that the Snapdragon Profiler NPU
plugin names but whose raw ids we have not confirmed. That family is the only
HMX instrumentation on this part, because the simulator runs HMX functionally
and never retires it in timing mode.

## prof/run.py — a measurement under the protocol

A number from a phone that was charging, or that throttled halfway, is not
comparable with any other number. This driver refuses those runs.

```
tools/prof/run.py check --serial 192.168.14.130:5555
tools/prof/run.py bench --serial <s> --model /sdcard/qwen/models/<file>.gguf --set stalls
```

It gates on four things: the phone is off the charger, the thermal status is 0
before and after, no invocation exceeds 120 seconds, and exactly one adb
transport is named. The last one matters more than it looks. Two transports are
often attached at once, one over USB and one over TCP, and a bare `adb` command
then fails with "more than one device/emulator". **Prefer the TCP transport,
because the USB cable charges the phone.**

Every run writes one JSON record under `prof/store/` that names the device, the
event set, the environment and the thermal status at both ends, thus a number
can always be attributed to a build.

## htp-lab/ — the kernel lab

Compiles one kernel with the production flags, runs it in the cycle-accurate
simulator, and prints the per-function stalls and the packets that stall most,
each with its disassembly. This is the Nsight Compute equivalent and it needs no
phone. Recorded outputs for the conv, Q4, Q8 and HMX kernels are in `out/`.

Its one hole is HMX: the timing model never retires an HMX instruction, thus the
matrix engine has no simulated cycle story. HMX numbers have to come from the
device counters instead.

## trace/ — the timeline

`htp_trace.py` turns a DSP profile log into a Chrome trace for Perfetto, with a
row per DSP thread and a counter track per PMU event. `perfetto/capture.sh`
takes an Android system trace with the app sections on the boottime clock.

The two are not on one clock, thus they cannot be overlaid. Four changes can
join them: keep the absolute qtimer start of each DSP batch, read CNTVCT_EL0
next to CLOCK_BOOTTIME on the host, read dspqueue_get_stat() for each batch,
and write the hostprof times as ATrace slices.

## The order that works

1. `bytes.py` to decide whether the change can pay at all.
2. `htp-lab` to see the packets, offline and with no phone.
3. `run.py --set stalls` to find out what the phone really did.
4. `run.py --set bandwidth` or `dma` once the stall set names the suspect.
5. `trace/` when the question is about order and gaps rather than about one op.
