// Target i8read: the device bench tools/hmx-bench/src/i8read.c in the simulator. The simulator
// runs the HMX in functional mode only, thus this target runs the correctness checks of the bench
// (the 4-plane and 3-plane reads, the accumulator swap, the HVX join) and its times are 0. Run it
// with MODE=functional.
#include "../../hmx-bench/src/i8read.c"
