// Stub of the QuRT API for the simulator. The kernels use only the types and the PMU names here.
#ifndef LAB_STUB_QURT_H
#define LAB_STUB_QURT_H

#include <stdint.h>

#include "qurt_thread.h"
#include "qurt_futex.h"
#include "qurt_memory.h"

#define QURT_PMUCNT0 0
#define QURT_PMUCNT1 1
#define QURT_PMUCNT2 2
#define QURT_PMUCNT3 3
#define QURT_PMUCNT4 4
#define QURT_PMUCNT5 5
#define QURT_PMUCNT6 6
#define QURT_PMUCNT7 7

static inline uint32_t qurt_pmu_get(int counter) {
    (void) counter;
    return 0;
}

#endif
