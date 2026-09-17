// Stub of the DSP log API for the simulator. ERROR and ALWAYS print, the other levels are silent.
#ifndef LAB_STUB_HAP_FARF_H
#define LAB_STUB_HAP_FARF_H

#include <stdio.h>

#define LAB_FARF_ON_ERROR  1
#define LAB_FARF_ON_ALWAYS 1
#define LAB_FARF_ON_FATAL  1
#define LAB_FARF_ON_HIGH   0
#define LAB_FARF_ON_MEDIUM 0
#define LAB_FARF_ON_LOW    0

#define FARF(level, ...)                             \
    do {                                             \
        if (LAB_FARF_ON_##level) {                   \
            printf("farf %s: ", #level);             \
            printf(__VA_ARGS__);                     \
            printf("\n");                            \
        }                                            \
    } while (0)

#endif
