// The first lines of the copy of the mapping table of the DSP for fuzz_mmap (CMakeLists.txt writes the copy).
// It gives the headers that the functions of main.c use: the stub of the log (stubs/dsp/HAP_farf.h), the
// descriptors of htp-ops.h, and the fake VA of htp-mmap-api.h in place of HAP_mmap2 and HAP_munmap2.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "HAP_farf.h"
#include "htp-mmap-api.h"
#include "htp-ops.h"
