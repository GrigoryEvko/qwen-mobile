// Stub of the Hexagon SDK header AEEStdErr.h for the x86 fuzz harness.
// The values are the values of the SDK 6.6.0.0 for the HLOS side.
#pragma once

#include "AEEStdDef.h"

#define AEE_EOFFSET            0x80000400
#define AEE_SUCCESS            0
#define AEE_EFAILED            (AEE_EOFFSET + 0x001)
#define AEE_ENOMEMORY          (AEE_EOFFSET + 0x002)
#define AEE_EVERSIONNOTSUPPORT (AEE_EOFFSET + 0x004)
#define AEE_EEXPIRED           (AEE_EOFFSET + 0x00C)
#define AEE_EBADPARM           (AEE_EOFFSET + 0x00E)
#define AEE_EUNSUPPORTED       (AEE_EOFFSET + 0x014)
#define AEE_EITEMBUSY          (AEE_EOFFSET + 0x020)
#define AEE_ENOSUCH            (39)
#define AEE_EINTERRUPTED       (46)
#define AEE_EWOULDBLOCK        (516)
#define AEE_EUNSUPPORTEDAPI    (AEE_EOFFSET + 0x06C)
