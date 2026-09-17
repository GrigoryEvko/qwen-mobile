// Stub of the QuRT HVX lock API.
#ifndef LAB_STUB_QURT_HVX_H
#define LAB_STUB_QURT_HVX_H

static inline int qurt_hvx_lock(int mode) { (void) mode; return 0; }
static inline int qurt_hvx_unlock(void) { return 0; }

#endif
