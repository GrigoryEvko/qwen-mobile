// Stub of the QuRT futex API. Only the prototypes are necessary for the inline queue code.
#ifndef LAB_STUB_QURT_FUTEX_H
#define LAB_STUB_QURT_FUTEX_H

int qurt_futex_wait(void * addr, int value);
int qurt_futex_wake(void * addr, int count);

#endif
