// Stub of the QuRT thread API. The lab runs the kernel threads on the standalone runtime instead.
#ifndef LAB_STUB_QURT_THREAD_H
#define LAB_STUB_QURT_THREAD_H

#include <stdint.h>

typedef unsigned int qurt_thread_t;

typedef struct {
    void *       stack_addr;
    unsigned int stack_size;
    unsigned int priority;
    char         name[16];
} qurt_thread_attr_t;

static inline void qurt_thread_attr_init(qurt_thread_attr_t * attr) { (void) attr; }
static inline void qurt_thread_attr_set_stack_addr(qurt_thread_attr_t * attr, void * addr) { attr->stack_addr = addr; }
static inline void qurt_thread_attr_set_stack_size(qurt_thread_attr_t * attr, unsigned int size) { attr->stack_size = size; }
static inline void qurt_thread_attr_set_priority(qurt_thread_attr_t * attr, unsigned int prio) { attr->priority = prio; }
static inline void qurt_thread_attr_set_name(qurt_thread_attr_t * attr, const char * name) { (void) attr; (void) name; }

int qurt_thread_create(qurt_thread_t * thread, qurt_thread_attr_t * attr, void (*entry)(void *), void * arg);
int qurt_thread_join(qurt_thread_t thread, int * status);
qurt_thread_t qurt_thread_get_id(void);
int qurt_thread_get_priority(qurt_thread_t thread);

#endif
