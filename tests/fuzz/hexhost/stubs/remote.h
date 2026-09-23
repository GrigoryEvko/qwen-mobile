// Stub of the Hexagon SDK header remote.h for the x86 fuzz harness.
// It gives the types, the request codes and the functions of the FastRPC
// session control that the host part of the Hexagon backend uses. The fake DSP
// of the harness (common/fake_dsp.cpp) gives the implementation.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t remote_handle;
typedef uint64_t remote_handle64;

typedef struct {
    void * pv;
    size_t nLen;
} remote_buf;

typedef union {
    remote_buf buf;
    remote_handle h;
    remote_handle64 h64;
} remote_arg;

typedef struct {
    int          id;
    const char * uri;
} domain;

#define CDSP_DOMAIN_ID   3
#define CDSP_DOMAIN_NAME "cdsp"

enum fastrpc_map_flags {
    FASTRPC_MAP_STATIC = 0,
    FASTRPC_MAP_RESERVED,
    FASTRPC_MAP_FD,
    FASTRPC_MAP_FD_DELAYED,
    FASTRPC_MAP_FD_NOMAP,
};

// Request codes of remote_session_control and remote_handle64_control
enum {
    DSPRPC_CONTROL_LATENCY           = 1,
    DSPRPC_CONTROL_UNSIGNED_MODULE   = 7,
    FASTRPC_RESERVE_NEW_SESSION      = 12,
    FASTRPC_GET_EFFECTIVE_DOMAIN_ID  = 13,
    FASTRPC_GET_URI                  = 14,
};

struct remote_rpc_control_latency {
    uint32_t enable;
    uint32_t latency;
};

struct remote_rpc_control_unsigned_module {
    int domain;
    int enable;
};

struct remote_rpc_reserve_new_session {
    char *   domain_name;
    uint32_t domain_name_len;
    char *   session_name;
    uint32_t session_name_len;
    uint32_t effective_domain_id;
    uint32_t session_id;
};

struct remote_rpc_effective_domain_id {
    char *   domain_name;
    uint32_t domain_name_len;
    uint32_t session_id;
    uint32_t effective_domain_id;
};

struct remote_rpc_get_uri {
    char *   domain_name;
    uint32_t domain_name_len;
    uint32_t session_id;
    char *   module_uri;
    uint32_t module_uri_len;
    char *   uri;
    uint32_t uri_len;
};

// The system request of the domain discovery
enum fastrpc_domain_type {
    FASTRPC_LPASS = 0,
    FASTRPC_HPASS,
    FASTRPC_NSP,
};

#define FASTRPC_GET_DOMAINS                      1
#define DOMAINS_LIST_FLAGS_SET_TYPE(flags, type) ((flags) | ((uint32_t) (type) << 8))

typedef struct {
    enum fastrpc_domain_type type;
    int                      id;
    char                     name[32];
    int                      status;
    int                      instance_id;
} fastrpc_domain;

typedef struct {
    fastrpc_domain * domains;
    int              max_domains;
    uint32_t         flags;
    int              num_domains;
} system_req_domains;

typedef struct {
    int id;
    union {
        system_req_domains sys;
    };
} system_req_payload;

int remote_session_control(uint32_t req, void * data, uint32_t datalen);
int remote_handle_control(uint32_t req, void * data, uint32_t datalen);
int remote_handle64_control(remote_handle64 h, uint32_t req, void * data, uint32_t datalen);
int fastrpc_mmap(int domain, int fd, void * addr, int offset, size_t length, enum fastrpc_map_flags flags);
int fastrpc_munmap(int domain, int fd, void * addr, size_t length);

#ifdef __cplusplus
}
#endif
