// The files of the HVX census. Refer to census_io.h.

#include "census_io.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "isa_kernels.h"

_Static_assert(sizeof(isa_file_header) == 128, "the census file header must have 128 bytes");

int census_join(char * out, size_t size, const char * dir, const char * name) {
    const int n = snprintf(out, size, "%s/%s", dir, name);
    if (n < 0 || (size_t) n >= size) {
        fprintf(stderr, "isaprobe: error: the path %s/%s is too long\n", dir, name);
        return -1;
    }
    return 0;
}

int census_read_file(const char * path, uint8_t ** data, size_t * len) {
    FILE * f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "isaprobe: error: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }
    long size = -1;
    if (fseek(f, 0, SEEK_END) == 0) {
        size = ftell(f);
    }
    if (size < 0 || fseek(f, 0, SEEK_SET) != 0) {
        fprintf(stderr, "isaprobe: error: cannot get the size of %s: %s\n", path, strerror(errno));
        fclose(f);
        return -1;
    }
    uint8_t * buf = (uint8_t *) malloc(size > 0 ? (size_t) size : 1u);
    if (buf == NULL) {
        fprintf(stderr, "isaprobe: error: no memory for the %ld bytes of %s\n", size, path);
        fclose(f);
        return -1;
    }
    if (size > 0 && fread(buf, 1, (size_t) size, f) != (size_t) size) {
        fprintf(stderr, "isaprobe: error: cannot read %s: %s\n", path, ferror(f) ? strerror(errno) : "short read");
        free(buf);
        fclose(f);
        return -1;
    }
    fclose(f);
    *data = buf;
    *len  = (size_t) size;
    return 0;
}

int census_read_stream(const char * path, uint32_t id, uint8_t ** buf, size_t * data_len) {
    uint8_t * b   = NULL;
    size_t    len = 0;
    if (census_read_file(path, &b, &len) != 0) {
        return -1;
    }
    isa_file_header h;
    const char *    why = NULL;
    if (len < sizeof(h)) {
        why = "the file is shorter than the header";
    } else {
        memcpy(&h, b, sizeof(h));
        const size_t n = len - sizeof(h);
        if (memcmp(h.magic, ISA_FILE_MAGIC_CORPUS, 4) != 0) {
            why = "the magic is not " ISA_FILE_MAGIC_CORPUS;
        } else if (h.version != ISA_FILE_VERSION) {
            why = "the file version is not the version of isa_kernels.h";
        } else if (h.id != id) {
            why = "the stream id of the header is not the id of the file name";
        } else if (h.bytes_per_vector != ISA_VEC_BYTES || (size_t) h.n_vectors * ISA_VEC_BYTES != n) {
            why = "the vector count of the header does not agree with the file size";
        } else if (h.hash != isa_hash(b + sizeof(h), n)) {
            why = "the hash of the vectors is not the hash of the header (a damaged or partial copy)";
        }
    }
    if (why != NULL) {
        fprintf(stderr, "isaprobe: error: %s is not a valid corpus stream: %s\n", path, why);
        free(b);
        return -1;
    }
    *buf      = b;
    *data_len = len - sizeof(h);
    return 0;
}

int census_write_output(const char * path, uint32_t id, const char * name, uint32_t n_vectors, uint32_t bytes_per_vector,
                        uint32_t arch, uint32_t source, const uint8_t * data, size_t len) {
    isa_file_header h;
    memset(&h, 0, sizeof(h));
    memcpy(h.magic, ISA_FILE_MAGIC_OUTPUT, 4);
    h.version          = ISA_FILE_VERSION;
    h.id               = id;
    h.n_vectors        = n_vectors;
    h.bytes_per_vector = bytes_per_vector;
    h.arch             = arch;
    h.hash             = isa_hash(data, len);
    h.source           = source;
    snprintf(h.name, sizeof(h.name), "%s", name);

    FILE * f = fopen(path, "wb");
    if (f == NULL) {
        fprintf(stderr, "isaprobe: error: cannot create %s: %s\n", path, strerror(errno));
        return -1;
    }
    const int write_ok = fwrite(&h, sizeof(h), 1, f) == 1 && (len == 0 || fwrite(data, len, 1, f) == 1);
    const int close_ok = fclose(f) == 0;
    if (!write_ok || !close_ok) {
        fprintf(stderr, "isaprobe: error: cannot write %s: %s\n", path, strerror(errno));
        return -1;
    }
    return 0;
}
