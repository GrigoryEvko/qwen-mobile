// The files of the HVX census (format of tools/htp-lab/isa/isa_kernels.h): a header of 128 bytes,
// then the vectors. The host program and the test drivers use these functions.
#ifndef ISAPROBE_CENSUS_IO_H
#define ISAPROBE_CENSUS_IO_H

#include <stddef.h>
#include <stdint.h>

// The values of the "source" field of the header. 0 and 1 come from isa_kernels.h.
#define CENSUS_SOURCE_SIM    0  // hexagon-sim
#define CENSUS_SOURCE_CHIP   1  // a real chip
#define CENSUS_SOURCE_ORACLE 2  // the CPU oracle of the probe (host/oracle.c)

// Read a full file into a new buffer. Returns 0, or -1 after a message on stderr. The caller
// frees *data.
int census_read_file(const char * path, uint8_t ** data, size_t * len);

// Read and check one corpus stream: the magic, the version, the id, the size and the hash.
// Returns 0 and the full file in *buf (the vectors start at byte 128, *data_len bytes), or -1
// after a message. O(size of the file).
int census_read_stream(const char * path, uint32_t id, uint8_t ** buf, size_t * data_len);

// Write an output file: the header ("HVXO", the id, the name, the vector count, the bytes of one
// vector, the arch, the source and the hash of the data), then the data. Returns 0, or -1 after
// a message.
int census_write_output(const char * path, uint32_t id, const char * name, uint32_t n_vectors, uint32_t bytes_per_vector,
                        uint32_t arch, uint32_t source, const uint8_t * data, size_t len);

// Join a directory and a file name into out. Returns 0, or -1 after a message when the path is
// too long.
int census_join(char * out, size_t size, const char * dir, const char * name);

#endif
