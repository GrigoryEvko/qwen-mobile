// The oracle of the census on the build machine, for a check of host/oracle.c against the
// simulator outputs. Refer to oracle-sim-test.sh.
//
//   oracle_sim <table.txt> <corpus_dir> <out_dir>
//
// table.txt (made by oracle-sim-test.sh from tools/htp-lab/isa/isa_ops.csv) has two kinds of lines:
//   stream <id> <name>
//   op <id> <name> <out_type> <out_bytes> <n_vectors> <in_type0> <in_type1> <in_type2>
//      <in_bytes0> <in_bytes1> <in_bytes2> <stream0> <stream1> <stream2>
// The types are enum isa_type values and the streams are stream IDs (0 for none). For each op
// with an oracle the program reads the corpus streams corpus_<name>.bin, runs oracle_run, and
// writes out_<op name>.oracle.bin into out_dir. Exit code 0, or 1 after an error message.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "census_io.h"
#include "isa_kernels.h"
#include "oracle.h"

#define MAX_STREAMS 64

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: oracle_sim <table.txt> <corpus_dir> <out_dir>\n");
        return 1;
    }
    FILE * f = fopen(argv[1], "r");
    if (f == NULL) {
        fprintf(stderr, "oracle_sim: error: cannot open %s\n", argv[1]);
        return 1;
    }
    char      stream_name[MAX_STREAMS][64];
    uint8_t * stream_file[MAX_STREAMS] = { 0 };
    size_t    stream_bytes[MAX_STREAMS] = { 0 };
    memset(stream_name, 0, sizeof(stream_name));

    char line[1024];
    int  n_ops = 0, n_oracle = 0, code = 0;
    while (code == 0 && fgets(line, sizeof(line), f) != NULL) {
        unsigned id = 0;
        char     name[256];
        if (sscanf(line, "stream %u %63s", &id, name) == 2 && id < MAX_STREAMS) {
            snprintf(stream_name[id], sizeof(stream_name[id]), "%s", name);
            continue;
        }
        struct oracle_op op;
        unsigned         t[3], b[3], s[3], out_type, out_bytes, n_vec;
        if (sscanf(line, "op %u %255s %u %u %u %u %u %u %u %u %u %u %u %u", &id, name, &out_type, &out_bytes, &n_vec,
                   &t[0], &t[1], &t[2], &b[0], &b[1], &b[2], &s[0], &s[1], &s[2]) != 14) {
            continue;
        }
        n_ops++;
        op.name      = name;
        op.out_type  = out_type;
        op.out_bytes = out_bytes;
        op.n_vectors = n_vec;
        for (int j = 0; j < 3; j++) {
            op.in_type[j]  = t[j];
            op.in_bytes[j] = b[j];
        }
        struct oracle_spec spec;
        if (!oracle_find(&op, &spec)) {
            continue;
        }
        const uint8_t * in[3] = { NULL, NULL, NULL };
        for (int j = 0; j < 3; j++) {
            if (s[j] == 0 || b[j] == 0) {
                continue;
            }
            if (s[j] >= MAX_STREAMS || stream_name[s[j]][0] == '\0') {
                fprintf(stderr, "oracle_sim: error: op %s reads the unknown stream %u\n", name, s[j]);
                code = 1;
                break;
            }
            if (stream_file[s[j]] == NULL) {
                char file[128], path[4096];
                snprintf(file, sizeof(file), "corpus_%s.bin", stream_name[s[j]]);
                if (census_join(path, sizeof(path), argv[2], file) != 0 ||
                    census_read_stream(path, s[j], &stream_file[s[j]], &stream_bytes[s[j]]) != 0) {
                    code = 1;
                    break;
                }
            }
            if ((size_t) n_vec * b[j] > stream_bytes[s[j]]) {
                fprintf(stderr, "oracle_sim: error: op %s reads more than the stream %s has\n", name, stream_name[s[j]]);
                code = 1;
                break;
            }
            in[j] = stream_file[s[j]] + sizeof(isa_file_header);
        }
        if (code != 0) {
            break;
        }
        const size_t out_len = (size_t) n_vec * out_bytes;
        uint8_t *    out     = (uint8_t *) calloc(out_len ? out_len : 1, 1);
        if (out == NULL) {
            fprintf(stderr, "oracle_sim: error: no memory\n");
            code = 1;
            break;
        }
        oracle_run(&op, &spec, in, out);
        char file[300], path[4096];
        snprintf(file, sizeof(file), "out_%s.oracle.bin", name);
        if (census_join(path, sizeof(path), argv[3], file) != 0 ||
            census_write_output(path, id, name, n_vec, out_bytes, 0, CENSUS_SOURCE_ORACLE, out, out_len) != 0) {
            code = 1;
        }
        free(out);
        n_oracle++;
    }
    fclose(f);
    for (int k = 0; k < MAX_STREAMS; k++) {
        free(stream_file[k]);
    }
    printf("oracle_sim: %d ops, %d with an oracle\n", n_ops, n_oracle);
    return code;
}
