/**
 * A main for a fuzzer that is built without libFuzzer (FUZZ_LIBFUZZER=OFF):
 * it runs each file of the command line, or each file of each directory,
 * as one input. Thus the regression inputs run on a build with no libFuzzer.
 */
#include "fuzz_death.h"

#include <dirent.h>
#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t * data, size_t size);
extern "C" __attribute__((weak)) int LLVMFuzzerInitialize(int * argc, char *** argv);

namespace {

/** Run one file as one input. */
void run_file(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    const std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    fprintf(stderr, "run %s (%zu bytes)\n", path.c_str(), bytes.size());
    // In a TSan build, the death callback of fuzz_death.h writes this input.
    fuzz_death_note_input(bytes.data(), bytes.size());
    LLVMFuzzerTestOneInput(bytes.data(), bytes.size());
}

}  // namespace

/** Run each argument: a file, or each regular file of a directory. */
int main(int argc, char ** argv) {
    if (LLVMFuzzerInitialize != nullptr) {
        LLVMFuzzerInitialize(&argc, &argv);
    }
    for (int i = 1; i < argc; ++i) {
        struct stat st = {};
        if (stat(argv[i], &st) != 0) {
            fprintf(stderr, "error: %s does not exist\n", argv[i]);
            return 2;
        }
        if (!S_ISDIR(st.st_mode)) {
            run_file(argv[i]);
            continue;
        }
        DIR * d = opendir(argv[i]);
        while (dirent * e = d != nullptr ? readdir(d) : nullptr) {
            const std::string p = std::string(argv[i]) + "/" + e->d_name;
            struct stat fs = {};
            if (stat(p.c_str(), &fs) == 0 && S_ISREG(fs.st_mode)) {
                run_file(p);
            }
        }
        if (d != nullptr) {
            closedir(d);
        }
    }
    return 0;
}
