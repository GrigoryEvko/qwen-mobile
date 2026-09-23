#include "common.h"
#include <cstdlib>
int main(int argc, char **) {
    std::printf("callback %p\n", (void *) g_logger.log_callback);
    return helper_b(argc) > 0 ? 0 : 1;
}
