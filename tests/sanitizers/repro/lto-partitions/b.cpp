#include "common.h"
int helper_b(int x) { g_logger.log("b: called\n"); return helper_a(x) * 2; }
