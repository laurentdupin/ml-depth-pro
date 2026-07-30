#include "depth_pro_native.h"

#include <assert.h>

int main(void) {
    depth_pro_context* context = 0;
    assert(depth_pro_abi_version() == DEPTH_PRO_ABI_VERSION);
    assert(depth_pro_version_string() != 0);
    assert(depth_pro_last_error() != 0);
    assert(
        depth_pro_create(0, &context) ==
        DEPTH_PRO_STATUS_INVALID_ARGUMENT);
    assert(context == 0);
    assert(
        depth_pro_infer_bgra8_f32(
            0, 0, 0, 0, 0, 0.0f, 0, 0, 0) ==
        DEPTH_PRO_STATUS_INVALID_ARGUMENT);
    depth_pro_destroy(0);
    return 0;
}
