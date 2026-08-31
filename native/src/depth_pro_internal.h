#pragma once

#include "external_gpu.h"

#include <memory>

struct depth_pro_context;

namespace depth_pro_native {

std::shared_ptr<ExternalGpu> create_metal_external_gpu(
    depth_pro_context* context,
    float forced_fov_degrees);

}  // namespace depth_pro_native
