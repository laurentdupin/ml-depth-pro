#pragma once

#include "model.h"

#include <cstdint>
#include <vector>

namespace depth_pro_native {

struct InferenceOutput {
    std::vector<float> depth;
    float focal_length_pixels = 0.0f;
};

InferenceOutput infer_cpu(
    const ModelFile& model,
    const float* rgb_chw,
    std::uint32_t width,
    std::uint32_t height,
    float forced_fov_degrees = 0.0f);

}  // namespace depth_pro_native
