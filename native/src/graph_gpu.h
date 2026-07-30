#pragma once

#include "gpu_model.h"
#include "operators.h"

#include <cstdint>
#include <vector>

namespace depth_pro_native {

struct GpuInferenceOutput {
    std::vector<float> depth;
    float focal_length_pixels = 0.0f;
};

GpuInferenceOutput infer_gpu(
    VulkanContext& context,
    GpuModel& model,
    VulkanOperators& operators,
    const float* rgb_chw,
    std::uint32_t width,
    std::uint32_t height);

}  // namespace depth_pro_native
