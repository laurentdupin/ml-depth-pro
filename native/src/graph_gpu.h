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

struct GpuDeviceInferenceOutput {
    VulkanBuffer depth;
    VulkanBuffer fov_degrees;
};

GpuDeviceInferenceOutput infer_gpu_device(
    VulkanContext& context,
    GpuModel& model,
    VulkanOperators& operators,
    const VulkanBuffer& pyramid_patches,
    const VulkanBuffer& image_patch,
    const VulkanBuffer& zero,
    const VulkanBuffer* forced_fov,
    std::uint32_t width,
    std::uint32_t height);

GpuInferenceOutput infer_gpu(
    VulkanContext& context,
    GpuModel& model,
    VulkanOperators& operators,
    const float* rgb_chw,
    std::uint32_t width,
    std::uint32_t height,
    float forced_fov_degrees = 0.0f);

}  // namespace depth_pro_native
