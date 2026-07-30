#pragma once

#include "gpu_model.h"
#include "operators.h"

#include <string_view>
#include <vector>

namespace depth_pro_native {

struct GpuEncoderOutput {
    VulkanBuffer capture5;
    VulkanBuffer capture11;
    VulkanBuffer final;
};

GpuEncoderOutput encoder_gpu(
    VulkanContext& context,
    GpuModel& model,
    VulkanOperators& operators,
    std::string_view prefix,
    const VulkanBuffer& image,
    std::vector<VulkanBuffer>* debug_blocks = nullptr);

}  // namespace depth_pro_native
