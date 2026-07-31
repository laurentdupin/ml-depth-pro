#pragma once

#include "vulkan.h"

#include <cstdint>

namespace depth_pro_native {

class GpuIo {
public:
    explicit GpuIo(VulkanContext& context);
    void preprocess_x0(
        VulkanBuffer& destination, const VulkanImage& source);
    void assemble_pyramid(
        VulkanBuffer& destination, const VulkanBuffer& x0,
        const VulkanBuffer& x1, const VulkanBuffer& x2);
    void write_depth(
        VulkanImage& destination, const VulkanBuffer& depth);

private:
    VulkanContext& context_;
    VulkanPipeline preprocess_;
    VulkanPipeline assemble_;
    VulkanPipeline write_depth_;
};

}  // namespace depth_pro_native
