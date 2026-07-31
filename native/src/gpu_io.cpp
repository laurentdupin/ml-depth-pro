#include "gpu_io.h"

#include "preprocess_texture_spv.h"
#include "pyramid_patches_spv.h"
#include "write_depth_image_spv.h"

#include <stdexcept>

namespace depth_pro_native {

GpuIo::GpuIo(VulkanContext& context)
    : context_(context),
      preprocess_(context.create_pipeline(
          dpro_preprocess_texture_spv, dpro_preprocess_texture_spv_size,
          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT},
          2u * sizeof(std::uint32_t))),
      assemble_(context.create_pipeline(
          dpro_pyramid_patches_spv, dpro_pyramid_patches_spv_size,
          4, 0)),
      write_depth_(context.create_pipeline(
          dpro_write_depth_image_spv, dpro_write_depth_image_spv_size,
          {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
          {VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT},
          2u * sizeof(std::uint32_t))) {
    preprocess_.set_debug_name("depth_pro_preprocess_texture");
    assemble_.set_debug_name("depth_pro_pyramid_patches");
    write_depth_.set_debug_name("depth_pro_write_depth_image");
}

void GpuIo::preprocess_x0(
    VulkanBuffer& destination, const VulkanImage& source) {
    constexpr std::uint64_t required =
        std::uint64_t(3) * 1536 * 1536 * sizeof(float);
    if (source.width() == 0u || source.height() == 0u ||
        destination.size() < required) {
        throw std::invalid_argument("invalid Depth Pro GPU input shape");
    }
    const std::uint32_t parameters[2] = {
        source.width(), source.height()};
    context_.dispatch_image_to_buffer(
        preprocess_, source, destination, parameters, sizeof(parameters),
        192, 192);
}

void GpuIo::assemble_pyramid(
    VulkanBuffer& destination, const VulkanBuffer& x0,
    const VulkanBuffer& x1, const VulkanBuffer& x2) {
    constexpr std::uint64_t patch_elements =
        std::uint64_t(3) * 384 * 384;
    if (destination.size() < 35u * patch_elements * sizeof(float) ||
        x0.size() < std::uint64_t(3) * 1536 * 1536 * sizeof(float) ||
        x1.size() < std::uint64_t(3) * 768 * 768 * sizeof(float) ||
        x2.size() < patch_elements * sizeof(float)) {
        throw std::invalid_argument("invalid Depth Pro GPU pyramid shape");
    }
    context_.dispatch(
        assemble_, {&destination, &x0, &x1, &x2},
        nullptr, 0, 48, 48, 35);
}

void GpuIo::write_depth(
    VulkanImage& destination, const VulkanBuffer& depth) {
    if (destination.format() != VK_FORMAT_R32_SFLOAT ||
        destination.width() == 0u || destination.height() == 0u ||
        depth.size() < std::uint64_t(destination.width()) *
            destination.height() * sizeof(float)) {
        throw std::invalid_argument("invalid Depth Pro GPU output shape");
    }
    const std::uint32_t parameters[2] = {
        destination.width(), destination.height()};
    context_.dispatch_buffer_to_image(
        write_depth_, depth, destination, parameters, sizeof(parameters),
        (destination.width() + 7u) / 8u,
        (destination.height() + 7u) / 8u);
}

}  // namespace depth_pro_native
