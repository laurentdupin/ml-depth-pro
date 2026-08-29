#pragma once

#include "model.h"
#include "vulkan.h"

#include <array>
#include <cstdint>
#include <string_view>
#include <unordered_map>

namespace depth_pro_native {

struct GpuTensor {
    VulkanBuffer buffer;
    VulkanBuffer half_buffer;
    VulkanBuffer int8_buffer;
    VulkanBuffer int8_scales;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint32_t rank = 0;
    std::uint64_t elements = 0;
};

class GpuModel {
public:
    GpuModel(const ModelFile& model, VulkanContext& context);

    const GpuTensor& tensor(std::string_view name) const;
    bool uses_half_weights() const { return uses_half_weights_; }
    bool uses_int8_weights() const { return uses_int8_weights_; }
    std::size_t tensor_count() const { return tensors_.size(); }
    bool linear_tuned() const { return linear_tuned_; }
    bool linear_block16() const { return linear_block16_; }
    bool linear_vectorized() const { return linear_vectorized_; }
    std::uint32_t linear_vector_tile() const {
        return linear_vector_tile_;
    }
    void set_linear_tuning(
        bool block16,
        bool vectorized,
        std::uint32_t vector_tile) {
        linear_block16_ = block16;
        linear_vectorized_ = vectorized;
        linear_vector_tile_ = vector_tile;
        linear_tuned_ = true;
    }

private:
    std::unordered_map<std::string_view, GpuTensor> tensors_;
    bool uses_half_weights_ = false;
    bool uses_int8_weights_ = false;
    bool linear_tuned_ = false;
    bool linear_block16_ = false;
    bool linear_vectorized_ = false;
    std::uint32_t linear_vector_tile_ = 0;
};

}  // namespace depth_pro_native
