#include "encoder_gpu.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace depth_pro_native {
namespace {

constexpr std::uint32_t tokens = 577;
constexpr std::uint32_t embedding = 1024;
constexpr std::uint32_t heads = 16;
constexpr std::uint64_t token_elements =
    std::uint64_t(tokens) * embedding;

const VulkanBuffer& fp32(
    const GpuModel& model, const std::string& name) {
    const VulkanBuffer& result = model.tensor(name).buffer;
    if (result.handle() == VK_NULL_HANDLE) {
        throw std::runtime_error("expected FP32 GPU tensor: " + name);
    }
    return result;
}

const VulkanBuffer& fp16(
    const GpuModel& model, const std::string& name) {
    const VulkanBuffer& result = model.tensor(name).half_buffer;
    if (result.handle() == VK_NULL_HANDLE) {
        throw std::runtime_error("expected FP16 GPU tensor: " + name);
    }
    return result;
}

}  // namespace

GpuEncoderOutput encoder_gpu(
    VulkanContext& context,
    GpuModel& model,
    VulkanOperators& operators,
    std::string_view prefix_view,
    const VulkanBuffer& image,
    std::vector<VulkanBuffer>* debug_blocks) {
    const std::string prefix(prefix_view);
    const VkDeviceSize token_bytes =
        token_elements * sizeof(float);
    VulkanBuffer current = context.create_device_buffer(token_bytes);
    VulkanBuffer next = context.create_device_buffer(token_bytes);
    VulkanBuffer normalized = context.create_device_buffer(token_bytes);
    VulkanBuffer branch = context.create_device_buffer(token_bytes);
    VulkanBuffer qkv = context.create_device_buffer(token_bytes * 3);
    VulkanBuffer hidden = context.create_device_buffer(token_bytes * 4);
    VulkanBuffer scores = context.create_device_buffer(
        std::uint64_t(heads) * tokens * tokens * sizeof(float));

    context.batch([&] {
        operators.prepare_tokens16(
            current,
            image,
            fp32(model, prefix + "patch_embed.proj.weight"),
            fp32(model, prefix + "patch_embed.proj.bias"),
            fp32(model, prefix + "cls_token"),
            fp32(model, prefix + "pos_embed"));
    });

    GpuEncoderOutput output;
    for (std::uint32_t block = 0; block < 24; ++block) {
        const std::string base =
            prefix + "blocks." + std::to_string(block) + ".";
        context.batch([&] {
            operators.layer_norm(
                normalized, current,
                fp32(model, base + "norm1.weight"),
                fp32(model, base + "norm1.bias"),
                tokens, embedding, 1.0e-6f);
            operators.linear(
                qkv, normalized,
                fp16(model, base + "attn.qkv.weight"),
                fp32(model, base + "attn.qkv.bias"),
                tokens, embedding, embedding * 3,
                false, false, true);
            operators.attention_head64(
                branch, qkv, tokens, heads, &scores, false);
            operators.linear(
                normalized, branch,
                fp16(model, base + "attn.proj.weight"),
                fp32(model, base + "attn.proj.bias"),
                tokens, embedding, embedding,
                false, false, true);
            operators.add_scaled(
                next, current, normalized,
                fp32(model, base + "ls1.gamma"),
                static_cast<std::uint32_t>(token_elements), embedding);
            std::swap(current, next);
            operators.layer_norm(
                normalized, current,
                fp32(model, base + "norm2.weight"),
                fp32(model, base + "norm2.bias"),
                tokens, embedding, 1.0e-6f);
            operators.linear(
                hidden, normalized,
                fp16(model, base + "mlp.fc1.weight"),
                fp32(model, base + "mlp.fc1.bias"),
                tokens, embedding, embedding * 4,
                true, false, true);
            operators.linear(
                branch, hidden,
                fp16(model, base + "mlp.fc2.weight"),
                fp32(model, base + "mlp.fc2.bias"),
                tokens, embedding * 4, embedding,
                false, false, true);
            operators.add_scaled(
                next, current, branch,
                fp32(model, base + "ls2.gamma"),
                static_cast<std::uint32_t>(token_elements), embedding);
            std::swap(current, next);
        });
        if (block == 5) {
            output.capture5 =
                context.create_device_buffer(token_bytes);
            context.copy(
                output.capture5, 0, current, 0, token_bytes);
        } else if (block == 11) {
            output.capture11 =
                context.create_device_buffer(token_bytes);
            context.copy(
                output.capture11, 0, current, 0, token_bytes);
        }
        if (debug_blocks && block < 6) {
            VulkanBuffer debug =
                context.create_device_buffer(token_bytes);
            context.copy(debug, 0, current, 0, token_bytes);
            debug_blocks->push_back(std::move(debug));
        }
    }
    output.final = context.create_device_buffer(token_bytes);
    context.batch([&] {
        operators.layer_norm(
            output.final, current,
            fp32(model, prefix + "norm.weight"),
            fp32(model, prefix + "norm.bias"),
            tokens, embedding, 1.0e-6f);
    });
    return output;
}

}  // namespace depth_pro_native
