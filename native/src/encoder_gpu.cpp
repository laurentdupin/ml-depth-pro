#include "encoder_gpu.h"

#include "inferbridge/native_harness_environment.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
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

const VulkanBuffer& weight(
    const GpuModel& model, const std::string& name) {
    const GpuTensor& tensor = model.tensor(name);
    const VulkanBuffer& result = tensor.half_buffer.handle() != VK_NULL_HANDLE
        ? tensor.half_buffer : tensor.buffer;
    if (result.handle() == VK_NULL_HANDLE) {
        throw std::runtime_error("expected GPU weight: " + name);
    }
    return result;
}

void linear_model(
    GpuModel& model, VulkanOperators& operators,
    VulkanBuffer& output, const VulkanBuffer& input,
    const std::string& weight_name, const std::string& bias_name,
    std::uint32_t rows, std::uint32_t input_columns,
    std::uint32_t output_columns, bool gelu = false) {
    const GpuTensor& tensor = model.tensor(weight_name);
    if (model.uses_int8_weights() &&
        tensor.int8_buffer.handle() != VK_NULL_HANDLE) {
        operators.linear_int8(output, input, tensor.int8_buffer,
            tensor.int8_scales, fp32(model, bias_name), rows,
            input_columns, output_columns, gelu);
    } else {
        const bool half_weight =
            tensor.half_buffer.handle() != VK_NULL_HANDLE;
        operators.linear(output, input, weight(model, weight_name),
            fp32(model, bias_name), rows, input_columns, output_columns,
            gelu, model.linear_block16(), half_weight,
            model.linear_vectorized(), model.linear_vector_tile());
    }
}

}  // namespace

GpuEncoderOutput encoder_gpu(
    VulkanContext& context,
    GpuModel& model,
    VulkanOperators& operators,
    std::string_view prefix_view,
    const VulkanBuffer& image,
    std::uint32_t batches,
    std::vector<VulkanBuffer>* debug_blocks) {
    if (batches == 0) {
        throw std::invalid_argument("encoder batch cannot be zero");
    }
    const std::string prefix(prefix_view);
    const VkDeviceSize token_bytes =
        std::uint64_t(batches) * token_elements * sizeof(float);
    const std::uint32_t rows = batches * tokens;
    VulkanBuffer current = context.create_device_buffer(token_bytes);
    VulkanBuffer next = context.create_device_buffer(token_bytes);
    VulkanBuffer normalized = context.create_device_buffer(token_bytes);
    VulkanBuffer branch = context.create_device_buffer(token_bytes);
    VulkanBuffer hidden = context.create_device_buffer(token_bytes * 4);
    // QKV is dead before the MLP hidden projection is produced. Reuse the
    // larger allocation instead of retaining another 3x-token scratch buffer.
    const bool alias_qkv =
        inferbridge::native_harness::scratch_aliasing_enabled();
    VulkanBuffer qkv_storage = alias_qkv
        ? VulkanBuffer{}
        : context.create_device_buffer(token_bytes * 3);
    VulkanBuffer& qkv = alias_qkv ? hidden : qkv_storage;
    VulkanBuffer scores = context.create_device_buffer(
        std::uint64_t(batches) * heads * tokens * tokens *
        sizeof(float));

    context.batch([&] {
        operators.prepare_tokens16(
            current,
            image,
            fp32(model, prefix + "patch_embed.proj.weight"),
            fp32(model, prefix + "patch_embed.proj.bias"),
            fp32(model, prefix + "cls_token"),
            fp32(model, prefix + "pos_embed"),
            batches);
    });

    if (!model.linear_tuned() && model.uses_int8_weights()) {
        model.set_linear_tuning(false, false, 0);
    } else if (!model.linear_tuned()) {
        const std::string base = prefix + "blocks.0.";
        operators.layer_norm(
            normalized, current,
            fp32(model, base + "norm1.weight"),
            fp32(model, base + "norm1.bias"),
            rows, embedding, 1.0e-6f);
        struct Candidate {
            bool block16;
            bool vectorized;
            std::uint32_t vector_tile;
            std::array<double, 3> samples{};
        };
        std::array<Candidate, 6> candidates{{
            {false, false, 0, {}},
            {true, false, 0, {}},
            {false, true, 4, {}},
            {false, true, 8, {}},
            {false, true, 16, {}},
            {false, true, 24, {}},
        }};
        const auto run = [&](const Candidate& candidate) {
            if (candidate.vector_tile == 24 && !model.uses_half_weights()) {
                return std::numeric_limits<double>::max();
            }
            const auto start = std::chrono::steady_clock::now();
            operators.linear(
                qkv, normalized,
                weight(model, base + "attn.qkv.weight"),
                fp32(model, base + "attn.qkv.bias"),
                rows, embedding, embedding * 3,
                false, candidate.block16, model.uses_half_weights(),
                candidate.vectorized, candidate.vector_tile);
            return std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - start).count();
        };
        for (Candidate& candidate : candidates) {
            run(candidate);
        }
        for (std::size_t sample = 0; sample < 3; ++sample) {
            if ((sample & 1u) == 0) {
                for (Candidate& candidate : candidates) {
                    candidate.samples[sample] = run(candidate);
                }
            } else {
                for (auto candidate = candidates.rbegin();
                     candidate != candidates.rend();
                     ++candidate) {
                    candidate->samples[sample] = run(*candidate);
                }
            }
        }
        Candidate* best = &candidates[0];
        double best_time =
            std::numeric_limits<double>::max();
        for (Candidate& candidate : candidates) {
            std::sort(
                candidate.samples.begin(), candidate.samples.end());
            const double median = candidate.samples[1];
            if (median < best_time) {
                best = &candidate;
                best_time = median;
            }
        }
        model.set_linear_tuning(
            best->block16, best->vectorized,
            best->vector_tile);
    }

    GpuEncoderOutput output;
    for (std::uint32_t block = 0; block < 24; ++block) {
        const std::string base =
            prefix + "blocks." + std::to_string(block) + ".";
        context.batch([&] {
            operators.layer_norm(
                normalized, current,
                fp32(model, base + "norm1.weight"),
                fp32(model, base + "norm1.bias"),
                rows, embedding, 1.0e-6f);
            linear_model(model, operators, qkv, normalized,
                base + "attn.qkv.weight", base + "attn.qkv.bias",
                rows, embedding, embedding * 3);
            operators.attention_head64(
                branch, qkv, tokens, heads, &scores, false, batches);
            linear_model(model, operators, normalized, branch,
                base + "attn.proj.weight", base + "attn.proj.bias",
                rows, embedding, embedding);
            operators.add_scaled(
                next, current, normalized,
                fp32(model, base + "ls1.gamma"),
                static_cast<std::uint32_t>(
                    std::uint64_t(batches) * token_elements),
                embedding);
            std::swap(current, next);
            operators.layer_norm(
                normalized, current,
                fp32(model, base + "norm2.weight"),
                fp32(model, base + "norm2.bias"),
                rows, embedding, 1.0e-6f);
            linear_model(model, operators, hidden, normalized,
                base + "mlp.fc1.weight", base + "mlp.fc1.bias",
                rows, embedding, embedding * 4, true);
            linear_model(model, operators, branch, hidden,
                base + "mlp.fc2.weight", base + "mlp.fc2.bias",
                rows, embedding * 4, embedding);
            operators.add_scaled(
                next, current, branch,
                fp32(model, base + "ls2.gamma"),
                static_cast<std::uint32_t>(
                    std::uint64_t(batches) * token_elements),
                embedding);
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
            rows, embedding, 1.0e-6f);
    });
    return output;
}

}  // namespace depth_pro_native
