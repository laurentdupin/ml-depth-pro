#include "encoder_gpu.h"
#include "gpu_model.h"
#include "model.h"
#include "operators.h"
#include "vulkan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
std::vector<float> read(const std::string& path, std::uint64_t count) {
    std::vector<float> result(static_cast<std::size_t>(count));
    std::ifstream input(path, std::ios::binary);
    input.read(
        reinterpret_cast<char*>(result.data()),
        static_cast<std::streamsize>(count * sizeof(float)));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("invalid tensor file: " + path);
    }
    return result;
}

void compare(
    const std::vector<float>& output,
    const std::vector<float>& reference,
    const char* name) {
    double difference_sum = 0.0;
    double reference_sum = 0.0;
    float maximum = 0.0f;
    for (std::size_t index = 0; index < output.size(); ++index) {
        const float difference = std::abs(output[index] - reference[index]);
        difference_sum += difference;
        reference_sum += std::abs(reference[index]);
        maximum = std::max(maximum, difference);
    }
    std::cout << name << "_relative_l1="
              << difference_sum / reference_sum
              << "\n" << name << "_maximum_absolute=" << maximum << "\n";
}
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: probe model device input reference-dir\n";
        return 2;
    }
    try {
        constexpr std::uint64_t count = std::uint64_t(577) * 1024;
        const std::vector<float> input =
            read(argv[3], std::uint64_t(3) * 384 * 384);
        depth_pro_native::ModelFile file(argv[1]);
        depth_pro_native::VulkanContext context(
            static_cast<std::uint32_t>(std::stoul(argv[2])));
        depth_pro_native::GpuModel model(file, context);
        depth_pro_native::VulkanOperators operators(context);
        depth_pro_native::VulkanBuffer image =
            context.create_device_buffer(input.size() * sizeof(float));
        context.upload(image, input.data(), input.size() * sizeof(float));
        depth_pro_native::VulkanBuffer prepared =
            context.create_device_buffer(count * sizeof(float));
        operators.prepare_tokens16(
            prepared, image,
            model.tensor(
                "encoder.patch_encoder.patch_embed.proj.weight").buffer,
            model.tensor(
                "encoder.patch_encoder.patch_embed.proj.bias").buffer,
            model.tensor("encoder.patch_encoder.cls_token").buffer,
            model.tensor("encoder.patch_encoder.pos_embed").buffer);
        std::vector<float> prepared_values(count);
        context.download(
            prepared, prepared_values.data(),
            prepared_values.size() * sizeof(float));
        const auto& patch_weight = file.tensor(
            "encoder.patch_encoder.patch_embed.proj.weight");
        const auto& patch_bias = file.tensor(
            "encoder.patch_encoder.patch_embed.proj.bias");
        const auto& class_token =
            file.tensor("encoder.patch_encoder.cls_token");
        const auto& position =
            file.tensor("encoder.patch_encoder.pos_embed");
        double prepare_difference = 0.0;
        double prepare_reference = 0.0;
        for (std::uint32_t token = 0; token < 577; token += 37) {
            for (std::uint32_t feature = 0;
                 feature < 1024; feature += 61) {
                float expected;
                if (token == 0) {
                    expected = depth_pro_native::half_to_float(
                        class_token.data[feature]);
                } else {
                    const std::uint32_t patch_id = token - 1;
                    const std::uint32_t px = patch_id % 24;
                    const std::uint32_t py = patch_id / 24;
                    expected = depth_pro_native::half_to_float(
                        patch_bias.data[feature]);
                    for (std::uint32_t channel = 0; channel < 3; ++channel) {
                        for (std::uint32_t ky = 0; ky < 16; ++ky) {
                            for (std::uint32_t kx = 0; kx < 16; ++kx) {
                                expected += input[
                                    (std::uint64_t(channel) * 384 +
                                     py * 16 + ky) * 384 + px * 16 + kx] *
                                    depth_pro_native::half_to_float(
                                        patch_weight.data[
                                            ((std::uint64_t(feature) * 3 +
                                              channel) * 16 + ky) * 16 +
                                            kx]);
                            }
                        }
                    }
                }
                expected += depth_pro_native::half_to_float(
                    position.data[
                        std::uint64_t(token) * 1024 + feature]);
                prepare_difference += std::abs(
                    expected - prepared_values[
                        std::uint64_t(token) * 1024 + feature]);
                prepare_reference += std::abs(expected);
            }
        }
        std::cout << "prepare_sample_relative_l1="
                  << prepare_difference / prepare_reference << "\n";
        const auto fp32 = [&](const std::string& name)
            -> const depth_pro_native::VulkanBuffer& {
            return model.tensor(name).buffer;
        };
        const auto fp16 = [&](const std::string& name)
            -> const depth_pro_native::VulkanBuffer& {
            return model.tensor(name).half_buffer;
        };
        const std::string base =
            "encoder.patch_encoder.blocks.0.";
        depth_pro_native::VulkanBuffer normalized =
            context.create_device_buffer(count * sizeof(float));
        operators.layer_norm(
            normalized, prepared,
            fp32(base + "norm1.weight"),
            fp32(base + "norm1.bias"), 577, 1024, 1.0e-6f);
        std::vector<float> debug_values(count);
        context.download(
            normalized, debug_values.data(), count * sizeof(float));
        compare(
            debug_values,
            read(
                std::string(argv[4]) +
                "\\..\\build-vulkan\\encoder-debug\\norm1.bin", count),
            "norm1");
        depth_pro_native::VulkanBuffer qkv =
            context.create_device_buffer(count * 3 * sizeof(float));
        operators.linear(
            qkv, normalized,
            fp16(base + "attn.qkv.weight"),
            fp32(base + "attn.qkv.bias"),
            577, 1024, 3072, false, false, true);
        std::vector<float> qkv_values(count * 3);
        context.download(
            qkv, qkv_values.data(), qkv_values.size() * sizeof(float));
        compare(
            qkv_values,
            read(
                std::string(argv[4]) +
                "\\..\\build-vulkan\\encoder-debug\\qkv.bin", count * 3),
            "qkv");
        depth_pro_native::VulkanBuffer attended =
            context.create_device_buffer(count * sizeof(float));
        depth_pro_native::VulkanBuffer scores =
            context.create_device_buffer(
                std::uint64_t(16) * 577 * 577 * sizeof(float));
        operators.attention_head64(
            attended, qkv, 577, 16, &scores, false);
        depth_pro_native::VulkanBuffer projected =
            context.create_device_buffer(count * sizeof(float));
        operators.linear(
            projected, attended,
            fp16(base + "attn.proj.weight"),
            fp32(base + "attn.proj.bias"),
            577, 1024, 1024, false, false, true);
        context.download(
            projected, debug_values.data(), count * sizeof(float));
        compare(
            debug_values,
            read(
                std::string(argv[4]) +
                "\\..\\build-vulkan\\encoder-debug\\attention.bin", count),
            "attention");
        depth_pro_native::VulkanBuffer state =
            context.create_device_buffer(count * sizeof(float));
        operators.add_scaled(
            state, prepared, projected,
            fp32(base + "ls1.gamma"),
            static_cast<std::uint32_t>(count), 1024);
        operators.layer_norm(
            normalized, state,
            fp32(base + "norm2.weight"),
            fp32(base + "norm2.bias"), 577, 1024, 1.0e-6f);
        context.download(
            normalized, debug_values.data(), count * sizeof(float));
        compare(
            debug_values,
            read(
                std::string(argv[4]) +
                "\\..\\build-vulkan\\encoder-debug\\norm2.bin", count),
            "norm2");
        depth_pro_native::VulkanBuffer hidden =
            context.create_device_buffer(count * 4 * sizeof(float));
        operators.linear(
            hidden, normalized,
            fp16(base + "mlp.fc1.weight"),
            fp32(base + "mlp.fc1.bias"),
            577, 1024, 4096, true, false, true);
        operators.linear(
            projected, hidden,
            fp16(base + "mlp.fc2.weight"),
            fp32(base + "mlp.fc2.bias"),
            577, 4096, 1024, false, false, true);
        operators.add_scaled(
            state, state, projected,
            fp32(base + "ls2.gamma"),
            static_cast<std::uint32_t>(count), 1024);
        context.download(
            state, debug_values.data(), count * sizeof(float));
        compare(
            debug_values,
            read(
                std::string(argv[4]) +
                "\\..\\build-vulkan\\encoder-debug\\block0.bin", count),
            "block0");
        std::vector<depth_pro_native::VulkanBuffer> debug_blocks;
        depth_pro_native::GpuEncoderOutput encoded =
            depth_pro_native::encoder_gpu(
                context, model, operators,
                "encoder.patch_encoder.", image, 1, &debug_blocks);
        for (std::uint32_t block_index = 0;
             block_index < debug_blocks.size(); ++block_index) {
            context.download(
                debug_blocks[block_index], debug_values.data(),
                count * sizeof(float));
            const std::string name =
                "block" + std::to_string(block_index);
            compare(
                debug_values,
                read(
                    std::string(argv[4]) +
                    "\\..\\build-vulkan\\encoder-debug\\" +
                    name + ".bin", count),
                name.c_str());
        }
        const struct {
            const char* name;
            const depth_pro_native::VulkanBuffer* buffer;
        } outputs[] = {
            {"block5", &encoded.capture5},
            {"block11", &encoded.capture11},
            {"block23", &encoded.final},
        };
        for (const auto& entry : outputs) {
            std::vector<float> actual(count);
            context.download(
                *entry.buffer, actual.data(), actual.size() * sizeof(float));
            compare(
                actual,
                read(
                    std::string(argv[4]) + "\\patch384." +
                    entry.name + ".bin", count),
                entry.name);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
