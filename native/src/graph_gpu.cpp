#include "graph_gpu.h"

#include "encoder_gpu.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace depth_pro_native {
namespace {

struct HostImage {
    std::uint32_t channels;
    std::uint32_t height;
    std::uint32_t width;
    std::vector<float> values;
};

struct Feature {
    VulkanBuffer buffer;
    std::uint32_t channels;
    std::uint32_t height;
    std::uint32_t width;
};

std::uint64_t elements(
    std::uint32_t channels,
    std::uint32_t height,
    std::uint32_t width) {
    return std::uint64_t(channels) * height * width;
}

HostImage resize_host(
    const HostImage& input,
    std::uint32_t output_height,
    std::uint32_t output_width) {
    HostImage output{
        input.channels, output_height, output_width,
        std::vector<float>(
            static_cast<std::size_t>(
                elements(input.channels, output_height, output_width)))};
    for (std::uint32_t c = 0; c < input.channels; ++c) {
        for (std::uint32_t oy = 0; oy < output_height; ++oy) {
            const float raw_y =
                (static_cast<float>(oy) + 0.5f) *
                    input.height / output_height - 0.5f;
            const float sy = std::clamp(
                raw_y, 0.0f, static_cast<float>(input.height - 1));
            const std::uint32_t y0 = static_cast<std::uint32_t>(sy);
            const std::uint32_t y1 = std::min(y0 + 1, input.height - 1);
            const float wy = sy - y0;
            for (std::uint32_t ox = 0; ox < output_width; ++ox) {
                const float raw_x =
                    (static_cast<float>(ox) + 0.5f) *
                        input.width / output_width - 0.5f;
                const float sx = std::clamp(
                    raw_x, 0.0f, static_cast<float>(input.width - 1));
                const std::uint32_t x0 = static_cast<std::uint32_t>(sx);
                const std::uint32_t x1 =
                    std::min(x0 + 1, input.width - 1);
                const float wx = sx - x0;
                const auto at = [&](std::uint32_t y, std::uint32_t x) {
                    return input.values[
                        (std::uint64_t(c) * input.height + y) *
                            input.width + x];
                };
                output.values[
                    (std::uint64_t(c) * output_height + oy) *
                        output_width + ox] =
                    (at(y0, x0) * (1.0f - wx) + at(y0, x1) * wx) *
                        (1.0f - wy) +
                    (at(y1, x0) * (1.0f - wx) + at(y1, x1) * wx) * wy;
            }
        }
    }
    return output;
}

std::vector<float> crop_host(
    const HostImage& input,
    std::uint32_t top,
    std::uint32_t left) {
    std::vector<float> output(std::uint64_t(3) * 384 * 384);
    for (std::uint32_t c = 0; c < 3; ++c) {
        for (std::uint32_t y = 0; y < 384; ++y) {
            std::copy_n(
                input.values.data() +
                    (std::uint64_t(c) * input.height + top + y) *
                        input.width + left,
                384,
                output.data() +
                    (std::uint64_t(c) * 384 + y) * 384);
        }
    }
    return output;
}

const VulkanBuffer& weight(
    const GpuModel& model, const std::string& name) {
    const VulkanBuffer& result = model.tensor(name).half_buffer;
    if (result.handle() == VK_NULL_HANDLE) {
        throw std::runtime_error("expected FP16 GPU weight: " + name);
    }
    return result;
}

const VulkanBuffer& value(
    const GpuModel& model, const std::string& name) {
    const VulkanBuffer& result = model.tensor(name).buffer;
    if (result.handle() == VK_NULL_HANDLE) {
        throw std::runtime_error("expected FP32 GPU tensor: " + name);
    }
    return result;
}

Feature conv(
    VulkanContext& context,
    GpuModel& model,
    VulkanOperators& operators,
    const VulkanBuffer& zero,
    const Feature& input,
    const std::string& weight_name,
    const std::string& bias_name,
    std::uint32_t stride,
    std::uint32_t padding) {
    const GpuTensor& shape = model.tensor(weight_name);
    if (shape.rank != 4 || shape.dimensions[1] != input.channels) {
        throw std::runtime_error("GPU convolution shape mismatch: " + weight_name);
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(shape.dimensions[0]);
    const std::uint32_t kernel =
        static_cast<std::uint32_t>(shape.dimensions[2]);
    const std::uint32_t output_height =
        (input.height + 2 * padding - kernel) / stride + 1;
    const std::uint32_t output_width =
        (input.width + 2 * padding - kernel) / stride + 1;
    Feature output{
        context.create_device_buffer(
            elements(output_channels, output_height, output_width) *
            sizeof(float)),
        output_channels, output_height, output_width};
    operators.conv2d(
        output.buffer, input.buffer, weight(model, weight_name),
        bias_name.empty() ? zero : value(model, bias_name),
        input.width, input.height, input.channels, output_channels,
        kernel, stride, padding, !bias_name.empty(), true, true);
    return output;
}

Feature deconv(
    VulkanContext& context,
    GpuModel& model,
    VulkanOperators& operators,
    const VulkanBuffer& zero,
    const Feature& input,
    const std::string& weight_name,
    const std::string& bias_name) {
    const GpuTensor& shape = model.tensor(weight_name);
    if (shape.rank != 4 || shape.dimensions[0] != input.channels) {
        throw std::runtime_error("GPU deconvolution shape mismatch: " + weight_name);
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(shape.dimensions[1]);
    const std::uint32_t kernel =
        static_cast<std::uint32_t>(shape.dimensions[2]);
    Feature output{
        context.create_device_buffer(
            elements(
                output_channels, input.height * kernel,
                input.width * kernel) * sizeof(float)),
        output_channels, input.height * kernel, input.width * kernel};
    operators.conv_transpose_nonoverlap(
        output.buffer, input.buffer, weight(model, weight_name),
        bias_name.empty() ? zero : value(model, bias_name),
        input.width, input.height, input.channels, output_channels,
        kernel, true);
    return output;
}

void relu(VulkanOperators& operators, Feature& feature) {
    operators.relu(
        feature.buffer, feature.buffer,
        static_cast<std::uint32_t>(
            elements(feature.channels, feature.height, feature.width)));
}

Feature residual(
    VulkanContext& context,
    GpuModel& model,
    VulkanOperators& operators,
    const VulkanBuffer& zero,
    const Feature& input,
    const std::string& base) {
    Feature branch{
        context.create_device_buffer(
            elements(input.channels, input.height, input.width) *
            sizeof(float)),
        input.channels, input.height, input.width};
    operators.relu(
        branch.buffer, input.buffer,
        static_cast<std::uint32_t>(
            elements(input.channels, input.height, input.width)));
    branch = conv(
        context, model, operators, zero, branch,
        base + ".1.weight", base + ".1.bias", 1, 1);
    relu(operators, branch);
    branch = conv(
        context, model, operators, zero, branch,
        base + ".3.weight", base + ".3.bias", 1, 1);
    operators.add(
        branch.buffer, branch.buffer, input.buffer,
        static_cast<std::uint32_t>(
            elements(input.channels, input.height, input.width)));
    return branch;
}

Feature fusion(
    VulkanContext& context,
    GpuModel& model,
    VulkanOperators& operators,
    const VulkanBuffer& zero,
    Feature path,
    const Feature* skip,
    std::uint32_t level) {
    const std::string base =
        "decoder.fusions." + std::to_string(level);
    if (skip) {
        Feature processed = residual(
            context, model, operators, zero, *skip,
            base + ".resnet1.residual");
        operators.add(
            path.buffer, path.buffer, processed.buffer,
            static_cast<std::uint32_t>(
                elements(path.channels, path.height, path.width)));
    }
    path = residual(
        context, model, operators, zero, path,
        base + ".resnet2.residual");
    if (level != 0) {
        path = deconv(
            context, model, operators, zero, path,
            base + ".deconv.weight", "");
    }
    return conv(
        context, model, operators, zero, path,
        base + ".out_conv.weight", base + ".out_conv.bias", 1, 0);
}

Feature project_upsample(
    VulkanContext& context,
    GpuModel& model,
    VulkanOperators& operators,
    const VulkanBuffer& zero,
    Feature image,
    const std::string& base,
    std::uint32_t layers) {
    image = conv(
        context, model, operators, zero, image,
        base + ".0.weight", "", 1, 0);
    for (std::uint32_t index = 0; index < layers; ++index) {
        image = deconv(
            context, model, operators, zero, image,
            base + "." + std::to_string(index + 1) + ".weight", "");
    }
    return image;
}

Feature token_image(
    VulkanContext& context,
    VulkanOperators& operators,
    const VulkanBuffer& tokens,
    std::uint32_t channels,
    std::uint32_t batch_index = 0) {
    Feature result{
        context.create_device_buffer(
            std::uint64_t(channels) * 576 * sizeof(float)),
        channels, 24, 24};
    operators.tokens_to_nchw(
        result.buffer, tokens, channels, batch_index);
    return result;
}

}  // namespace

GpuInferenceOutput infer_gpu(
    VulkanContext& context,
    GpuModel& model,
    VulkanOperators& operators,
    const float* rgb,
    std::uint32_t width,
    std::uint32_t height,
    float forced_fov_degrees) {
    if (!rgb || width == 0 || height == 0) {
        throw std::invalid_argument("invalid Depth Pro GPU image");
    }
    HostImage input{3, height, width, std::vector<float>(
        static_cast<std::size_t>(elements(3, height, width)))};
    for (std::size_t index = 0; index < input.values.size(); ++index) {
        input.values[index] = (rgb[index] - 0.5f) / 0.5f;
    }
    HostImage x0 = resize_host(input, 1536, 1536);
    HostImage x1 = resize_host(x0, 768, 768);
    HostImage x2 = resize_host(x0, 384, 384);

    VulkanBuffer zero =
        context.create_device_buffer(1024 * sizeof(float));
    const std::vector<float> zero_values(1024, 0.0f);
    context.upload(
        zero, zero_values.data(),
        zero_values.size() * sizeof(float));
    Feature latent0{
        context.create_device_buffer(
            elements(1024, 96, 96) * sizeof(float)), 1024, 96, 96};
    Feature latent1{
        context.create_device_buffer(
            elements(1024, 96, 96) * sizeof(float)), 1024, 96, 96};
    Feature feature0{
        context.create_device_buffer(
            elements(1024, 96, 96) * sizeof(float)), 1024, 96, 96};
    Feature feature1{
        context.create_device_buffer(
            elements(1024, 48, 48) * sizeof(float)), 1024, 48, 48};

    constexpr std::uint32_t pyramid_patch_count = 35;
    constexpr VkDeviceSize large_memory_threshold =
        VkDeviceSize{10} * 1024 * 1024 * 1024;
    const std::uint32_t encoder_batch_limit =
        context.device_local_memory_bytes() >=
            large_memory_threshold
        ? 35u : 14u;
    constexpr std::uint64_t patch_elements =
        std::uint64_t(3) * 384 * 384;
    std::vector<float> pyramid_patches(
        pyramid_patch_count * patch_elements);
    for (std::uint32_t j = 0; j < 5; ++j) {
        for (std::uint32_t i = 0; i < 5; ++i) {
            const std::uint32_t index = j * 5 + i;
            const std::vector<float> pixels =
                crop_host(x0, j * 288, i * 288);
            std::copy(
                pixels.begin(), pixels.end(),
                pyramid_patches.begin() + index * patch_elements);
        }
    }
    for (std::uint32_t j = 0; j < 3; ++j) {
        for (std::uint32_t i = 0; i < 3; ++i) {
            const std::uint32_t index = 25 + j * 3 + i;
            const std::vector<float> pixels =
                crop_host(x1, j * 192, i * 192);
            std::copy(
                pixels.begin(), pixels.end(),
                pyramid_patches.begin() + index * patch_elements);
        }
    }
    std::copy(
        x2.values.begin(), x2.values.end(),
        pyramid_patches.begin() + 34 * patch_elements);

    Feature feature2{
        context.create_device_buffer(
            elements(1024, 24, 24) * sizeof(float)),
        1024, 24, 24};
    Feature patch_image{
        context.create_device_buffer(
            elements(1024, 24, 24) * sizeof(float)),
        1024, 24, 24};
    for (std::uint32_t batch_start = 0;
         batch_start < pyramid_patch_count;
         batch_start += encoder_batch_limit) {
        const std::uint32_t batch_count = std::min(
            encoder_batch_limit,
            pyramid_patch_count - batch_start);
        VulkanBuffer patch_batch = context.create_device_buffer(
            std::uint64_t(batch_count) * patch_elements *
            sizeof(float));
        context.upload(
            patch_batch,
            pyramid_patches.data() + batch_start * patch_elements,
            std::uint64_t(batch_count) * patch_elements *
                sizeof(float));
        GpuEncoderOutput encoded = encoder_gpu(
            context, model, operators,
            "encoder.patch_encoder.", patch_batch, batch_count);
        context.batch([&] {
            for (std::uint32_t local_index = 0;
                 local_index < batch_count;
                 ++local_index) {
                const std::uint32_t index =
                    batch_start + local_index;
                if (index < 25) {
                    operators.tokens_to_nchw(
                        patch_image.buffer, encoded.capture5,
                        1024, local_index);
                    operators.merge_patch(
                        latent0.buffer, patch_image.buffer,
                        1024, 5, 3, index);
                    operators.tokens_to_nchw(
                        patch_image.buffer, encoded.capture11,
                        1024, local_index);
                    operators.merge_patch(
                        latent1.buffer, patch_image.buffer,
                        1024, 5, 3, index);
                    operators.tokens_to_nchw(
                        patch_image.buffer, encoded.final,
                        1024, local_index);
                    operators.merge_patch(
                        feature0.buffer, patch_image.buffer,
                        1024, 5, 3, index);
                } else if (index < 34) {
                    operators.tokens_to_nchw(
                        patch_image.buffer, encoded.final,
                        1024, local_index);
                    operators.merge_patch(
                        feature1.buffer, patch_image.buffer,
                        1024, 3, 6, index - 25);
                } else {
                    operators.tokens_to_nchw(
                        feature2.buffer, encoded.final,
                        1024, local_index);
                }
            }
        });
    }

    VulkanBuffer patch = context.create_device_buffer(
        patch_elements * sizeof(float));
    context.upload(
        patch, x2.values.data(), x2.values.size() * sizeof(float));
    GpuEncoderOutput image_encoded = encoder_gpu(
        context, model, operators, "encoder.image_encoder.", patch);
    Feature global =
        token_image(context, operators, image_encoded.final, 1024);

    latent0 = project_upsample(
        context, model, operators, zero, std::move(latent0),
        "encoder.upsample_latent0", 3);
    latent1 = project_upsample(
        context, model, operators, zero, std::move(latent1),
        "encoder.upsample_latent1", 2);
    feature0 = project_upsample(
        context, model, operators, zero, std::move(feature0),
        "encoder.upsample0", 1);
    feature1 = project_upsample(
        context, model, operators, zero, std::move(feature1),
        "encoder.upsample1", 1);
    feature2 = project_upsample(
        context, model, operators, zero, std::move(feature2),
        "encoder.upsample2", 1);
    global = deconv(
        context, model, operators, zero, global,
        "encoder.upsample_lowres.weight",
        "encoder.upsample_lowres.bias");
    Feature joined{
        context.create_device_buffer(
            elements(2048, 48, 48) * sizeof(float)),
        2048, 48, 48};
    const std::uint32_t feature_count =
        static_cast<std::uint32_t>(elements(1024, 48, 48));
    operators.concatenate(
        joined.buffer, feature2.buffer, global.buffer,
        feature_count, feature_count);
    global = conv(
        context, model, operators, zero, joined,
        "encoder.fuse_lowres.weight",
        "encoder.fuse_lowres.bias", 1, 0);

    Feature path;
    Feature lowres;
    path = conv(
        context, model, operators, zero, global,
        "decoder.convs.4.weight", "", 1, 1);
    lowres = Feature{
        context.create_device_buffer(
            elements(256, 48, 48) * sizeof(float)), 256, 48, 48};
    context.copy(
        lowres.buffer, 0, path.buffer, 0,
        elements(256, 48, 48) * sizeof(float));
    path = fusion(
        context, model, operators, zero,
        std::move(path), nullptr, 4);
    Feature projected = conv(
        context, model, operators, zero, feature1,
        "decoder.convs.3.weight", "", 1, 1);
    path = fusion(
        context, model, operators, zero,
        std::move(path), &projected, 3);
    projected = conv(
        context, model, operators, zero, feature0,
        "decoder.convs.2.weight", "", 1, 1);
    path = fusion(
        context, model, operators, zero,
        std::move(path), &projected, 2);
    projected = conv(
        context, model, operators, zero, latent1,
        "decoder.convs.1.weight", "", 1, 1);
    path = fusion(
        context, model, operators, zero,
        std::move(path), &projected, 1);
    path = fusion(
        context, model, operators, zero,
        std::move(path), &latent0, 0);
    path = conv(
        context, model, operators, zero, path,
        "head.0.weight", "head.0.bias", 1, 1);
    path = deconv(
        context, model, operators, zero, path,
        "head.1.weight", "head.1.bias");
    path = conv(
        context, model, operators, zero, path,
        "head.2.weight", "head.2.bias", 1, 1);
    relu(operators, path);
    path = conv(
        context, model, operators, zero, path,
        "head.4.weight", "head.4.bias", 1, 0);
    relu(operators, path);

    Feature fov;
    if (forced_fov_degrees > 0.0f &&
        forced_fov_degrees < 180.0f) {
        fov = Feature{
            context.create_device_buffer(sizeof(float)), 1, 1, 1};
        context.upload(
            fov.buffer, &forced_fov_degrees, sizeof(float));
    } else {
    GpuEncoderOutput fov_encoded = encoder_gpu(
        context, model, operators, "fov.encoder.0.", patch);
    Feature fov_tokens{
        context.create_device_buffer(
            elements(128, 24, 24) * sizeof(float)), 128, 24, 24};
    VulkanBuffer fov_projected = context.create_device_buffer(
        std::uint64_t(577) * 128 * sizeof(float));
    operators.linear(
        fov_projected, fov_encoded.final,
        weight(model, "fov.encoder.1.weight"),
        value(model, "fov.encoder.1.bias"),
        577, 1024, 128, false, true, true);
    operators.tokens_to_nchw(
        fov_tokens.buffer, fov_projected, 128);
    fov = conv(
        context, model, operators, zero, lowres,
        "fov.downsample.0.weight",
        "fov.downsample.0.bias", 2, 1);
    relu(operators, fov);
    operators.add(
        fov.buffer, fov.buffer, fov_tokens.buffer,
        static_cast<std::uint32_t>(elements(128, 24, 24)));
    fov = conv(
        context, model, operators, zero, fov,
        "fov.head.0.weight", "fov.head.0.bias", 2, 1);
    relu(operators, fov);
    fov = conv(
        context, model, operators, zero, fov,
        "fov.head.2.weight", "fov.head.2.bias", 2, 1);
    relu(operators, fov);
    fov = conv(
        context, model, operators, zero, fov,
        "fov.head.4.weight", "fov.head.4.bias", 1, 0);
    }

    Feature scaled{
        context.create_device_buffer(
            elements(1, 1536, 1536) * sizeof(float)), 1, 1536, 1536};
    Feature resized{
        context.create_device_buffer(
            elements(1, height, width) * sizeof(float)), 1, height, width};
    Feature depth{
        context.create_device_buffer(
            elements(1, height, width) * sizeof(float)), 1, height, width};
    context.batch([&] {
        operators.scale_inverse(
            scaled.buffer, path.buffer, fov.buffer, 1536 * 1536);
        operators.bilinear_half_pixel(
            resized.buffer, scaled.buffer,
            1536, 1536, width, height, 1);
        operators.reciprocal_depth(
            depth.buffer, resized.buffer, width * height);
    });
    GpuInferenceOutput output;
    output.depth.resize(std::uint64_t(width) * height);
    context.download(
        depth.buffer, output.depth.data(),
        output.depth.size() * sizeof(float));
    float fov_degrees = 0.0f;
    context.download(fov.buffer, &fov_degrees, sizeof(fov_degrees));
    output.focal_length_pixels =
        0.5f * width /
        std::tan(
            0.5f * fov_degrees *
            3.14159265358979323846f / 180.0f);
    return output;
}

}  // namespace depth_pro_native
