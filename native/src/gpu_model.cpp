#include "gpu_model.h"

#include <inferbridge/native_harness_environment.h>
#include <inferbridge/native_harness_precision.h>
#include <inferbridge/native_harness_vulkan_initialization.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace depth_pro_native {
namespace {

bool is_large_weight(std::string_view name) {
    constexpr std::string_view suffix = ".weight";
    if (name.size() < suffix.size() ||
        name.substr(name.size() - suffix.size()) != suffix) {
        return false;
    }
    return name.find(".norm") == std::string_view::npos &&
        name.find("patch_embed.proj.weight") == std::string_view::npos;
}

}  // namespace

GpuModel::GpuModel(
    const ModelFile& model, VulkanContext& context,
    bool load_fov_weights) {
    const auto precision = inferbridge::native::require_supported_precision(
        inferbridge::native::requested_precision(),
        {context.supports_float16(), context.supports_packed_int8_dot()},
        context.supports_float16() ? inferbridge::native::Precision::fp16
                                   : inferbridge::native::Precision::fp32);
    // DPROFMOD stores every source tensor as FP16. Expanding large weights to
    // FP32 cannot restore information, doubles their device footprint, and can
    // push the 1536px graph over the WDDM budget. The half-weight shaders
    // unpack to float and retain FP32 accumulation, so native storage is also
    // the lossless representation for FP32 compute.
    uses_half_weights_ = precision != inferbridge::native::Precision::int8 &&
        !inferbridge::native_harness::environment_flag_enabled(
            "DPRO_DISABLE_NATIVE_WEIGHT_STORAGE");
    uses_int8_weights_ = precision == inferbridge::native::Precision::int8;
    tensors_.reserve(model.tensor_count());
    std::uint64_t uploaded_bytes = 0u;
    std::size_t uploaded_buffers = 0u;
    const auto upload = [&](std::string_view name, VulkanBuffer& buffer,
                            const void* data, std::size_t bytes) {
        try {
            context.upload(buffer, data, bytes);
            uploaded_bytes += bytes;
            ++uploaded_buffers;
        } catch (const std::exception& error) {
            throw std::runtime_error(
                "failed to upload Depth Pro tensor " + std::string(name) +
                " (" + std::to_string(bytes) + " bytes after " +
                std::to_string(uploaded_bytes) + " successful bytes): " +
                std::to_string(uploaded_buffers) + " buffers: " + error.what());
        }
    };
    const std::vector<std::string_view> tensor_names = model.tensor_names();
    inferbridge::native_harness::batch_vulkan_initialization_uploads(
        context, tensor_names,
        [&](std::string_view name) {
            const TensorView& source = model.tensor(name);
            // FP32 expansion is the largest single representation produced by
            // this loader. Using it as the budget keeps retained staging
            // allocations bounded for every precision.
            return source.elements * sizeof(float);
        },
        [&](std::string_view name) {
        if (!load_fov_weights && name.rfind("fov.", 0) == 0) return;
        const TensorView& source = model.tensor(name);
        if (source.elements >
            std::numeric_limits<std::size_t>::max() / sizeof(float)) {
            throw std::runtime_error(
                "model tensor is too large for this process: " +
                std::string(name));
        }
        GpuTensor destination{
            {},
            {},
            {},
            {},
            source.dimensions,
            source.rank,
            source.elements,
        };
        if (is_large_weight(name) && uses_int8_weights_ &&
            source.rank == 2 && source.dimensions[1] % 4u == 0u) {
            std::vector<float> decoded(
                static_cast<std::size_t>(source.elements));
            for (std::size_t index = 0; index < decoded.size(); ++index)
                decoded[index] = half_to_float(source.data[index]);
            const auto quantized = inferbridge::native::quantize_int8_rows(
                decoded.data(), static_cast<std::size_t>(source.dimensions[0]),
                static_cast<std::size_t>(source.dimensions[1]));
            destination.int8_buffer = context.create_device_buffer(
                quantized.packed.size() * sizeof(std::uint32_t));
            destination.int8_scales = context.create_device_buffer(
                quantized.scales.size() * sizeof(float));
            upload(name, destination.int8_buffer, quantized.packed.data(),
                quantized.packed.size() * sizeof(std::uint32_t));
            upload(name, destination.int8_scales, quantized.scales.data(),
                quantized.scales.size() * sizeof(float));
        } else if (is_large_weight(name) && uses_half_weights_) {
            const std::size_t packed_bytes =
                static_cast<std::size_t>((source.elements + 1) / 2) *
                sizeof(std::uint32_t);
            std::vector<std::uint8_t> packed(packed_bytes, 0);
            const std::size_t source_bytes =
                static_cast<std::size_t>(source.elements) *
                sizeof(std::uint16_t);
            std::copy_n(
                reinterpret_cast<const std::uint8_t*>(source.data),
                source_bytes, packed.data());
            destination.half_buffer =
                context.create_device_buffer(packed_bytes);
            upload(name, destination.half_buffer,
                packed.data(), packed.size());
        } else {
            std::vector<float> decoded(
                static_cast<std::size_t>(source.elements));
            for (std::size_t index = 0; index < decoded.size(); ++index) {
                decoded[index] = half_to_float(source.data[index]);
            }
            destination.buffer =
                context.create_device_buffer(
                    decoded.size() * sizeof(float));
            upload(name, destination.buffer, decoded.data(),
                decoded.size() * sizeof(float));
        }
        if (!tensors_.emplace(name, std::move(destination)).second) {
            throw std::runtime_error(
                "duplicate GPU tensor name: " + std::string(name));
        }
        });
}

const GpuTensor& GpuModel::tensor(std::string_view name) const {
    const auto found = tensors_.find(name);
    if (found == tensors_.end()) {
        throw std::runtime_error(
            "GPU model is missing tensor: " + std::string(name));
    }
    return found->second;
}

}  // namespace depth_pro_native
