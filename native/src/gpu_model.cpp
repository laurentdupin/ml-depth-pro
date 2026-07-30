#include "gpu_model.h"

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

GpuModel::GpuModel(const ModelFile& model, VulkanContext& context) {
    tensors_.reserve(model.tensor_count());
    for (std::string_view name : model.tensor_names()) {
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
            source.dimensions,
            source.rank,
            source.elements,
        };
        if (is_large_weight(name)) {
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
            context.upload(
                destination.half_buffer, packed.data(), packed.size());
        } else {
            std::vector<float> decoded(
                static_cast<std::size_t>(source.elements));
            for (std::size_t index = 0; index < decoded.size(); ++index) {
                decoded[index] = half_to_float(source.data[index]);
            }
            destination.buffer =
                context.create_device_buffer(
                    decoded.size() * sizeof(float));
            context.upload(
                destination.buffer, decoded.data(),
                decoded.size() * sizeof(float));
        }
        if (!tensors_.emplace(name, std::move(destination)).second) {
            throw std::runtime_error(
                "duplicate GPU tensor name: " + std::string(name));
        }
    }
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
