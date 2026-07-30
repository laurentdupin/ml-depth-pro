#pragma once

#include "model.h"

#include <cstdint>
#include <string_view>
#include <vector>

namespace depth_pro_native {

struct EncoderOutput {
    std::uint32_t patch_width = 0;
    std::uint32_t patch_height = 0;
    std::vector<std::vector<float>> captures;
    std::vector<float> final;
};

EncoderOutput encoder_cpu(
    const ModelFile& model,
    std::string_view encoder_prefix,
    const float* normalized_rgb_chw,
    std::uint32_t width,
    std::uint32_t height);

}  // namespace depth_pro_native
