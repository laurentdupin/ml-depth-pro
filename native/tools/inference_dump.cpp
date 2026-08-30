#include "depth_pro_native.h"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "usage: depth_pro_inference_dump model size fov output\n";
        return 2;
    }
    depth_pro_context* context = nullptr;
    try {
        const std::uint32_t size = std::stoul(argv[2]);
        const float fov = std::stof(argv[3]);
        std::vector<std::uint8_t> bgra(
            static_cast<std::size_t>(size) * size * 4);
        for (std::uint32_t y = 0; y < size; ++y) {
            for (std::uint32_t x = 0; x < size; ++x) {
                const std::size_t p =
                    (static_cast<std::size_t>(y) * size + x) * 4;
                bgra[p] = static_cast<std::uint8_t>((x * 7 + y * 3) & 255);
                bgra[p + 1] = static_cast<std::uint8_t>((x * 2 + y * 9) & 255);
                bgra[p + 2] = static_cast<std::uint8_t>((x * 5 + y * 4) & 255);
                bgra[p + 3] = 255;
            }
        }
        depth_pro_status status = depth_pro_create_vulkan(argv[1], 0, &context);
        if (status != DEPTH_PRO_STATUS_OK)
            throw std::runtime_error(depth_pro_last_error());
        std::vector<float> depth(static_cast<std::size_t>(size) * size);
        float focal = 0.0f;
        status = depth_pro_infer_bgra8_f32(
            context, bgra.data(), static_cast<std::uint64_t>(size) * 4,
            size, size, fov, depth.data(), depth.size(), &focal);
        if (status != DEPTH_PRO_STATUS_OK)
            throw std::runtime_error(depth_pro_last_error());
        depth_pro_destroy(context);
        context = nullptr;
        std::ofstream output(argv[4], std::ios::binary);
        output.write(reinterpret_cast<const char*>(depth.data()),
                     depth.size() * sizeof(float));
        output.write(reinterpret_cast<const char*>(&focal), sizeof(float));
        if (!output) throw std::runtime_error("could not write output");
        return 0;
    } catch (const std::exception& error) {
        depth_pro_destroy(context);
        std::cerr << error.what() << "\n";
        return 1;
    }
}
