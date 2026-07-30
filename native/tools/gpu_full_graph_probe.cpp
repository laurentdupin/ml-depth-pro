#include "depth_pro_native.h"

#include <algorithm>
#include <chrono>
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
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "usage: depth_pro_gpu_full_graph_probe "
                     "model device size rgb depth focal tolerance\n";
        return 2;
    }
    depth_pro_context* context = nullptr;
    try {
        const std::uint32_t device =
            static_cast<std::uint32_t>(std::stoul(argv[2]));
        const std::uint32_t size =
            static_cast<std::uint32_t>(std::stoul(argv[3]));
        const double tolerance = std::stod(argv[7]);
        const std::vector<float> rgb =
            read(argv[4], std::uint64_t(3) * size * size);
        const std::vector<float> reference =
            read(argv[5], std::uint64_t(size) * size);
        const float focal_reference = read(argv[6], 1)[0];
        const auto create_start = std::chrono::steady_clock::now();
        depth_pro_status status =
            depth_pro_create_vulkan(argv[1], device, &context);
        if (status != DEPTH_PRO_STATUS_OK) {
            throw std::runtime_error(depth_pro_last_error());
        }
        const double create_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - create_start).count();
        std::vector<float> output(reference.size());
        float focal = 0.0f;
        const auto start = std::chrono::steady_clock::now();
        status = depth_pro_infer_rgb_f32(
            context, rgb.data(), static_cast<int32_t>(size),
            static_cast<int32_t>(size), output.data(), output.size(), &focal);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        if (status != DEPTH_PRO_STATUS_OK) {
            throw std::runtime_error(depth_pro_last_error());
        }
        double difference_sum = 0.0;
        double reference_sum = 0.0;
        float maximum = 0.0f;
        for (std::size_t index = 0; index < output.size(); ++index) {
            const float difference =
                std::abs(output[index] - reference[index]);
            difference_sum += difference;
            reference_sum += std::abs(reference[index]);
            maximum = std::max(maximum, difference);
        }
        const double relative =
            difference_sum / std::max(reference_sum, 1.0e-30);
        const double focal_relative =
            std::abs(focal - focal_reference) /
            std::max(std::abs(focal_reference), 1.0e-30f);
        std::cout << "create_seconds=" << create_seconds
                  << "\nseconds=" << seconds
                  << "\nmaximum_absolute=" << maximum
                  << "\nrelative_l1=" << relative
                  << "\nfocal=" << focal
                  << "\nfocal_relative=" << focal_relative << "\n";
        depth_pro_destroy(context);
        return relative <= tolerance &&
            focal_relative <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        depth_pro_destroy(context);
        std::cerr << error.what() << "\n";
        return 1;
    }
}
