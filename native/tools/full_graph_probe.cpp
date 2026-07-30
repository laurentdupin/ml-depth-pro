#include "graph_cpu.h"
#include "model.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::vector<float> read(
    const std::string& path,
    std::uint64_t count) {
    std::vector<float> values(static_cast<std::size_t>(count));
    std::ifstream input(path, std::ios::binary);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(float)));
    if (!input ||
        input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("invalid tensor file: " + path);
    }
    return values;
}
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr
            << "usage: depth_pro_full_graph_probe model size rgb "
               "depth focal tolerance\n";
        return 2;
    }
    try {
        const std::uint32_t size =
            static_cast<std::uint32_t>(std::stoul(argv[2]));
        const double tolerance = std::stod(argv[6]);
        const std::vector<float> rgb =
            read(argv[3], std::uint64_t(3) * size * size);
        const std::vector<float> reference =
            read(argv[4], std::uint64_t(size) * size);
        const float focal_reference = read(argv[5], 1)[0];
        depth_pro_native::ModelFile model(argv[1]);
        const auto start = std::chrono::steady_clock::now();
        const depth_pro_native::InferenceOutput output =
            depth_pro_native::infer_cpu(
                model, rgb.data(), size, size);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        double absolute_sum = 0.0;
        double reference_sum = 0.0;
        float maximum = 0.0f;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            const float difference =
                std::abs(output.depth[i] - reference[i]);
            maximum = std::max(maximum, difference);
            absolute_sum += difference;
            reference_sum += std::abs(reference[i]);
        }
        const double relative =
            absolute_sum / std::max(reference_sum, 1.0e-30);
        const double focal_relative =
            std::abs(output.focal_length_pixels - focal_reference) /
            std::abs(focal_reference);
        std::cout << "seconds=" << seconds
                  << "\nmaximum_absolute=" << maximum
                  << "\nrelative_l1=" << relative
                  << "\nfocal=" << output.focal_length_pixels
                  << "\nfocal_relative=" << focal_relative << "\n";
        return relative <= tolerance &&
            focal_relative <= tolerance ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}

