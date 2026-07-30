#include "model.h"

#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: depth_pro_model_probe model.dpro\n";
        return 2;
    }
    try {
        depth_pro_native::ModelFile model(argv[1]);
        const auto& patch = model.tensor(
            "encoder.patch_encoder.patch_embed.proj.weight");
        const auto& derivation = model.derivation();
        const bool expected =
            model.tensor_count() == 1119 &&
            patch.rank == 4 &&
            patch.dimensions[0] == 1024 &&
            patch.dimensions[1] == 3 &&
            patch.dimensions[2] == 16 &&
            patch.dimensions[3] == 16 &&
            model.contains("head.4.weight") &&
            model.contains("fov.head.4.weight") &&
            derivation.converter ==
                "depth-pro-export-pytorch-v1";
        std::cout << "tensors=" << model.tensor_count()
                  << "\npatch_shape="
                  << patch.dimensions[0] << "x"
                  << patch.dimensions[1] << "x"
                  << patch.dimensions[2] << "x"
                  << patch.dimensions[3]
                  << "\ncanonical_sha256=";
        for (const std::uint8_t value :
             derivation.canonical_sha256) {
            std::cout << std::hex << std::setw(2)
                      << std::setfill('0')
                      << static_cast<unsigned>(value);
        }
        std::cout << "\n";
        return expected ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}

