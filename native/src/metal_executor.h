#pragma once

#include "graph_cpu.h"
#include "model.h"

#include <cstdint>
#include <memory>

namespace depth_pro_native {

class MetalExecutor {
public:
    explicit MetalExecutor(const ModelFile& model);
    ~MetalExecutor();
    MetalExecutor(const MetalExecutor&) = delete;
    MetalExecutor& operator=(const MetalExecutor&) = delete;

    InferenceOutput infer(
        const float* rgb,
        std::uint32_t width,
        std::uint32_t height,
        float forced_fov_degrees);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace depth_pro_native
