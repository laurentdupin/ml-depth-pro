#pragma once

#include "graph_cpu.h"
#include "model.h"
#include "external_gpu.h"

#include <cstdint>
#include <memory>
#include <string>

namespace depth_pro_native {

class MetalExecutor {
public:
    explicit MetalExecutor(const ModelFile& model);
    ~MetalExecutor();
    MetalExecutor(const MetalExecutor&) = delete;
    MetalExecutor& operator=(const MetalExecutor&) = delete;

    void set_cache_path(const std::string& cache_path);

    InferenceOutput infer(
        const float* rgb,
        std::uint32_t width,
        std::uint32_t height,
        float forced_fov_degrees);
    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request,
        float forced_fov_degrees);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace depth_pro_native
