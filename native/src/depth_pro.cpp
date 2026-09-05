#include "depth_pro_native.h"

#include "graph_cpu.h"
#include "model.h"
#if defined(DEPTH_PRO_WITH_METAL)
#include "depth_pro_internal.h"
#include "metal_executor.h"
#endif
#if defined(DEPTH_PRO_WITH_VULKAN)
#include "gpu_model.h"
#include "graph_gpu.h"
#include "operators.h"
#include "vulkan.h"
#endif

#include <algorithm>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

struct depth_pro_context {
    std::unique_ptr<depth_pro_native::ModelFile> model;
#if defined(DEPTH_PRO_WITH_METAL)
    std::unique_ptr<depth_pro_native::MetalExecutor> metal;
#endif
#if defined(DEPTH_PRO_WITH_VULKAN)
    std::unique_ptr<depth_pro_native::VulkanContext> vulkan;
    std::unique_ptr<depth_pro_native::GpuModel> gpu_model;
    std::unique_ptr<depth_pro_native::VulkanOperators> operators;
#endif
};

namespace {
thread_local std::string last_error;

depth_pro_status fail(
    depth_pro_status status,
    const char* message) {
    last_error = message ? message : "";
    return status;
}

template <typename Function>
depth_pro_status protect(Function&& function) {
    try {
        function();
        last_error.clear();
        return DEPTH_PRO_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(
            DEPTH_PRO_STATUS_OUT_OF_MEMORY, "out of memory");
    } catch (const std::invalid_argument& error) {
        return fail(
            DEPTH_PRO_STATUS_INVALID_ARGUMENT, error.what());
    } catch (const std::exception& error) {
        return fail(
            DEPTH_PRO_STATUS_INTERNAL_ERROR, error.what());
    } catch (...) {
        return fail(
            DEPTH_PRO_STATUS_INTERNAL_ERROR,
            "unknown internal error");
    }
}
}

#if defined(DEPTH_PRO_WITH_METAL)
namespace depth_pro_native {

class ContextMetalExternalGpu final : public ExternalGpu {
public:
    ContextMetalExternalGpu(
        depth_pro_context* context, float forced_fov_degrees)
        : context_(context), forced_fov_degrees_(forced_fov_degrees) {
        if (context_ == nullptr || context_->metal == nullptr)
            throw std::invalid_argument("Depth Pro Metal context is unavailable");
    }

    ExternalGpuCapabilities capabilities() const override {
        return {true, 0u, 3u};
    }

    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) override {
        return context_->metal->submit_texture(request, forced_fov_degrees_);
    }

    void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const override {
        upload_bytes = 0u;
        download_bytes = 0u;
    }

private:
    depth_pro_context* context_;
    float forced_fov_degrees_;
};

std::shared_ptr<ExternalGpu> create_metal_external_gpu(
    depth_pro_context* context, float forced_fov_degrees,
    const std::string& cache_path) {
    if (!context || !context->metal)
        throw std::invalid_argument("Depth Pro Metal context is unavailable");
    context->metal->set_cache_path(cache_path);
    return std::make_shared<ContextMetalExternalGpu>(
        context, forced_fov_degrees);
}

}  // namespace depth_pro_native
#endif

extern "C" {

uint32_t DEPTH_PRO_CALL depth_pro_abi_version(void) {
    return DEPTH_PRO_ABI_VERSION;
}

const char* DEPTH_PRO_CALL depth_pro_version_string(void) {
    return "0.5.0-d3d12-vulkan-metal-gpu-resident";
}

depth_pro_status DEPTH_PRO_CALL depth_pro_get_transfer_counters(
    depth_pro_transfer_counters* counters) {
    if (counters == nullptr ||
        counters->struct_size < sizeof(*counters)) {
        return fail(
            DEPTH_PRO_STATUS_INVALID_ARGUMENT,
            "invalid transfer counter output");
    }
    counters->abi_version = DEPTH_PRO_ABI_VERSION;
#if defined(DEPTH_PRO_WITH_VULKAN)
    depth_pro_native::global_transfer_counters(
        counters->tensor_upload_bytes,
        counters->tensor_download_bytes);
#else
    counters->tensor_upload_bytes = 0u;
    counters->tensor_download_bytes = 0u;
#endif
    return DEPTH_PRO_STATUS_OK;
}

depth_pro_status DEPTH_PRO_CALL depth_pro_create_vulkan(
    const char* path,
    uint32_t device_index,
    depth_pro_context** context) {
    if (!context) {
        return fail(
            DEPTH_PRO_STATUS_INVALID_ARGUMENT,
            "context output is null");
    }
    *context = nullptr;
    if (!path || path[0] == '\0') {
        return fail(
            DEPTH_PRO_STATUS_INVALID_ARGUMENT,
            "model path is empty");
    }
#if defined(DEPTH_PRO_WITH_METAL)
    (void)device_index;
    return protect([&] {
        auto result = std::make_unique<depth_pro_context>();
        result->model =
            std::make_unique<depth_pro_native::ModelFile>(path);
        result->metal =
            std::make_unique<depth_pro_native::MetalExecutor>(*result->model);
        *context = result.release();
    });
#elif !defined(DEPTH_PRO_WITH_VULKAN)
    (void)device_index;
    return fail(
        DEPTH_PRO_STATUS_INTERNAL_ERROR,
        "this DLL was built without Vulkan");
#else
    return protect([&] {
        auto result = std::make_unique<depth_pro_context>();
        result->model =
            std::make_unique<depth_pro_native::ModelFile>(path);
        result->vulkan =
            std::make_unique<depth_pro_native::VulkanContext>(device_index);
        result->gpu_model =
            std::make_unique<depth_pro_native::GpuModel>(
                *result->model, *result->vulkan);
        result->operators =
            std::make_unique<depth_pro_native::VulkanOperators>(
                *result->vulkan);
        *context = result.release();
    });
#endif
}

const char* DEPTH_PRO_CALL depth_pro_last_error(void) {
    return last_error.c_str();
}

depth_pro_status DEPTH_PRO_CALL depth_pro_create(
    const char* path,
    depth_pro_context** context) {
    if (!context) {
        return fail(
            DEPTH_PRO_STATUS_INVALID_ARGUMENT,
            "context output is null");
    }
    *context = nullptr;
    if (!path || path[0] == '\0') {
        return fail(
            DEPTH_PRO_STATUS_INVALID_ARGUMENT,
            "model path is empty");
    }
    return protect([&] {
        auto result = std::make_unique<depth_pro_context>();
        result->model =
            std::make_unique<depth_pro_native::ModelFile>(path);
        *context = result.release();
    });
}

void DEPTH_PRO_CALL depth_pro_destroy(
    depth_pro_context* context) {
    delete context;
}

depth_pro_status DEPTH_PRO_CALL depth_pro_infer_rgb_f32(
    depth_pro_context* context,
    const float* rgb,
    int32_t width,
    int32_t height,
    float* depth,
    uint64_t depth_elements,
    float* focal) {
    if (!context || !context->model || !rgb || !depth ||
        width <= 0 || height <= 0 ||
        depth_elements <
            static_cast<std::uint64_t>(width) *
                static_cast<std::uint64_t>(height)) {
        return fail(
            DEPTH_PRO_STATUS_INVALID_ARGUMENT,
            "invalid Depth Pro tensor inference input");
    }
    return protect([&] {
#if defined(DEPTH_PRO_WITH_METAL)
        if (context->metal) {
            depth_pro_native::InferenceOutput result =
                context->metal->infer(
                    rgb, static_cast<std::uint32_t>(width),
                    static_cast<std::uint32_t>(height), 0.0f);
            std::copy(result.depth.begin(), result.depth.end(), depth);
            if (focal) *focal = result.focal_length_pixels;
            return;
        }
#endif
#if defined(DEPTH_PRO_WITH_VULKAN)
        if (context->vulkan) {
            depth_pro_native::GpuInferenceOutput result =
                depth_pro_native::infer_gpu(
                    *context->vulkan, *context->gpu_model,
                    *context->operators, rgb,
                    static_cast<std::uint32_t>(width),
                    static_cast<std::uint32_t>(height));
            std::copy(result.depth.begin(), result.depth.end(), depth);
            if (focal) {
                *focal = result.focal_length_pixels;
            }
            return;
        }
#endif
        depth_pro_native::InferenceOutput result =
            depth_pro_native::infer_cpu(
                *context->model, rgb,
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height));
        std::copy(result.depth.begin(), result.depth.end(), depth);
        if (focal) {
            *focal = result.focal_length_pixels;
        }
    });
}

depth_pro_status DEPTH_PRO_CALL depth_pro_infer_bgra8_f32(
    depth_pro_context* context,
    const uint8_t* bgra,
    uint64_t bgra_stride_bytes,
    int32_t width,
    int32_t height,
    float forced_fov_degrees,
    float* depth,
    uint64_t depth_elements,
    float* focal) {
    if (!context || !context->model || !bgra || !depth ||
        width <= 0 || height <= 0 ||
        bgra_stride_bytes <
            static_cast<std::uint64_t>(width) * 4 ||
        depth_elements <
            static_cast<std::uint64_t>(width) * height ||
        forced_fov_degrees < 0.0f ||
        forced_fov_degrees >= 180.0f) {
        return fail(
            DEPTH_PRO_STATUS_INVALID_ARGUMENT,
            "invalid Depth Pro BGRA image inference input");
    }
    return protect([&] {
        const std::uint64_t plane =
            static_cast<std::uint64_t>(width) * height;
        std::vector<float> rgb(
            static_cast<std::size_t>(3 * plane));
        for (int32_t y = 0; y < height; ++y) {
            const std::uint8_t* row =
                bgra + static_cast<std::uint64_t>(y) *
                    bgra_stride_bytes;
            for (int32_t x = 0; x < width; ++x) {
                const std::uint64_t pixel =
                    static_cast<std::uint64_t>(y) * width + x;
                for (std::uint32_t channel = 0;
                     channel < 3; ++channel) {
                    rgb[static_cast<std::size_t>(
                        std::uint64_t(channel) * plane + pixel)] =
                        row[static_cast<std::uint64_t>(x) * 4 +
                            channel] /
                        255.0f;
                }
            }
        }
#if defined(DEPTH_PRO_WITH_VULKAN)
        if (context->vulkan) {
            depth_pro_native::GpuInferenceOutput result =
                depth_pro_native::infer_gpu(
                    *context->vulkan, *context->gpu_model,
                    *context->operators, rgb.data(),
                    static_cast<std::uint32_t>(width),
                    static_cast<std::uint32_t>(height),
                    forced_fov_degrees);
            std::copy(result.depth.begin(), result.depth.end(), depth);
            if (focal) {
                *focal = result.focal_length_pixels;
            }
            return;
        }
#endif
#if defined(DEPTH_PRO_WITH_METAL)
        if (context->metal) {
            depth_pro_native::InferenceOutput result =
                context->metal->infer(
                    rgb.data(), static_cast<std::uint32_t>(width),
                    static_cast<std::uint32_t>(height), forced_fov_degrees);
            std::copy(result.depth.begin(), result.depth.end(), depth);
            if (focal) *focal = result.focal_length_pixels;
            return;
        }
#endif
        depth_pro_native::InferenceOutput result =
            depth_pro_native::infer_cpu(
                *context->model, rgb.data(),
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(height),
                forced_fov_degrees);
        std::copy(result.depth.begin(), result.depth.end(), depth);
        if (focal) {
            *focal = result.focal_length_pixels;
        }
    });
}

}

#include "linux_capture.inl"
