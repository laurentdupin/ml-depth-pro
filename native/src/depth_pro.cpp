#include "depth_pro_native.h"

#include "graph_cpu.h"
#include "model.h"
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

extern "C" {

uint32_t DEPTH_PRO_CALL depth_pro_abi_version(void) {
    return DEPTH_PRO_ABI_VERSION;
}

const char* DEPTH_PRO_CALL depth_pro_version_string(void) {
    return "0.2.0-cpu-vulkan-full-graph";
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
#if !defined(DEPTH_PRO_WITH_VULKAN)
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

}
