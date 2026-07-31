#ifndef DEPTH_PRO_NATIVE_H
#define DEPTH_PRO_NATIVE_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(DEPTH_PRO_BUILD_DLL)
#    define DEPTH_PRO_API __declspec(dllexport)
#  else
#    define DEPTH_PRO_API __declspec(dllimport)
#  endif
#  define DEPTH_PRO_CALL __cdecl
#else
#  define DEPTH_PRO_API __attribute__((visibility("default")))
#  define DEPTH_PRO_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define DEPTH_PRO_ABI_VERSION 4u

typedef struct depth_pro_context depth_pro_context;

typedef enum depth_pro_status {
    DEPTH_PRO_STATUS_OK = 0,
    DEPTH_PRO_STATUS_INVALID_ARGUMENT = 1,
    DEPTH_PRO_STATUS_OUT_OF_MEMORY = 2,
    DEPTH_PRO_STATUS_INTERNAL_ERROR = 3
} depth_pro_status;

typedef struct depth_pro_transfer_counters {
    uint32_t struct_size;
    uint32_t abi_version;
    uint64_t tensor_upload_bytes;
    uint64_t tensor_download_bytes;
} depth_pro_transfer_counters;

DEPTH_PRO_API depth_pro_status DEPTH_PRO_CALL depth_pro_get_transfer_counters(
    depth_pro_transfer_counters* counters);

DEPTH_PRO_API uint32_t DEPTH_PRO_CALL depth_pro_abi_version(void);
DEPTH_PRO_API const char* DEPTH_PRO_CALL depth_pro_version_string(void);
DEPTH_PRO_API const char* DEPTH_PRO_CALL depth_pro_last_error(void);

DEPTH_PRO_API depth_pro_status DEPTH_PRO_CALL depth_pro_create(
    const char* native_model_path_utf8,
    depth_pro_context** context);
/*
 * Creates a real full-graph Vulkan context on the zero-based physical-device
 * index. Failure is reported; this function never falls back to CPU.
 */
DEPTH_PRO_API depth_pro_status DEPTH_PRO_CALL depth_pro_create_vulkan(
    const char* native_model_path_utf8,
    uint32_t device_index,
    depth_pro_context** context);
DEPTH_PRO_API void DEPTH_PRO_CALL depth_pro_destroy(
    depth_pro_context* context);

/*
 * Input is contiguous planar RGB FP32 in [0,1]. Output is same-size metric
 * depth HW FP32. focal_length_pixels may be null.
 */
DEPTH_PRO_API depth_pro_status DEPTH_PRO_CALL depth_pro_infer_rgb_f32(
    depth_pro_context* context,
    const float* rgb_chw,
    int32_t width,
    int32_t height,
    float* depth_hw,
    uint64_t depth_elements,
    float* focal_length_pixels);

/*
 * Complete InferBridge capture path. The worker passes the first three BGRA
 * bytes directly through ToTensor, so their BGR ordering is intentionally
 * preserved. forced_fov_degrees in (0,180) matches FovEstimation=NO and skips
 * the learned FOV branch; zero enables learned FOV estimation.
 */
DEPTH_PRO_API depth_pro_status DEPTH_PRO_CALL depth_pro_infer_bgra8_f32(
    depth_pro_context* context,
    const uint8_t* bgra,
    uint64_t bgra_stride_bytes,
    int32_t width,
    int32_t height,
    float forced_fov_degrees,
    float* depth_hw,
    uint64_t depth_elements,
    float* focal_length_pixels);

#ifdef __cplusplus
}
#endif

#endif
