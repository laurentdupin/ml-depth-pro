#pragma once
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
struct depth_pro_context;
ibr_linux_capture_capabilities
depth_pro_linux_capture_capabilities(depth_pro_context *);
void depth_pro_infer_linux_capture(
    depth_pro_context *, const inferbridge::linux_capture::LinuxDmaBufImage &,
    float, float *);
#endif
