#if defined(__linux__) && !defined(__ANDROID__) &&                             \
    defined(DEPTH_PRO_WITH_VULKAN)
#include "gpu_io.h"
#include "linux_capture.h"
#include <inferbridge/linux_capture_preprocess.h>
ibr_linux_capture_capabilities
depth_pro_linux_capture_capabilities(depth_pro_context *context) {
  return context->vulkan ? context->vulkan->linux_capture_capabilities()
                         : ibr_linux_capture_capabilities{};
}
void depth_pro_infer_linux_capture(
    depth_pro_context *context,
    const inferbridge::linux_capture::LinuxDmaBufImage &source, float fov,
    float *output) {
  if (!context->vulkan)
    throw std::runtime_error("Depth Pro Vulkan context unavailable");
  auto &vk = *context->vulkan;
  auto &operators = *context->operators;
  depth_pro_native::GpuIo io(vk);
  auto x0 = inferbridge::linux_capture::capture_tensor(
      vk, source, 1536, 1536,
      {1, true, {.5f, .5f, .5f, 0}, {.5f, .5f, .5f, 1}});
  auto x1 = vk.create_device_buffer(uint64_t(3) * 768 * 768 * sizeof(float));
  auto x2 = vk.create_device_buffer(uint64_t(3) * 384 * 384 * sizeof(float));
  operators.bilinear_half_pixel(x1, x0, 1536, 1536, 768, 768, 3);
  operators.bilinear_half_pixel(x2, x0, 1536, 1536, 384, 384, 3);
  auto patches =
      vk.create_device_buffer(uint64_t(35) * 3 * 384 * 384 * sizeof(float));
  io.assemble_pyramid(patches, x0, x1, x2);
  auto zero = vk.create_device_buffer(1024 * sizeof(float));
  const std::vector<float> zeros(1024, 0);
  vk.upload(zero, zeros.data(), zeros.size() * sizeof(float));
  depth_pro_native::VulkanBuffer forced;
  if (fov > 0) {
    forced = vk.create_device_buffer(sizeof(float));
    vk.upload(forced, &fov, sizeof(float));
  }
  auto depth = depth_pro_native::infer_gpu_device(
      vk, *context->gpu_model, operators, patches, x2, zero,
      fov > 0 ? &forced : nullptr, source.width, source.height);
  vk.download(depth.depth, output,
              uint64_t(source.width) * source.height * sizeof(float));
}
#endif
