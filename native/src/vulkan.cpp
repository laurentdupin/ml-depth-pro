#include "vulkan.h"
#include "inferbridge/native_harness_vulkan_queue_priority.h"
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
#include <linux_capture_preprocess_spv.h>
#endif

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace depth_pro_native {
namespace {

std::atomic<std::uint64_t> g_tensor_upload_bytes{0u};
std::atomic<std::uint64_t> g_tensor_download_bytes{0u};

template <typename Handle>
void exchange_handle(Handle& left, Handle& right) {
    std::swap(left, right);
}

#if defined(_WIN32)
bool has_extension(
    const std::vector<VkExtensionProperties>& extensions,
    const char* name) {
    return std::any_of(
        extensions.begin(),
        extensions.end(),
        [name](const VkExtensionProperties& extension) {
            return std::strcmp(extension.extensionName, name) == 0;
        });
}
#endif

}  // namespace

struct VulkanSubmission::Resources {
    std::vector<VulkanBatchedDescriptor> descriptor_sets;
    std::vector<VulkanDeferredBuffer> deferred_buffers;
    std::vector<VulkanSubmission> prior_submissions;
    VulkanSemaphore wait;
    VulkanSemaphore signal;
};

#include <inferbridge/vulkan/context/vulkan_context_check.h>

VulkanSemaphore::VulkanSemaphore(VulkanSemaphore&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_semaphore_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_semaphore_destroy_VulkanSemaphore.h>

VulkanSubmission::VulkanSubmission(VulkanSubmission&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_submission_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_submission_destroy_VulkanSubmission.h>

#include <inferbridge/vulkan/context/vulkan_submission_ready.h>

#include <inferbridge/vulkan/context/vulkan_submission_wait.h>

VulkanBuffer::VulkanBuffer(VulkanBuffer&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_buffer_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_buffer_destroy_VulkanBuffer.h>

VulkanImage::VulkanImage(VulkanImage&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_image_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_image_destroy_VulkanImage.h>

VulkanPipeline::VulkanPipeline(VulkanPipeline&& other) noexcept {
    *this = std::move(other);
}

#include <inferbridge/vulkan/context/vulkan_pipeline_operator_assign.h>

#include <inferbridge/vulkan/context/vulkan_pipeline_destroy_VulkanPipeline.h>

#include <inferbridge/vulkan/context/vulkan_pipeline_set_debug_name.h>

VulkanContext::VulkanContext(
    std::uint32_t device_index,
    bool track_resource_hazards)
    : track_resource_hazards_(track_resource_hazards) {
    try {
    const VkApplicationInfo application{
        VK_STRUCTURE_TYPE_APPLICATION_INFO,
        nullptr,
        "Depth Anything V2 C",
        1,
        "DPRO",
        1,
        VK_API_VERSION_1_3,
    };
    const VkInstanceCreateInfo instance_info{
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        nullptr,
        0,
        &application,
        0,
        nullptr,
        0,
        nullptr,
    };
    check(vkCreateInstance(&instance_info, nullptr, &instance_), "vkCreateInstance");

    std::uint32_t device_count = 0;
    check(
        vkEnumeratePhysicalDevices(instance_, &device_count, nullptr),
        "vkEnumeratePhysicalDevices");
    if (device_count == 0 || device_index >= device_count) {
        throw std::runtime_error("requested Vulkan device does not exist");
    }
    std::vector<VkPhysicalDevice> devices(device_count);
    check(
        vkEnumeratePhysicalDevices(instance_, &device_count, devices.data()),
        "vkEnumeratePhysicalDevices");
    physical_device_ = devices[device_index];

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(physical_device_, &properties);
    device_name_ = properties.deviceName;
    VkPhysicalDeviceShaderFloat16Int8Features float16_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
    };
    VkPhysicalDevice16BitStorageFeatures storage16_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        &float16_features,
    };
    VkPhysicalDeviceShaderIntegerDotProductFeatures integer_dot_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
    };
    float16_features.pNext = &integer_dot_features;
    VkPhysicalDeviceFeatures2 precision_features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        &storage16_features,
    };
    vkGetPhysicalDeviceFeatures2(physical_device_, &precision_features);
    float16_supported_ =
        storage16_features.storageBuffer16BitAccess == VK_TRUE &&
        float16_features.shaderFloat16 == VK_TRUE;
    VkPhysicalDeviceShaderIntegerDotProductProperties integer_dot_properties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 integer_dot_properties2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        &integer_dot_properties,
    };
    vkGetPhysicalDeviceProperties2(physical_device_, &integer_dot_properties2);
    packed_int8_dot_supported_ =
        integer_dot_features.shaderIntegerDotProduct == VK_TRUE &&
        integer_dot_properties
            .integerDotProduct4x8BitPackedSignedAccelerated == VK_TRUE;
#if defined(_WIN32)
    VkPhysicalDeviceIDProperties identity{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES,
    };
    VkPhysicalDeviceProperties2 properties2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        &identity,
    };
    vkGetPhysicalDeviceProperties2(physical_device_, &properties2);
    if (identity.deviceLUIDValid) {
        static_assert(VK_LUID_SIZE == sizeof(adapter_luid_));
        std::memcpy(
            &adapter_luid_, identity.deviceLUID, VK_LUID_SIZE);
    }
#endif
    vkGetPhysicalDeviceMemoryProperties(
        physical_device_, &memory_properties_);
    for (std::uint32_t heap_index = 0;
         heap_index < memory_properties_.memoryHeapCount;
         ++heap_index) {
        const VkMemoryHeap& heap =
            memory_properties_.memoryHeaps[heap_index];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            device_local_memory_bytes_ = std::max(
                device_local_memory_bytes_, heap.size);
        }
    }

    std::uint32_t extension_count = 0;
    check(
        vkEnumerateDeviceExtensionProperties(
            physical_device_, nullptr, &extension_count, nullptr),
        "vkEnumerateDeviceExtensionProperties");
    std::vector<VkExtensionProperties> extensions(extension_count);
    check(
        vkEnumerateDeviceExtensionProperties(
            physical_device_, nullptr, &extension_count, extensions.data()),
        "vkEnumerateDeviceExtensionProperties");
    std::vector<const char*> enabled_extensions;
#if defined(__linux__) && !defined(__ANDROID__)
    linux_dma_buf_enabled_ = inferbridge::linux_capture::enable_extensions(
        extensions, enabled_extensions);
#endif
#if defined(_WIN32)
    const bool has_external_memory_win32 = has_extension(
        extensions, VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    const bool has_external_semaphore_win32 = has_extension(
        extensions, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    if (has_external_memory_win32) {
        const VkPhysicalDeviceExternalBufferInfo external_buffer{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
            nullptr,
            0,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
        };
        VkExternalBufferProperties external_properties{
            VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES,
        };
        vkGetPhysicalDeviceExternalBufferProperties(
            physical_device_,
            &external_buffer,
            &external_properties);
        external_capabilities_.d3d12_resource_import =
            (external_properties.externalMemoryProperties
                 .externalMemoryFeatures &
             VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
        const auto image_importable =
            [&](VkFormat format, VkImageUsageFlags usage) {
                const VkPhysicalDeviceExternalImageFormatInfo external{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
                    nullptr,
                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE_BIT,
                };
                const VkPhysicalDeviceImageFormatInfo2 image_info{
                    VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
                    &external,
                    format,
                    VK_IMAGE_TYPE_2D,
                    VK_IMAGE_TILING_OPTIMAL,
                    usage,
                    0,
                };
                VkExternalImageFormatProperties external_properties{
                    VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
                };
                VkImageFormatProperties2 image_properties{
                    VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
                    &external_properties,
                };
                const VkResult queried =
                    vkGetPhysicalDeviceImageFormatProperties2(
                        physical_device_,
                        &image_info,
                        &image_properties);
                return queried == VK_SUCCESS &&
                    (external_properties.externalMemoryProperties
                         .externalMemoryFeatures &
                     VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT) != 0;
            };
        external_capabilities_
            .d3d12_bgra8_sampled_image_import =
            image_importable(
                VK_FORMAT_B8G8R8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        external_capabilities_
            .d3d12_rgba8_sampled_image_import =
            image_importable(
                VK_FORMAT_R8G8B8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        external_capabilities_
            .d3d12_r32_storage_image_import =
            image_importable(
                VK_FORMAT_R32_SFLOAT,
                VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        enabled_extensions.push_back(
            VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
    }
    if (has_external_semaphore_win32) {
        const VkPhysicalDeviceExternalSemaphoreInfo external_semaphore{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
            nullptr,
            VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT,
        };
        VkExternalSemaphoreProperties semaphore_properties{
            VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES,
        };
        vkGetPhysicalDeviceExternalSemaphoreProperties(
            physical_device_,
            &external_semaphore,
            &semaphore_properties);
        external_capabilities_.d3d12_fence_import =
            (semaphore_properties.externalSemaphoreFeatures &
             VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT) != 0;
        enabled_extensions.push_back(
            VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME);
    }
#else
    (void)extensions;
#endif

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device_, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device_, &family_count, families.data());
#if defined(_WIN32)
    constexpr bool prefer_dedicated = true;
#else
    constexpr bool prefer_dedicated = false;
#endif
    auto family = inferbridge::native_harness::select_inference_queue_family(families, prefer_dedicated);
    if (family == families.end()) {
        throw std::runtime_error("Vulkan device has no compute queue");
    }
    queue_family_ = static_cast<std::uint32_t>(
        std::distance(families.begin(), family));
    const char* profile_environment =
        std::getenv("DPRO_VULKAN_PROFILE");
    profile_dispatches_ =
        profile_environment != nullptr &&
        profile_environment[0] != '\0' &&
        profile_environment[0] != '0' &&
        family->timestampValidBits != 0;
    timestamp_period_ns_ = properties.limits.timestampPeriod;

    constexpr float priority = 1.0f;
    const VkDeviceQueueCreateInfo queue_info{
        VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        nullptr,
        0,
        queue_family_,
        1,
        &priority,
    };
    VkPhysicalDeviceShaderFloat16Int8Features enabled_float16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        nullptr,
        float16_supported_ ? VK_TRUE : VK_FALSE,
        VK_FALSE,
    };
    VkPhysicalDevice16BitStorageFeatures enabled_storage16{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        float16_supported_ ? &enabled_float16 : nullptr,
        float16_supported_ ? VK_TRUE : VK_FALSE,
        VK_FALSE,
        VK_FALSE,
        VK_FALSE,
    };
    VkPhysicalDeviceShaderIntegerDotProductFeatures enabled_integer_dot{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES,
        nullptr,
        packed_int8_dot_supported_ ? VK_TRUE : VK_FALSE,
    };
    enabled_float16.pNext = &enabled_integer_dot;
    const VkDeviceCreateInfo device_info{
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        float16_supported_ ? static_cast<void*>(&enabled_storage16)
        : packed_int8_dot_supported_
        ? static_cast<void*>(&enabled_integer_dot) : nullptr,
        0,
        1,
        &queue_info,
        0,
        nullptr,
        static_cast<std::uint32_t>(enabled_extensions.size()),
        enabled_extensions.data(),
        nullptr,
    };
    check(
        inferbridge::native_harness::create_inference_vulkan_device(physical_device_, &device_info, nullptr, &device_),
        "vkCreateDevice");
#if defined(_WIN32)
    get_memory_win32_handle_properties_ =
        reinterpret_cast<
            PFN_vkGetMemoryWin32HandlePropertiesKHR>(
            vkGetDeviceProcAddr(
                device_,
                "vkGetMemoryWin32HandlePropertiesKHR"));
    import_semaphore_win32_handle_ =
        reinterpret_cast<PFN_vkImportSemaphoreWin32HandleKHR>(
            vkGetDeviceProcAddr(
                device_,
                "vkImportSemaphoreWin32HandleKHR"));
    external_capabilities_.d3d12_resource_import =
        external_capabilities_.d3d12_resource_import &&
        get_memory_win32_handle_properties_ != nullptr;
    external_capabilities_.d3d12_fence_import =
        external_capabilities_.d3d12_fence_import &&
        import_semaphore_win32_handle_ != nullptr;
#endif
    vkGetDeviceQueue(device_, queue_family_, 0, &queue_);

    const VkCommandPoolCreateInfo command_pool_info{
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        nullptr,
        VK_COMMAND_POOL_CREATE_TRANSIENT_BIT |
            VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        queue_family_,
    };
    check(
        vkCreateCommandPool(
            device_, &command_pool_info, nullptr, &command_pool_),
        "vkCreateCommandPool");

    const VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32768},
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8192},
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 64},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64},
    };
    const VkDescriptorPoolCreateInfo descriptor_pool_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        nullptr,
        VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        8192,
        4,
        pool_sizes,
    };
    check(
        vkCreateDescriptorPool(
            device_, &descriptor_pool_info, nullptr, &descriptor_pool_),
        "vkCreateDescriptorPool");
    if (profile_dispatches_) {
        const VkQueryPoolCreateInfo query_pool_info{
            VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            nullptr,
            0,
            VK_QUERY_TYPE_TIMESTAMP,
            2,
            0,
        };
        check(
            vkCreateQueryPool(
                device_,
                &query_pool_info,
                nullptr,
                &profile_query_pool_),
            "vkCreateQueryPool");
    }
    } catch (...) {
        release();
        throw;
    }
}

#include <inferbridge/vulkan/context/vulkan_context_destroy_VulkanContext.h>

void VulkanContext::release() noexcept {
    if (device_) {
#if defined(__linux__) && !defined(__ANDROID__)
        vkDeviceWaitIdle(device_);
        linux_capture_images_.clear();
        linux_capture_pipeline_.reset();
#endif
        print_profile();
        cancel_batch();
        for (VulkanDeferredBuffer& buffer : device_buffer_pool_) {
            buffer.cacheable = false;
            recycle_or_destroy(buffer);
        }
        device_buffer_pool_.clear();
        pooled_device_bytes_ = 0;
        for (VulkanDeferredBuffer& buffer : host_buffer_pool_) {
            buffer.cacheable = false;
            recycle_or_destroy(buffer);
        }
        host_buffer_pool_.clear();
        pooled_host_bytes_ = 0;
        if (descriptor_pool_) {
            vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
        }
        if (profile_query_pool_) {
            vkDestroyQueryPool(
                device_, profile_query_pool_, nullptr);
        }
        if (command_pool_) {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
        }
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
        descriptor_pool_ = VK_NULL_HANDLE;
        profile_query_pool_ = VK_NULL_HANDLE;
        command_pool_ = VK_NULL_HANDLE;
    }
    if (instance_) {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

#include <inferbridge/vulkan/context/vulkan_context_record_profile.h>

void VulkanContext::print_profile() const noexcept {
    if (!profile_dispatches_ || profile_stats_.empty()) return;
    std::vector<std::pair<std::string, ProfileStat>> sorted(
        profile_stats_.begin(), profile_stats_.end());
    std::sort(
        sorted.begin(),
        sorted.end(),
        [](const auto& left, const auto& right) {
            return left.second.total_ticks >
                right.second.total_ticks;
        });
    std::fprintf(
        stderr,
        "DPRO Vulkan profile: %s (GPU timestamps)\n",
        device_name_.c_str());
    for (const auto& [name, stat] : sorted) {
        const double total_ms =
            static_cast<double>(stat.total_ticks) *
            timestamp_period_ns_ / 1.0e6;
        const double average_ms =
            total_ms / static_cast<double>(stat.dispatches);
        const double maximum_ms =
            static_cast<double>(stat.maximum_ticks) *
            timestamp_period_ns_ / 1.0e6;
        std::fprintf(
            stderr,
            "  %-34s total=%10.3f ms count=%6llu "
            "avg=%8.4f ms max=%8.4f ms\n",
            name.c_str(),
            total_ms,
            static_cast<unsigned long long>(stat.dispatches),
            average_ms,
            maximum_ms);
    }
}

#include <inferbridge/vulkan/context/vulkan_context_find_memory_type.h>

#include <inferbridge/vulkan/context/vulkan_context_create_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_create_device_buffer.h>

#if defined(_WIN32)
#include <inferbridge/vulkan/context/vulkan_context_import_d3d12_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_import_d3d12_image.h>

#include <inferbridge/vulkan/context/vulkan_context_import_d3d12_fence.h>

#endif

#include <inferbridge/vulkan/context/vulkan_context_create_host_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_write_host.h>

VkCommandBuffer VulkanContext::begin_commands() {
    const VkCommandBufferAllocateInfo allocate_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        nullptr,
        command_pool_,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        1,
    };
    VkCommandBuffer command = VK_NULL_HANDLE;
    check(
        vkAllocateCommandBuffers(device_, &allocate_info, &command),
        "vkAllocateCommandBuffers");
    const VkCommandBufferBeginInfo begin_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        nullptr,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        nullptr,
    };
    check(vkBeginCommandBuffer(command, &begin_info), "vkBeginCommandBuffer");
    if (deferred_sequence_active_) {
        const VkMemoryBarrier barrier{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        };
        vkCmdPipelineBarrier(
            command, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier,
            0, nullptr, 0, nullptr);
    }
    return command;
}

void VulkanSubmission::retire() {
    if (!ready()) {
        throw std::logic_error("Vulkan submission is not complete");
    }
    owner_->destroy(*this);
}

void VulkanContext::end_commands(
    VkCommandBuffer command,
    const VulkanSemaphore* wait) {
    if (deferred_sequence_active_) {
        const VulkanSemaphore* effective_wait = wait;
        if (effective_wait == nullptr &&
            deferred_sequence_submissions_.empty() &&
            deferred_sequence_wait_.owner_ != nullptr) {
            effective_wait = &deferred_sequence_wait_;
        }
        VulkanSubmission submission =
            submit_commands(command, effective_wait);
        if (effective_wait == &deferred_sequence_wait_) {
            auto resources = std::make_unique<VulkanSubmission::Resources>();
            resources->wait = std::move(deferred_sequence_wait_);
            submission.resources_ = resources.release();
        }
        deferred_sequence_submissions_.push_back(std::move(submission));
        return;
    }
    VulkanSubmission submission = submit_commands(command, wait);
    submission.wait();
}

void VulkanContext::begin_deferred_sequence(VulkanSemaphore wait) {
    if (deferred_sequence_active_ || batch_command_ != VK_NULL_HANDLE ||
        wait.owner_ != this) {
        throw std::logic_error("invalid deferred Vulkan sequence start");
    }
    deferred_sequence_active_ = true;
    deferred_sequence_wait_ = std::move(wait);
}

VulkanSubmission VulkanContext::end_deferred_sequence(
    VulkanSemaphore signal) {
    if (!deferred_sequence_active_ || batch_command_ != VK_NULL_HANDLE ||
        signal.owner_ != this) {
        throw std::logic_error("invalid deferred Vulkan sequence end");
    }
    VkCommandBuffer command = begin_commands();
    VulkanSubmission final_submission;
    try {
        const VulkanSemaphore* wait =
            deferred_sequence_submissions_.empty() ?
                &deferred_sequence_wait_ : nullptr;
        final_submission = submit_commands(command, wait, &signal);
        auto resources = std::make_unique<VulkanSubmission::Resources>();
        resources->descriptor_sets =
            std::move(deferred_sequence_descriptors_);
        resources->deferred_buffers =
            std::move(deferred_sequence_buffers_);
        resources->prior_submissions =
            std::move(deferred_sequence_submissions_);
        resources->wait = std::move(deferred_sequence_wait_);
        resources->signal = std::move(signal);
        final_submission.resources_ = resources.release();
        deferred_sequence_active_ = false;
        return final_submission;
    } catch (...) {
        deferred_sequence_active_ = false;
        cancel_deferred_sequence();
        throw;
    }
}

void VulkanContext::cancel_deferred_sequence() noexcept {
    deferred_sequence_active_ = false;
    deferred_sequence_submissions_.clear();
    for (const VulkanBatchedDescriptor& descriptor :
         deferred_sequence_descriptors_) {
        if (descriptor.pipeline && descriptor.set) {
            descriptor.pipeline->cached_descriptor_sets_.push_back(
                descriptor.set);
        }
    }
    deferred_sequence_descriptors_.clear();
    for (const VulkanDeferredBuffer& buffer : deferred_sequence_buffers_) {
        recycle_or_destroy(buffer);
    }
    deferred_sequence_buffers_.clear();
    deferred_sequence_wait_ = VulkanSemaphore{};
}

#include <inferbridge/vulkan/context/vulkan_context_submit_commands.h>

#include <inferbridge/vulkan/context/vulkan_context_begin_batch.h>

#include <inferbridge/vulkan/context/vulkan_context_release_batch_resources.h>

#include <inferbridge/vulkan/context/vulkan_context_end_batch_async_variant_1.h>

void VulkanContext::end_batch() {
    if (batch_command_ == VK_NULL_HANDLE) {
        throw std::logic_error("no active Vulkan batch");
    }
    VkCommandBuffer command = batch_command_;
    batch_command_ = VK_NULL_HANDLE;
    batch_has_dispatch_ = false;
    end_commands(command);
    if (deferred_sequence_active_) {
        deferred_sequence_descriptors_.insert(
            deferred_sequence_descriptors_.end(),
            std::make_move_iterator(batch_descriptor_sets_.begin()),
            std::make_move_iterator(batch_descriptor_sets_.end()));
        batch_descriptor_sets_.clear();
        deferred_sequence_buffers_.insert(
            deferred_sequence_buffers_.end(),
            std::make_move_iterator(batch_deferred_buffers_.begin()),
            std::make_move_iterator(batch_deferred_buffers_.end()));
        batch_deferred_buffers_.clear();
        batch_buffer_access_.clear();
        batch_image_access_.clear();
        batch_image_layout_.clear();
    } else {
        release_batch_resources();
    }
}

#include <inferbridge/vulkan/context/vulkan_context_cancel_batch_variant_1.h>

void VulkanContext::copy_buffer_raw(
    VkBuffer source,
    VkBuffer destination,
    VkDeviceSize source_offset,
    VkDeviceSize destination_offset,
    VkDeviceSize bytes) {
    VkCommandBuffer command = begin_commands();
    const VkBufferCopy region{
        source_offset, destination_offset, bytes};
    vkCmdCopyBuffer(command, source, destination, 1, &region);
    end_commands(command);
}

#include <inferbridge/vulkan/context/vulkan_context_copy.h>

void VulkanContext::clear(VulkanBuffer& destination) {
    if (destination.owner_ != this ||
        destination.buffer_ == VK_NULL_HANDLE) {
        throw std::invalid_argument("invalid Vulkan buffer clear");
    }
    VkCommandBuffer command = begin_commands();
    vkCmdFillBuffer(
        command, destination.buffer_, 0, destination.size_, 0u);
    end_commands(command);
}

#include <inferbridge/vulkan/context/vulkan_context_upload_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_download_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_transfer_counters.h>

void global_transfer_counters(
    std::uint64_t& upload_bytes, std::uint64_t& download_bytes) {
    upload_bytes = g_tensor_upload_bytes.load(std::memory_order_relaxed);
    download_bytes = g_tensor_download_bytes.load(std::memory_order_relaxed);
}

#include <inferbridge/vulkan/context/vulkan_context_acquire_external_buffer_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_release_external_buffer_variant_2.h>

#include <inferbridge/linux_capture_vulkan_context.inl>


#include <inferbridge/vulkan/context/vulkan_context_acquire_external_image_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_release_external_image_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_create_pipeline_variant_4.h>

#include <inferbridge/vulkan/context/vulkan_context_create_pipeline_variant_1.h>

#include <inferbridge/vulkan/context/vulkan_context_create_pipeline_variant_3.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch_image_to_buffer.h>

#include <inferbridge/vulkan/context/vulkan_context_dispatch_buffer_to_image.h>

void VulkanContext::dispatch_resources(
    const VulkanPipeline& pipeline,
    const std::vector<VulkanDispatchResource>& resources,
    const void* push_constants,
    std::uint32_t push_constant_bytes,
    std::uint32_t group_x,
    std::uint32_t group_y,
    std::uint32_t group_z,
    const VulkanSemaphore* wait) {
    if (pipeline.owner_ != this ||
        resources.size() != pipeline.descriptor_types_.size() ||
        push_constant_bytes != pipeline.push_constant_bytes_ ||
        (push_constant_bytes && push_constants == nullptr) ||
        group_x == 0 || group_y == 0 || group_z == 0) {
        throw std::invalid_argument("invalid Vulkan dispatch");
    }

    VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
    if (!pipeline.cached_descriptor_sets_.empty()) {
        descriptor_set = pipeline.cached_descriptor_sets_.back();
        pipeline.cached_descriptor_sets_.pop_back();
    } else {
        const VkDescriptorSetAllocateInfo allocate_info{
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            nullptr,
            descriptor_pool_,
            1,
            &pipeline.descriptor_layout_,
        };
        check(
            vkAllocateDescriptorSets(
                device_, &allocate_info, &descriptor_set),
            "vkAllocateDescriptorSets");
    }
    std::vector<VkDescriptorBufferInfo> buffer_info(resources.size());
    std::vector<VkDescriptorImageInfo> image_info(resources.size());
    std::vector<VkWriteDescriptorSet> writes(resources.size());
    for (std::size_t index = 0; index < resources.size(); ++index) {
        const VkDescriptorType type =
            pipeline.descriptor_types_[index];
        const bool image_descriptor =
            type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ||
            type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
            type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        if (image_descriptor) {
            const VulkanImage* image = resources[index].image;
            if (image == nullptr || image->owner_ != this ||
                image->view_ == VK_NULL_HANDLE ||
                (type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                 image->sampler_ == VK_NULL_HANDLE)) {
                pipeline.cached_descriptor_sets_.push_back(
                    descriptor_set);
                throw std::invalid_argument(
                    "foreign or invalid Vulkan image");
            }
            image_info[index] = {
                type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                    ? image->sampler_
                    : VK_NULL_HANDLE,
                image->view_,
                type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                    ? VK_IMAGE_LAYOUT_GENERAL
                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            writes[index] = {
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                nullptr,
                descriptor_set,
                static_cast<std::uint32_t>(index),
                0,
                1,
                type,
                &image_info[index],
                nullptr,
                nullptr,
            };
            continue;
        }
        const VulkanBuffer* buffer = resources[index].buffer;
        if (buffer == nullptr || buffer->owner_ != this) {
            pipeline.cached_descriptor_sets_.push_back(descriptor_set);
            throw std::invalid_argument("foreign Vulkan buffer");
        }
        buffer_info[index] = {
            buffer->buffer_,
            0,
            buffer->size_,
        };
        writes[index] = {
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            nullptr,
            descriptor_set,
            static_cast<std::uint32_t>(index),
            0,
            1,
            type,
            nullptr,
            &buffer_info[index],
            nullptr,
        };
    }
    vkUpdateDescriptorSets(
        device_,
        static_cast<std::uint32_t>(writes.size()),
        writes.data(),
        0,
        nullptr);

    const bool batched = batch_command_ != VK_NULL_HANDLE;
    if (batched && wait != nullptr) {
        if (deferred_sequence_active_) {
            deferred_sequence_descriptors_.push_back(
                {const_cast<VulkanPipeline*>(&pipeline), descriptor_set});
        } else {
            pipeline.cached_descriptor_sets_.push_back(descriptor_set);
        }
        throw std::invalid_argument(
            "external wait is not supported inside a Vulkan batch");
    }
    VkCommandBuffer command =
        batched ? batch_command_ : begin_commands();
    const bool profile =
        !batched && profile_query_pool_ != VK_NULL_HANDLE;
    if (profile) {
        vkCmdResetQueryPool(
            command, profile_query_pool_, 0, 2);
        vkCmdWriteTimestamp(
            command,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            profile_query_pool_,
            0);
    }
    if (batched && batch_has_dispatch_ &&
        !track_resource_hazards_) {
        const VkMemoryBarrier barrier{
            VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            nullptr,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_ACCESS_SHADER_READ_BIT |
                VK_ACCESS_SHADER_WRITE_BIT,
        };
        vkCmdPipelineBarrier(
            command,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &barrier,
            0,
            nullptr,
            0,
            nullptr);
    } else if (batched) {
        std::unordered_map<VkBuffer, VkAccessFlags>
            current_buffers;
        std::unordered_map<VkImage, VkAccessFlags>
            current_images;
        std::unordered_map<VkImage, VkImageLayout> image_layouts;
        for (std::size_t index = 0;
             index < resources.size();
             ++index) {
            const VkAccessFlags access =
                pipeline.descriptor_access_[index];
            if (resources[index].buffer != nullptr) {
                current_buffers[
                    resources[index].buffer->buffer_] |= access;
            } else {
                const VkImage image =
                    resources[index].image->image_;
                current_images[image] |= access;
                image_layouts[image] =
                    pipeline.descriptor_types_[index] ==
                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                    ? VK_IMAGE_LAYOUT_GENERAL
                    : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            }
        }
        std::vector<VkBufferMemoryBarrier> buffer_barriers;
        std::vector<VkImageMemoryBarrier> image_barriers;
        for (const auto& [buffer, access] : current_buffers) {
            const auto previous =
                batch_buffer_access_.find(buffer);
            if (previous != batch_buffer_access_.end() &&
                ((previous->second | access) &
                 VK_ACCESS_SHADER_WRITE_BIT) != 0) {
                buffer_barriers.push_back(
                    {
                        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        nullptr,
                        previous->second,
                        access,
                        VK_QUEUE_FAMILY_IGNORED,
                        VK_QUEUE_FAMILY_IGNORED,
                        buffer,
                        0,
                        VK_WHOLE_SIZE,
                    });
            }
            batch_buffer_access_[buffer] = access;
        }
        for (const auto& [image, access] : current_images) {
            const auto previous =
                batch_image_access_.find(image);
            const VkImageLayout layout = image_layouts[image];
            if (previous != batch_image_access_.end()) {
                const VkImageLayout previous_layout =
                    batch_image_layout_.at(image);
                if (((previous->second | access) &
                     VK_ACCESS_SHADER_WRITE_BIT) == 0 &&
                    previous_layout == layout) {
                    batch_image_access_[image] = access;
                    continue;
                }
                image_barriers.push_back(
                    {
                        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        nullptr,
                        previous->second,
                        access,
                        previous_layout,
                        layout,
                        VK_QUEUE_FAMILY_IGNORED,
                        VK_QUEUE_FAMILY_IGNORED,
                        image,
                        {
                            VK_IMAGE_ASPECT_COLOR_BIT,
                            0,
                            1,
                            0,
                            1,
                        },
                    });
            }
            batch_image_access_[image] = access;
            batch_image_layout_[image] = layout;
        }
        if (!buffer_barriers.empty() ||
            !image_barriers.empty()) {
            vkCmdPipelineBarrier(
                command,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                nullptr,
                static_cast<std::uint32_t>(
                    buffer_barriers.size()),
                buffer_barriers.data(),
                static_cast<std::uint32_t>(
                    image_barriers.size()),
                image_barriers.data());
        }
    }
    vkCmdBindPipeline(
        command, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline_);
    vkCmdBindDescriptorSets(
        command,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        pipeline.layout_,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);
    if (push_constant_bytes) {
        vkCmdPushConstants(
            command,
            pipeline.layout_,
            VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            push_constant_bytes,
            push_constants);
    }
    vkCmdDispatch(command, group_x, group_y, group_z);
    if (profile) {
        vkCmdWriteTimestamp(
            command,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            profile_query_pool_,
            1);
    }
    if (batched) {
        batch_has_dispatch_ = true;
        batch_descriptor_sets_.push_back(
            {const_cast<VulkanPipeline*>(&pipeline), descriptor_set});
    } else {
        end_commands(command, wait);
        if (profile) {
            std::uint64_t timestamps[2]{};
            check(
                vkGetQueryPoolResults(
                    device_,
                    profile_query_pool_,
                    0,
                    2,
                    sizeof(timestamps),
                    timestamps,
                    sizeof(std::uint64_t),
                    VK_QUERY_RESULT_64_BIT |
                        VK_QUERY_RESULT_WAIT_BIT),
                "vkGetQueryPoolResults");
            record_profile(
                pipeline, timestamps[1] - timestamps[0]);
        }
        if (deferred_sequence_active_) {
            deferred_sequence_descriptors_.push_back(
                {const_cast<VulkanPipeline*>(&pipeline), descriptor_set});
        } else {
            pipeline.cached_descriptor_sets_.push_back(descriptor_set);
        }
    }
}

void VulkanContext::destroy(VulkanBuffer& buffer) noexcept {
    if (deferred_sequence_active_ &&
        (buffer.buffer_ != VK_NULL_HANDLE || buffer.memory_ != VK_NULL_HANDLE)) {
        // The sequence records later uses only after the logical lifetime has
        // ended. Reusing the still-live VkBuffer is safe because queue order
        // and the sequence-wide memory barriers serialize those uses. Never
        // destroy pending resources; retain the high-water allocation pool.
        if (buffer.mapped_) {
            host_buffer_pool_.push_back({
                buffer.buffer_, buffer.memory_, buffer.mapped_,
                buffer.size_, buffer.cacheable_});
            pooled_host_bytes_ += buffer.size_;
        } else {
            device_buffer_pool_.push_back({
                buffer.buffer_, buffer.memory_, nullptr,
                buffer.size_, buffer.cacheable_});
            pooled_device_bytes_ += buffer.size_;
        }
        buffer.owner_ = nullptr;
        buffer.buffer_ = VK_NULL_HANDLE;
        buffer.memory_ = VK_NULL_HANDLE;
        buffer.mapped_ = nullptr;
        buffer.size_ = 0;
        buffer.cacheable_ = false;
        return;
    }
    if (batch_command_ != VK_NULL_HANDLE &&
        (buffer.buffer_ != VK_NULL_HANDLE ||
         buffer.memory_ != VK_NULL_HANDLE)) {
        batch_deferred_buffers_.push_back({
            buffer.buffer_, buffer.memory_, buffer.mapped_,
            buffer.size_, buffer.cacheable_});
        buffer.owner_ = nullptr;
        buffer.buffer_ = VK_NULL_HANDLE;
        buffer.memory_ = VK_NULL_HANDLE;
        buffer.mapped_ = nullptr;
        buffer.size_ = 0;
        buffer.cacheable_ = false;
        return;
    }
    recycle_or_destroy(
        {
            buffer.buffer_,
            buffer.memory_,
            buffer.mapped_,
            buffer.size_,
            buffer.cacheable_,
        });
    buffer.owner_ = nullptr;
    buffer.buffer_ = VK_NULL_HANDLE;
    buffer.memory_ = VK_NULL_HANDLE;
    buffer.mapped_ = nullptr;
    buffer.size_ = 0;
    buffer.cacheable_ = false;
}

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_5.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_release_submission_resources_variant_2.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_1.h>

void VulkanContext::recycle_or_destroy(
    VulkanDeferredBuffer buffer) noexcept {
    // Depth Pro keeps several 768x768x256 feature maps live. A multi-gigabyte
    // cache competes with those maps on 8 GiB adapters, so retain only enough
    // scratch memory to reuse encoder and small decoder allocations.
    constexpr VkDeviceSize maximum_device_pool_bytes =
        VkDeviceSize{256} * 1024 * 1024;
    constexpr VkDeviceSize maximum_host_pool_bytes =
        VkDeviceSize{64} * 1024 * 1024;
    if (buffer.cacheable && buffer.buffer && buffer.memory) {
        if (buffer.mapped &&
            buffer.size <= maximum_host_pool_bytes - pooled_host_bytes_) {
            pooled_host_bytes_ += buffer.size;
            host_buffer_pool_.push_back(buffer);
            return;
        }
        if (!buffer.mapped &&
            buffer.size <=
                maximum_device_pool_bytes - pooled_device_bytes_) {
            pooled_device_bytes_ += buffer.size;
            device_buffer_pool_.push_back(buffer);
            return;
        }
    }
    if (buffer.mapped) {
        vkUnmapMemory(device_, buffer.memory);
    }
    if (buffer.buffer) {
        vkDestroyBuffer(device_, buffer.buffer, nullptr);
    }
    if (buffer.memory) {
        vkFreeMemory(device_, buffer.memory, nullptr);
    }
}

#include <inferbridge/vulkan/context/vulkan_context_discard.h>

#include <inferbridge/vulkan/context/vulkan_context_destroy_variant_3.h>

}  // namespace depth_pro_native
