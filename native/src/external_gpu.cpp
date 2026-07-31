
#include "external_gpu.h"

#include "gpu_io.h"
#include "gpu_model.h"
#include "graph_gpu.h"
#include "model.h"
#include "operators.h"
#include "vulkan.h"

#include <array>
#include <atomic>
#include <cstring>
#include <mutex>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#endif

namespace depth_pro_native {
namespace {

#if defined(_WIN32)
using Microsoft::WRL::ComPtr;
constexpr std::uint32_t kMaxInFlightJobs = 3u;

void check_hresult(HRESULT result, const char* operation) {
    if (FAILED(result)) {
        throw std::runtime_error(
            std::string(operation) + " failed with HRESULT " +
            std::to_string(static_cast<long>(result)));
    }
}
ComPtr<ID3D12Device> matching_d3d12_device(std::uint64_t luid) {
    if (luid == 0u) return {};
    ComPtr<IDXGIFactory6> factory;
    check_hresult(
        CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
        "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT result = factory->EnumAdapters1(index, &adapter);
        if (result == DXGI_ERROR_NOT_FOUND) break;
        check_hresult(result, "EnumAdapters1");
        DXGI_ADAPTER_DESC1 description{};
        check_hresult(adapter->GetDesc1(&description), "GetDesc1");
        std::uint64_t candidate = 0u;
        std::memcpy(&candidate, &description.AdapterLuid, sizeof(candidate));
        if (candidate != luid) continue;
        ComPtr<ID3D12Device> device;
        check_hresult(
            D3D12CreateDevice(
                adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&device)),
            "D3D12CreateDevice");
        return device;
    }
    return {};
}

void validate_texture(ID3D12Device* device,std::uintptr_t handle,std::uint32_t width,std::uint32_t height,DXGI_FORMAT format,const char* op){ComPtr<ID3D12Resource> r;check_hresult(device->OpenSharedHandle(reinterpret_cast<HANDLE>(handle),IID_PPV_ARGS(&r)),op);const auto d=r->GetDesc();if(d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D||d.Width!=width||d.Height!=height||d.DepthOrArraySize!=1u||d.MipLevels!=1u||d.SampleDesc.Count!=1u||d.Format!=format)throw std::invalid_argument("Depth Pro shared texture descriptor mismatch");}
class ExternalJobImpl final:public ExternalJob{public:ExternalJobImpl(std::shared_ptr<ExternalGpu> owner,VulkanImage input,VulkanImage output,VulkanSubmission submission):owner_(std::move(owner)),input_(std::move(input)),output_(std::move(output)),submission_(std::move(submission)){}~ExternalJobImpl()override{try{submission_.wait();}catch(...){}submission_={};output_={};input_={};}ExternalJobState state()const override{if(cancelled_.load())return ExternalJobState::cancelled;if(completed_.load())return ExternalJobState::complete;if(!submission_.ready())return ExternalJobState::running;completed_.store(true);return ExternalJobState::complete;}void cancel()override{cancelled_.store(true);}private:std::shared_ptr<ExternalGpu> owner_;VulkanImage input_,output_;mutable VulkanSubmission submission_;std::atomic<bool> cancelled_{false};mutable std::atomic<bool> completed_{false};};
#endif

class ExternalGpuImpl final : public ExternalGpu {
public:
    ExternalGpuImpl(
        const std::string& path, float forced_fov, std::uint32_t index)
        : model_(path), context_(index), gpu_model_(model_, context_),
          operators_(context_), io_(context_),
          zero_(context_.create_device_buffer(1024u * sizeof(float))),
          forced_fov_degrees_(forced_fov)
#if defined(_WIN32)
          , d3d12_(matching_d3d12_device(context_.adapter_luid()))
#endif
          {
        // Deferred asynchronous recording cannot benchmark kernels by host
        // call duration. Select the conservative universally-correct kernel
        // shape and avoid launching the synchronous tuner's extra workloads.
        gpu_model_.set_linear_tuning(false, false, 0u);
        const std::vector<float> zeros(1024u, 0.0f);
        context_.upload(zero_, zeros.data(), zeros.size() * sizeof(float));
        if (forced_fov_degrees_ > 0.0f) {
            forced_fov_ = context_.create_device_buffer(sizeof(float));
            context_.upload(
                forced_fov_, &forced_fov_degrees_, sizeof(float));
        }
    }

    ExternalGpuCapabilities capabilities() const override {
#if defined(_WIN32)
        const auto& capabilities = context_.external_capabilities();
        const bool available = d3d12_ != nullptr &&
            capabilities.d3d12_resource_import &&
            capabilities.d3d12_fence_import &&
            capabilities.d3d12_bgra8_sampled_image_import &&
            capabilities.d3d12_r32_storage_image_import;
        return {available, available ? context_.adapter_luid() : 0u,
                available ? kMaxInFlightJobs : 0u};
#else
        return {};
#endif
    }

    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) override {
#if !defined(_WIN32)
        (void)request;
        throw std::runtime_error("Depth Pro D3D12 interop is unavailable");
#else
        if (!capabilities().available) {
            throw std::runtime_error(
                "complete Depth Pro D3D12/Vulkan interop is unavailable");
        }
        if (!request.shared_texture_handle || !request.wait_fence_handle ||
            !request.output_texture_handle || !request.signal_fence_handle ||
            !request.width || !request.height ||
            request.output_width != request.width ||
            request.output_height != request.height) {
            throw std::invalid_argument("invalid Depth Pro GPU request");
        }
        validate_texture(d3d12_.Get(),request.shared_texture_handle,request.width,request.height,DXGI_FORMAT_B8G8R8A8_UNORM,"OpenSharedHandle(Depth Pro input)");
        validate_texture(d3d12_.Get(),request.output_texture_handle,request.output_width,request.output_height,DXGI_FORMAT_R32_FLOAT,"OpenSharedHandle(Depth Pro output)");
        try {
            std::lock_guard<std::mutex> lock(record_mutex_);
            VulkanImage output=context_.import_d3d12_image(reinterpret_cast<void*>(request.output_texture_handle),request.output_width,request.output_height,VK_FORMAT_R32_SFLOAT,VK_IMAGE_USAGE_STORAGE_BIT|VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
            VulkanImage input = context_.import_d3d12_image(
                reinterpret_cast<void*>(request.shared_texture_handle),
                request.width, request.height,
                VK_FORMAT_B8G8R8A8_UNORM,
                VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            VulkanSemaphore wait = context_.import_d3d12_fence(
                reinterpret_cast<void*>(request.wait_fence_handle),
                request.wait_fence_value);
            VulkanSemaphore signal=context_.import_d3d12_fence(reinterpret_cast<void*>(request.signal_fence_handle),request.signal_fence_value);
            context_.begin_deferred_sequence(std::move(wait));
            VulkanSubmission submission;
            try {
                context_.batch([&] {
                    context_.acquire_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.acquire_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                });
                VulkanBuffer x0 = context_.create_device_buffer(
                    std::uint64_t(3) * 1536 * 1536 * sizeof(float));
                VulkanBuffer x1 = context_.create_device_buffer(
                    std::uint64_t(3) * 768 * 768 * sizeof(float));
                VulkanBuffer x2 = context_.create_device_buffer(
                    std::uint64_t(3) * 384 * 384 * sizeof(float));
                io_.preprocess_x0(x0, input);
                operators_.bilinear_half_pixel(
                    x1, x0, 1536, 1536, 768, 768, 3);
                operators_.bilinear_half_pixel(
                    x2, x0, 1536, 1536, 384, 384, 3);
                VulkanBuffer patches = context_.create_device_buffer(
                    std::uint64_t(35) * 3 * 384 * 384 * sizeof(float));
                io_.assemble_pyramid(patches, x0, x1, x2);
                const VulkanBuffer* forced =
                    forced_fov_.handle() != VK_NULL_HANDLE ?
                        &forced_fov_ : nullptr;
                GpuDeviceInferenceOutput inference = infer_gpu_device(
                    context_, gpu_model_, operators_, patches, x2, zero_,
                    forced, request.width, request.height);
                context_.batch([&] {
                    io_.write_depth(output, inference.depth);
                    context_.release_external_image(
                        input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        VK_ACCESS_SHADER_READ_BIT);
                    context_.release_external_image(
                        output, VK_IMAGE_LAYOUT_GENERAL,
                        VK_ACCESS_SHADER_WRITE_BIT);
                });
                submission =
                    context_.end_deferred_sequence(std::move(signal));
            } catch (...) {
                context_.cancel_deferred_sequence();
                throw;
            }
            return std::make_shared<ExternalJobImpl>(shared_from_this(),std::move(input),std::move(output),std::move(submission));
        } catch (...) { throw; }
#endif
    }

    void transfer_counters(
        std::uint64_t& upload_bytes,
        std::uint64_t& download_bytes) const override {
        context_.transfer_counters(upload_bytes, download_bytes);
    }

private:
    ModelFile model_;
    VulkanContext context_;
    GpuModel gpu_model_;
    VulkanOperators operators_;
    GpuIo io_;
    VulkanBuffer zero_;
    VulkanBuffer forced_fov_;
    float forced_fov_degrees_ = 0.0f;
#if defined(_WIN32)
    ComPtr<ID3D12Device> d3d12_;
    std::mutex record_mutex_;
#endif
};

}  // namespace

std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string& path, float forced_fov_degrees,
    std::uint32_t index) {
    return std::make_shared<ExternalGpuImpl>(
        path, forced_fov_degrees, index);
}

ExternalGpuCapabilities probe_external_gpu(std::uint32_t index) {
#if defined(_WIN32)
    VulkanContext context(index);
    const auto device = matching_d3d12_device(context.adapter_luid());
    const auto& capabilities = context.external_capabilities();
    const bool available = device != nullptr &&
        capabilities.d3d12_resource_import &&
        capabilities.d3d12_fence_import &&
        capabilities.d3d12_bgra8_sampled_image_import &&
        capabilities.d3d12_r32_storage_image_import;
    return {available, available ? context.adapter_luid() : 0u,
            available ? kMaxInFlightJobs : 0u};
#else
    (void)index;
    return {};
#endif
}

}  // namespace depth_pro_native
