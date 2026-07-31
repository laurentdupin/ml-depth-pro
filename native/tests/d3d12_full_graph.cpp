#include "depth_pro_native.h"
#include "inferbridge_harness.h"
#include "gpu_model.h"
#include "graph_gpu.h"
#include "gpu_io.h"
#include "model.h"
#include "operators.h"
#include "vulkan.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
double maximum_submit_return_ms = 0.0;

using Microsoft::WRL::ComPtr;

void check(HRESULT value, const char* operation) {
    if (FAILED(value)) throw std::runtime_error(
        std::string(operation) + " failed: " +
        std::to_string(static_cast<long>(value)));
}
void check(ibrh_result value, const char* operation) {
    if (value != IBRH_OK) throw std::runtime_error(
        std::string(operation) + " failed: " +
        std::to_string(static_cast<unsigned>(value)));
}
void check(depth_pro_status value, const char* operation) {
    if (value != DEPTH_PRO_STATUS_OK) throw std::runtime_error(
        std::string(operation) + " failed: " + depth_pro_last_error());
}

struct Capture {
    ComPtr<ID3D12Resource> texture;
    ComPtr<ID3D12Resource> upload;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> commands;
    ComPtr<ID3D12Fence> fence;
    HANDLE texture_handle = nullptr;
    HANDLE fence_handle = nullptr;
    std::uint64_t value = 1u;
};

std::vector<std::uint8_t> pixels(
    std::uint32_t width, std::uint32_t height, std::uint32_t frame) {
    std::vector<std::uint8_t> result(
        static_cast<std::size_t>(width) * height * 4u);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t offset =
                (static_cast<std::size_t>(y) * width + x) * 4u;
            result[offset] = static_cast<std::uint8_t>((x * 11u + y + frame) & 255u);
            result[offset + 1u] = static_cast<std::uint8_t>((x + y * 7u + frame * 3u) & 255u);
            result[offset + 2u] = static_cast<std::uint8_t>((x * 3u + y * 5u + frame * 13u) & 255u);
            result[offset + 3u] = 255u;
        }
    }
    return result;
}

Capture upload_texture(
    ID3D12Device* device, ID3D12CommandQueue* queue,
    const std::vector<std::uint8_t>& source,
    std::uint32_t width, std::uint32_t height, bool signal = true) {
    Capture result;
    const D3D12_HEAP_PROPERTIES default_heap{
        D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC texture_desc{
        D3D12_RESOURCE_DIMENSION_TEXTURE2D, 0, width, height, 1, 1,
        DXGI_FORMAT_B8G8R8A8_UNORM, {1, 0},
        D3D12_TEXTURE_LAYOUT_UNKNOWN,
        D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET};
    check(device->CreateCommittedResource(
        &default_heap, D3D12_HEAP_FLAG_SHARED, &texture_desc,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&result.texture)), "CreateCommittedResource(input)");

    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0u;
    UINT64 row_bytes = 0u;
    UINT64 upload_bytes = 0u;
    device->GetCopyableFootprints(
        &texture_desc, 0, 1, 0, &footprint, &rows,
        &row_bytes, &upload_bytes);
    const D3D12_HEAP_PROPERTIES upload_heap{
        D3D12_HEAP_TYPE_UPLOAD, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC buffer_desc{
        D3D12_RESOURCE_DIMENSION_BUFFER, 0, upload_bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE};
    check(device->CreateCommittedResource(
        &upload_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
        D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(&result.upload)), "CreateCommittedResource(upload)");
    std::uint8_t* mapped = nullptr;
    const D3D12_RANGE no_read{0, 0};
    check(result.upload->Map(
        0, &no_read, reinterpret_cast<void**>(&mapped)), "Map(upload)");
    for (std::uint32_t y = 0; y < height; ++y)
        std::memcpy(
            mapped + footprint.Offset +
                static_cast<std::size_t>(y) * footprint.Footprint.RowPitch,
            source.data() + static_cast<std::size_t>(y) * width * 4u,
            static_cast<std::size_t>(width) * 4u);
    result.upload->Unmap(0, nullptr);

    check(device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&result.allocator)),
        "CreateCommandAllocator");
    check(device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, result.allocator.Get(), nullptr,
        IID_PPV_ARGS(&result.commands)), "CreateCommandList");
    const D3D12_TEXTURE_COPY_LOCATION destination{
        result.texture.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    D3D12_TEXTURE_COPY_LOCATION upload_location{};
    upload_location.pResource = result.upload.Get();
    upload_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    upload_location.PlacedFootprint = footprint;
    result.commands->CopyTextureRegion(
        &destination, 0, 0, 0, &upload_location, nullptr);
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = result.texture.Get();
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    result.commands->ResourceBarrier(1, &barrier);
    check(result.commands->Close(), "Close(upload list)");
    ID3D12CommandList* lists[] = {result.commands.Get()};
    queue->ExecuteCommandLists(1, lists);
    check(device->CreateFence(
        0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&result.fence)),
        "CreateFence(input)");
    if (signal) check(queue->Signal(result.fence.Get(), result.value), "Signal(input)");
    check(device->CreateSharedHandle(
        result.texture.Get(), nullptr, GENERIC_ALL, nullptr,
        &result.texture_handle), "CreateSharedHandle(input)");
    check(device->CreateSharedHandle(
        result.fence.Get(), nullptr, GENERIC_ALL, nullptr,
        &result.fence_handle), "CreateSharedHandle(input fence)");
    return result;
}

void close_capture(Capture& value) {
    if (value.texture_handle) CloseHandle(value.texture_handle);
    if (value.fence_handle) CloseHandle(value.fence_handle);
    value.texture_handle = nullptr;
    value.fence_handle = nullptr;
}

void wait_fence(ID3D12Device* device, const ibrh_synchronization& ready) {
    ComPtr<ID3D12Fence> fence;
    check(device->OpenSharedHandle(
        reinterpret_cast<HANDLE>(ready.native_handle),
        IID_PPV_ARGS(&fence)), "OpenSharedHandle(output fence)");
    if (fence->GetCompletedValue() >= ready.value) return;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("CreateEvent failed");
    check(fence->SetEventOnCompletion(ready.value, event), "SetEventOnCompletion");
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
}

double d3d12_heartbeat_fps(
    ID3D12Device* device, ID3D12CommandQueue* queue) {
    ComPtr<ID3D12Fence> fence;
    check(device->CreateFence(
        0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)),
        "CreateFence(heartbeat)");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) throw std::runtime_error("CreateEvent(heartbeat) failed");
    const auto start = std::chrono::steady_clock::now();
    std::uint64_t value = 0u;
    do {
        check(queue->Signal(fence.Get(), ++value), "Signal(heartbeat)");
        check(fence->SetEventOnCompletion(value, event),
              "SetEventOnCompletion(heartbeat)");
        if (WaitForSingleObject(event, 5000u) != WAIT_OBJECT_0) {
            CloseHandle(event);
            throw std::runtime_error("D3D12 heartbeat stalled for five seconds");
        }
    } while (std::chrono::duration<double>(
                 std::chrono::steady_clock::now() - start).count() < 1.0);
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    CloseHandle(event);
    return static_cast<double>(value) / seconds;
}

std::vector<float> read_output(
    ID3D12Device* device, ID3D12CommandQueue* queue,
    const ibrh_output_descriptor& output) {
    wait_fence(device, output.ready);
    ComPtr<ID3D12Resource> texture;
    check(device->OpenSharedHandle(
        reinterpret_cast<HANDLE>(output.resource.native_handle),
        IID_PPV_ARGS(&texture)), "OpenSharedHandle(output texture)");
    const D3D12_RESOURCE_DESC description = texture->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT rows = 0;
    UINT64 row_bytes = 0;
    UINT64 bytes = 0;
    device->GetCopyableFootprints(
        &description, 0, 1, 0, &footprint, &rows, &row_bytes, &bytes);
    const D3D12_HEAP_PROPERTIES heap{
        D3D12_HEAP_TYPE_READBACK, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    const D3D12_RESOURCE_DESC buffer{
        D3D12_RESOURCE_DIMENSION_BUFFER, 0, bytes, 1, 1, 1,
        DXGI_FORMAT_UNKNOWN, {1, 0}, D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        D3D12_RESOURCE_FLAG_NONE};
    ComPtr<ID3D12Resource> readback;
    check(device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &buffer,
        D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&readback)), "CreateCommittedResource(readback)");
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check(device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)),
        "CreateCommandAllocator(readback)");
    check(device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
        IID_PPV_ARGS(&list)), "CreateCommandList(readback)");
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = texture.Get();
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = readback.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint = footprint;
    list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    check(list->Close(), "Close(readback list)");
    ID3D12CommandList* lists[] = {list.Get()};
    queue->ExecuteCommandLists(1, lists);
    ComPtr<ID3D12Fence> done;
    check(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done)),
          "CreateFence(readback)");
    check(queue->Signal(done.Get(), 1), "Signal(readback)");
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    check(done->SetEventOnCompletion(1, event), "SetEventOnCompletion(readback)");
    WaitForSingleObject(event, INFINITE);
    CloseHandle(event);
    const std::uint8_t* mapped = nullptr;
    const D3D12_RANGE range{0, static_cast<SIZE_T>(bytes)};
    check(readback->Map(0, &range, reinterpret_cast<void**>(
        const_cast<std::uint8_t**>(&mapped))), "Map(readback)");
    std::vector<float> result(
        static_cast<std::size_t>(output.resource.width) * output.resource.height);
    for (std::uint32_t y = 0; y < output.resource.height; ++y)
        std::memcpy(
            result.data() + static_cast<std::size_t>(y) * output.resource.width,
            mapped + footprint.Offset +
                static_cast<std::size_t>(y) * footprint.Footprint.RowPitch,
            static_cast<std::size_t>(output.resource.width) * sizeof(float));
    readback->Unmap(0, nullptr);
    return result;
}

struct SelectedDevice {
    ComPtr<ID3D12Device> device;
    std::uint64_t luid = 0u;
    std::string luid_json;
    std::string name;
};

SelectedDevice select_device(const ibrh_api& api) {
    ComPtr<IDXGIFactory6> factory;
    check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)), "CreateDXGIFactory2");
    for (UINT index = 0;; ++index) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(index, &adapter) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 description{};
        check(adapter->GetDesc1(&description), "GetDesc1");
        const auto* bytes = reinterpret_cast<const unsigned char*>(
            &description.AdapterLuid);
        char json[40]{};
        std::snprintf(json, sizeof(json),
            "{\"luid\":\"%02x%02x%02x%02x%02x%02x%02x%02x\"}",
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7]);
        ibrh_runtime_create_request request{};
        request.struct_size = sizeof(request);
        request.api_version = IBRH_CURRENT_API_VERSION;
        request.backend = {"native", 6u};
        request.requested_device_json = {json, std::strlen(json)};
        ibrh_runtime* runtime = nullptr;
        if (api.runtime_create(sizeof(request), &request, &runtime) != IBRH_OK)
            continue;
        api.runtime_destroy(runtime);
        SelectedDevice result;
        check(D3D12CreateDevice(
            adapter.Get(), D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&result.device)), "D3D12CreateDevice");
        std::memcpy(&result.luid, &description.AdapterLuid, sizeof(result.luid));
        result.luid_json = json;
        char narrow[128]{};
        WideCharToMultiByte(CP_UTF8, 0, description.Description, -1,
                            narrow, sizeof(narrow), nullptr, nullptr);
        result.name = narrow;
        return result;
    }
    throw std::runtime_error("no D3D12 adapter accepted by Depth Pro Vulkan");
}

struct Submitted {
    ibrh_job* job = nullptr;
    ibrh_output_lease* lease = nullptr;
    ibrh_output_descriptor output{};
};

Submitted submit(
    const ibrh_api& api, ibrh_model* model,
    Capture& capture, std::uint32_t width, std::uint32_t height,
    std::uint64_t frame, const std::string& parameters,
    bool acquire_output = true) {
    ibrh_resource resource{};
    resource.struct_size = sizeof(resource);
    resource.api_version = IBRH_CURRENT_API_VERSION;
    resource.domain = IBRH_RESOURCE_DOMAIN_D3D12;
    resource.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
    resource.access = IBRH_RESOURCE_ACCESS_READ;
    resource.pixel_format = IBRH_PIXEL_BGRA8;
    resource.width = width;
    resource.height = height;
    resource.depth = 1u;
    resource.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    resource.native_handle = reinterpret_cast<std::uintptr_t>(capture.texture_handle);
    ibrh_synchronization wait{};
    wait.struct_size = sizeof(wait);
    wait.api_version = IBRH_CURRENT_API_VERSION;
    wait.kind = IBRH_SYNC_D3D12_FENCE;
    wait.operation = IBRH_SYNC_WAIT;
    wait.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    wait.native_handle = reinterpret_cast<std::uintptr_t>(capture.fence_handle);
    wait.value = capture.value;
    ibrh_submit_request request{};
    request.struct_size = sizeof(request);
    request.api_version = IBRH_CURRENT_API_VERSION;
    request.inputs = &resource;
    request.input_count = 1u;
    request.synchronizations = &wait;
    request.synchronization_count = 1u;
    request.source_frame_id = frame;
    request.timestamp_ns = 900000u + frame;
    request.parameters_json = {parameters.data(), parameters.size()};
    Submitted result;
    const auto submit_start = std::chrono::steady_clock::now();
    const ibrh_result submit_result =
        api.submit(model, sizeof(request), &request, &result.job);
    const double submit_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - submit_start).count();
    maximum_submit_return_ms = std::max(maximum_submit_return_ms, submit_ms);
    std::cout << "submit_ms=" << submit_ms << '\n';
    if (submit_ms >= 5.0)
        throw std::runtime_error(
            "Depth Pro GPU submit exceeded the 5 ms async contract");
    if (submit_result != IBRH_OK) {
        char message[1024]{};
        size_t required = 0u;
        (void)api.get_last_error(
            nullptr, message, sizeof(message), &required);
        throw std::runtime_error(
            std::string("submit failed: ") + message + " (" +
            std::to_string(static_cast<unsigned>(submit_result)) + ")");
    }
    close_capture(capture);
    if (!acquire_output) return result;
    ibrh_job_status queued_status{};
    for (std::uint32_t attempt = 0u; attempt < 10000u; ++attempt) {
        check(api.job_poll(
            result.job, sizeof(queued_status), &queued_status),
            "job_poll(recording)");
        if (queued_status.state != IBRH_JOB_QUEUED) break;
        Sleep(1);
    }
    if (queued_status.state == IBRH_JOB_QUEUED ||
        queued_status.state == IBRH_JOB_FAILED ||
        queued_status.state == IBRH_JOB_CANCELLED)
        throw std::runtime_error("Depth Pro asynchronous recording failed");
    check(api.output_acquire(
        result.job, 0, sizeof(result.output), &result.output, &result.lease),
        "output_acquire");
    if (result.output.source_frame_id != frame ||
        result.output.timestamp_ns != request.timestamp_ns ||
        result.output.resource.domain != IBRH_RESOURCE_DOMAIN_D3D12 ||
        result.output.resource.pixel_format != IBRH_PIXEL_DEPTH_FLOAT32 ||
        result.output.resource.width != width ||
        result.output.resource.height != height ||
        result.output.ready.kind != IBRH_SYNC_D3D12_FENCE)
        throw std::runtime_error("Depth Pro output descriptor correlation failed");
    return result;
}

void normalize(std::vector<float>& values) {
    const auto bounds = std::minmax_element(values.begin(), values.end());
    if (!std::isfinite(*bounds.first) || !std::isfinite(*bounds.second) ||
        !(*bounds.second > *bounds.first))
        throw std::runtime_error("Depth Pro output is not finite and varying");
    const float low = *bounds.first;
    const float denominator = 25.0f - low;
    if (!(denominator > 0.0f))
        throw std::runtime_error("Depth Pro worker normalization is invalid");
    for (float& value : values) value = (value - low) / denominator;
}

std::filesystem::path model_path() {
    if (const char* value = std::getenv("DEPTH_PRO_MODEL")) return value;
    return {};
}
}  // namespace

int main() try {
    std::cout << std::unitbuf;
    std::cerr << std::unitbuf;
    const auto model_file = model_path();
    if (model_file.empty() || !std::filesystem::exists(model_file)) return 77;
    ibrh_api api{};
    check(ibrh_get_api(IBRH_CURRENT_API_VERSION, sizeof(api), &api), "ibrh_get_api");
    ibrh_capabilities capabilities{};
    check(api.query_capabilities(sizeof(capabilities), &capabilities), "capabilities");
    const std::uint64_t required = IBRH_CAP_GPU_RESOURCES |
        IBRH_CAP_EXTERNAL_SYNCHRONIZATION | IBRH_CAP_GPU_RESIDENT_OUTPUT;
    if ((capabilities.flags & required) != required ||
        capabilities.maximum_in_flight_jobs != 3u)
        throw std::runtime_error("Depth Pro GPU capability contract is incomplete");
    SelectedDevice selected = select_device(api);
    std::cout << "device=" << selected.name
              << " luid=" << selected.luid_json << '\n';
    {
        depth_pro_native::VulkanContext scheduling_probe(0u, true);
        if (scheduling_probe.adapter_luid() != selected.luid)
            throw std::runtime_error(
                "Depth Pro cooperative queue probe selected the wrong LUID");
    }
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    check(selected.device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue)),
          "CreateCommandQueue");
    constexpr std::uint32_t width = 320u;
    constexpr std::uint32_t height = 180u;
    const bool learned_fov =
        std::getenv("DEPTH_PRO_FOV_MODE") != nullptr &&
        std::string(std::getenv("DEPTH_PRO_FOV_MODE")) == "learned";
    const float reference_fov = learned_fov ? 0.0f : 63.0f;
    const std::string parameters = learned_fov ?
        "{\"FovEstimation\":\"YES\"}" :
        "{\"FovEstimation\":\"NO\",\"FovForcedValue\":\"63\"}";
    const auto reused_pixels = pixels(width, height, 4);
    const std::string model_text = model_file.string();
    std::vector<float> reference(
        static_cast<std::size_t>(width) * height);
    std::vector<float> case_a;
    std::vector<float> case_b;
    std::vector<float> case_c;
    {
        depth_pro_context* cpu = nullptr;
        check(depth_pro_create_vulkan(model_text.c_str(), 0, &cpu),
              "depth_pro_create_vulkan(reference)");
        float reference_focal = 0.0f;
        check(depth_pro_infer_bgra8_f32(
            cpu, reused_pixels.data(), width * 4u,
            static_cast<int32_t>(width), static_cast<int32_t>(height),
            reference_fov,
            reference.data(), reference.size(), &reference_focal),
            "depth_pro_infer_bgra8_f32(reference)");
        depth_pro_destroy(cpu);
        normalize(reference);
    }
    {
        Capture capture = upload_texture(
            selected.device.Get(), queue.Get(), reused_pixels,
            width, height);
        depth_pro_native::VulkanContext context(0u);
        depth_pro_native::GpuIo io(context);
        depth_pro_native::VulkanOperators operators(context);
        auto input = context.import_d3d12_image(
            capture.texture_handle, width, height,
            VK_FORMAT_B8G8R8A8_UNORM,
            VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
        auto wait = context.import_d3d12_fence(
            capture.fence_handle, capture.value);
        auto signal = context.import_d3d12_fence(
            capture.fence_handle, capture.value + 1u);
        auto x0 = context.create_device_buffer(
            std::uint64_t(3) * 1536 * 1536 * sizeof(float));
        auto submission = context.batch_async(
            std::move(wait), std::move(signal), [&] {
                context.acquire_external_image(
                    input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_READ_BIT);
                io.preprocess_x0(x0, input);
                context.release_external_image(
                    input, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    VK_ACCESS_SHADER_READ_BIT);
            });
        submission.wait();
        std::vector<float> actual(std::uint64_t(3) * 1536 * 1536);
        std::vector<float> expected_x0(actual.size());
        context.download(x0, actual.data(), actual.size() * sizeof(float));
        float maximum = 0.0f;
        for (std::uint32_t channel = 0; channel < 3u; ++channel) {
            for (std::uint32_t y = 0; y < 1536u; ++y) {
                const float raw_y = (static_cast<float>(y) + 0.5f) *
                    height / 1536.0f - 0.5f;
                const float sy = std::clamp(
                    raw_y, 0.0f, static_cast<float>(height - 1u));
                const auto y0 = static_cast<std::uint32_t>(sy);
                const auto y1 = std::min(y0 + 1u, height - 1u);
                const float wy = sy - y0;
                for (std::uint32_t x = 0; x < 1536u; ++x) {
                    const float raw_x = (static_cast<float>(x) + 0.5f) *
                        width / 1536.0f - 0.5f;
                    const float sx = std::clamp(
                        raw_x, 0.0f, static_cast<float>(width - 1u));
                    const auto x0_index = static_cast<std::uint32_t>(sx);
                    const auto x1_index = std::min(x0_index + 1u, width - 1u);
                    const float wx = sx - x0_index;
                    const auto value = [&](std::uint32_t py, std::uint32_t px) {
                        return reused_pixels[
                            (std::uint64_t(py) * width + px) * 4u + channel] /
                            255.0f * 2.0f - 1.0f;
                    };
                    const float expected =
                        (value(y0, x0_index) * (1.0f - wx) +
                         value(y0, x1_index) * wx) * (1.0f - wy) +
                        (value(y1, x0_index) * (1.0f - wx) +
                         value(y1, x1_index) * wx) * wy;
                    const auto index =
                        (std::uint64_t(channel) * 1536u + y) * 1536u + x;
                    expected_x0[static_cast<std::size_t>(index)] = expected;
                    maximum = std::max(maximum,
                        std::abs(actual[static_cast<std::size_t>(index)] -
                                 expected));
                }
            }
        }
        close_capture(capture);
        std::cout << "preprocess maximum=" << maximum << '\n';
        if (maximum > 1.0e-4f)
            throw std::runtime_error("Depth Pro GPU preprocessing mismatch");

        auto x1 = context.create_device_buffer(
            std::uint64_t(3) * 768 * 768 * sizeof(float));
        auto x2 = context.create_device_buffer(
            std::uint64_t(3) * 384 * 384 * sizeof(float));
        auto patches = context.create_device_buffer(
            std::uint64_t(35) * 3 * 384 * 384 * sizeof(float));
        context.batch([&] {
            operators.bilinear_half_pixel(
                x1, x0, 1536, 1536, 768, 768, 3);
            operators.bilinear_half_pixel(
                x2, x0, 1536, 1536, 384, 384, 3);
            io.assemble_pyramid(patches, x0, x1, x2);
        });
        auto compare_resize = [&](const depth_pro_native::VulkanBuffer& gpu,
                                  std::uint32_t output_size) {
            std::vector<float> resized(
                std::uint64_t(3) * output_size * output_size);
            context.download(
                gpu, resized.data(), resized.size() * sizeof(float));
            float resize_maximum = 0.0f;
            for (std::uint32_t c = 0; c < 3u; ++c) {
                for (std::uint32_t y = 0; y < output_size; ++y) {
                    const float sy = (static_cast<float>(y) + 0.5f) *
                        1536.0f / output_size - 0.5f;
                    const auto y0 = static_cast<std::uint32_t>(sy);
                    const auto y1 = std::min(y0 + 1u, 1535u);
                    const float wy = sy - y0;
                    for (std::uint32_t x = 0; x < output_size; ++x) {
                        const float sx = (static_cast<float>(x) + 0.5f) *
                            1536.0f / output_size - 0.5f;
                        const auto x0i = static_cast<std::uint32_t>(sx);
                        const auto x1i = std::min(x0i + 1u, 1535u);
                        const float wx = sx - x0i;
                        const auto at = [&](std::uint32_t py, std::uint32_t px) {
                            return actual[(std::uint64_t(c) * 1536u + py) *
                                1536u + px];
                        };
                        const float expected =
                            (at(y0, x0i) * (1.0f - wx) + at(y0, x1i) * wx) *
                                (1.0f - wy) +
                            (at(y1, x0i) * (1.0f - wx) + at(y1, x1i) * wx) * wy;
                        const auto index =
                            (std::uint64_t(c) * output_size + y) * output_size + x;
                        resize_maximum = std::max(
                            resize_maximum,
                            std::abs(resized[static_cast<std::size_t>(index)] -
                                     expected));
                    }
                }
            }
            return resize_maximum;
        };
        const float x1_maximum = compare_resize(x1, 768u);
        const float x2_maximum = compare_resize(x2, 384u);
        std::cout << "pyramid resize maxima=" << x1_maximum << "/"
                  << x2_maximum << '\n';
        if (x1_maximum > 1.0e-4f || x2_maximum > 1.0e-4f)
            throw std::runtime_error("Depth Pro GPU pyramid resize mismatch");
        std::vector<float> x1_values(std::uint64_t(3) * 768 * 768);
        std::vector<float> x2_values(std::uint64_t(3) * 384 * 384);
        std::vector<float> patch_values(
            std::uint64_t(35) * 3 * 384 * 384);
        context.download(
            x1, x1_values.data(), x1_values.size() * sizeof(float));
        context.download(
            x2, x2_values.data(), x2_values.size() * sizeof(float));
        context.download(
            patches, patch_values.data(), patch_values.size() * sizeof(float));
        float patch_maximum = 0.0f;
        constexpr std::uint64_t patch_plane = 384u * 384u;
        for (std::uint32_t patch_index = 0; patch_index < 35u; ++patch_index) {
            const std::vector<float>* source = &actual;
            std::uint32_t source_size = 1536u;
            std::uint32_t left = (patch_index % 5u) * 288u;
            std::uint32_t top = (patch_index / 5u) * 288u;
            if (patch_index >= 25u && patch_index < 34u) {
                source = &x1_values;
                source_size = 768u;
                const auto local = patch_index - 25u;
                left = (local % 3u) * 192u;
                top = (local / 3u) * 192u;
            } else if (patch_index == 34u) {
                source = &x2_values;
                source_size = 384u;
                left = top = 0u;
            }
            for (std::uint32_t c = 0; c < 3u; ++c) {
                for (std::uint32_t y = 0; y < 384u; ++y) {
                    for (std::uint32_t x = 0; x < 384u; ++x) {
                        const auto destination =
                            std::uint64_t(patch_index) * 3u * patch_plane +
                            std::uint64_t(c) * patch_plane + y * 384u + x;
                        const auto source_index =
                            (std::uint64_t(c) * source_size + top + y) *
                            source_size + left + x;
                        patch_maximum = std::max(
                            patch_maximum,
                            std::abs(patch_values[static_cast<std::size_t>(destination)] -
                                     (*source)[static_cast<std::size_t>(source_index)]));
                    }
                }
            }
        }
        std::cout << "pyramid assembly maximum=" << patch_maximum << '\n';
        if (patch_maximum != 0.0f)
            throw std::runtime_error("Depth Pro GPU pyramid assembly mismatch");

        depth_pro_native::ModelFile internal_model(model_text);
        depth_pro_native::GpuModel internal_gpu_model(internal_model, context);
        auto zero = context.create_device_buffer(1024u * sizeof(float));
        auto forced = context.create_device_buffer(sizeof(float));
        const std::vector<float> zeros(1024u, 0.0f);
        const float forced_degrees = 63.0f;
        context.upload(zero, zeros.data(), zeros.size() * sizeof(float));
        context.upload(forced, &forced_degrees, sizeof(forced_degrees));
        auto run_graph = [&](const depth_pro_native::VulkanBuffer& graph_patches,
                             const depth_pro_native::VulkanBuffer& graph_patch) {
            auto inference = depth_pro_native::infer_gpu_device(
                context, internal_gpu_model, operators, graph_patches,
                graph_patch, zero, learned_fov ? nullptr : &forced,
                width, height);
            std::vector<float> result(std::uint64_t(width) * height);
            context.download(
                inference.depth, result.data(), result.size() * sizeof(float));
            return result;
        };
        // C: preprocessing remains device-resident from the imported texture.
        case_c = run_graph(patches, x2);
        auto run_uploaded_x0 = [&](const std::vector<float>& host_x0) {
            auto uploaded_x0 = context.create_device_buffer(
                host_x0.size() * sizeof(float));
            context.upload(
                uploaded_x0, host_x0.data(), host_x0.size() * sizeof(float));
            auto resized_x1 = context.create_device_buffer(
                std::uint64_t(3) * 768 * 768 * sizeof(float));
            auto resized_x2 = context.create_device_buffer(
                std::uint64_t(3) * 384 * 384 * sizeof(float));
            auto assembled = context.create_device_buffer(
                std::uint64_t(35) * 3 * 384 * 384 * sizeof(float));
            context.batch([&] {
                operators.bilinear_half_pixel(
                    resized_x1, uploaded_x0, 1536, 1536, 768, 768, 3);
                operators.bilinear_half_pixel(
                    resized_x2, uploaded_x0, 1536, 1536, 384, 384, 3);
                io.assemble_pyramid(
                    assembled, uploaded_x0, resized_x1, resized_x2);
            });
            return run_graph(assembled, resized_x2);
        };
        // A: exact host-built x0 upload. B: GPU x0 read back and re-uploaded.
        case_a = run_uploaded_x0(expected_x0);
        case_b = run_uploaded_x0(actual);
        normalize(case_a);
        normalize(case_b);
        normalize(case_c);
        const auto maximum_between = [](const std::vector<float>& left,
                                        const std::vector<float>& right) {
            float result = 0.0f;
            for (std::size_t index = 0; index < left.size(); ++index)
                result = std::max(
                    result, std::abs(left[index] - right[index]));
            return result;
        };
        const float a_reference = maximum_between(case_a, reference);
        const float a_b = maximum_between(case_a, case_b);
        const float b_c = maximum_between(case_b, case_c);
        std::cout << "A/reference=" << a_reference
                  << " A/B=" << a_b << " B/C=" << b_c << '\n';
        if (a_reference >= 0.01f || a_b >= 0.01f || b_c >= 0.01f)
            throw std::runtime_error("Depth Pro A/B/C graph isolation failed");
    }
    ibrh_runtime_create_request runtime_request{};
    runtime_request.struct_size = sizeof(runtime_request);
    runtime_request.api_version = IBRH_CURRENT_API_VERSION;
    runtime_request.backend = {"native", 6u};
    runtime_request.requested_device_json = {
        selected.luid_json.data(), selected.luid_json.size()};
    ibrh_runtime* runtime = nullptr;
    check(api.runtime_create(sizeof(runtime_request), &runtime_request, &runtime),
          "runtime_create");
    ibrh_model_load_request load{};
    load.struct_size = sizeof(load);
    load.api_version = IBRH_CURRENT_API_VERSION;
    load.model_path = {model_text.data(), model_text.size()};
    load.parameters_json = {parameters.data(), parameters.size()};
    ibrh_model* model = nullptr;
    check(api.model_load(runtime, sizeof(load), &load, &model), "model_load");

    depth_pro_transfer_counters before{sizeof(before), DEPTH_PRO_ABI_VERSION, 0u, 0u};
    check(depth_pro_get_transfer_counters(&before), "transfer counters before");
    std::array<Submitted, 3> retained{};
    for (std::uint32_t index = 0; index < retained.size(); ++index) {
        auto source = pixels(width, height, index);
        Capture capture = upload_texture(
            selected.device.Get(), queue.Get(), source, width, height);
        retained[index] = submit(
            api, model, capture, width, height, 1000u + index, parameters);
        if (index == 0u) {
            const double heartbeat_fps = d3d12_heartbeat_fps(
                selected.device.Get(), queue.Get());
            std::cout << "concurrentD3D12HeartbeatFps=" << heartbeat_fps << '\n';
            if (heartbeat_fps < 200.0)
                throw std::runtime_error(
                    "Depth Pro GPU work starved the D3D12 render queue");
        }
        wait_fence(selected.device.Get(), retained[index].output.ready);
        ibrh_job_status status{};
        for (std::uint32_t attempt = 0u; attempt < 1000u; ++attempt) {
            check(api.job_poll(
                retained[index].job, sizeof(status), &status), "job_poll");
            if (status.state != IBRH_JOB_RUNNING) break;
            Sleep(1);
        }
        if (status.state != IBRH_JOB_COMPLETE ||
            status.source_frame_id != 1000u + index)
            throw std::runtime_error("Depth Pro job completion correlation failed");
        api.job_release(retained[index].job);
        retained[index].job = nullptr;
    }
    if (retained[0].output.resource.native_handle ==
            retained[1].output.resource.native_handle ||
        retained[0].output.resource.native_handle ==
            retained[2].output.resource.native_handle)
        throw std::runtime_error("retained Depth Pro leases alias output slots");

    Capture dropped = upload_texture(
        selected.device.Get(), queue.Get(), reused_pixels, width, height);
    ibrh_resource dropped_resource{};
    dropped_resource.struct_size = sizeof(dropped_resource);
    dropped_resource.api_version = IBRH_CURRENT_API_VERSION;
    dropped_resource.domain = IBRH_RESOURCE_DOMAIN_D3D12;
    dropped_resource.kind = IBRH_RESOURCE_KIND_IMAGE_2D;
    dropped_resource.pixel_format = IBRH_PIXEL_BGRA8;
    dropped_resource.width = width;
    dropped_resource.height = height;
    dropped_resource.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    dropped_resource.native_handle = reinterpret_cast<std::uintptr_t>(dropped.texture_handle);
    ibrh_synchronization dropped_wait{};
    dropped_wait.struct_size = sizeof(dropped_wait);
    dropped_wait.api_version = IBRH_CURRENT_API_VERSION;
    dropped_wait.kind = IBRH_SYNC_D3D12_FENCE;
    dropped_wait.operation = IBRH_SYNC_WAIT;
    dropped_wait.native_handle_type = IBRH_NATIVE_HANDLE_WIN32_SHARED;
    dropped_wait.native_handle = reinterpret_cast<std::uintptr_t>(dropped.fence_handle);
    dropped_wait.value = dropped.value;
    ibrh_submit_request dropped_request{};
    dropped_request.struct_size = sizeof(dropped_request);
    dropped_request.api_version = IBRH_CURRENT_API_VERSION;
    dropped_request.inputs = &dropped_resource;
    dropped_request.input_count = 1u;
    dropped_request.synchronizations = &dropped_wait;
    dropped_request.synchronization_count = 1u;
    dropped_request.source_frame_id = 1003u;
    dropped_request.parameters_json = {parameters.data(), parameters.size()};
    ibrh_job* dropped_job = nullptr;
    if (api.submit(model, sizeof(dropped_request), &dropped_request, &dropped_job) !=
            IBRH_ERROR_INVALID_STATE || dropped_job != nullptr)
        throw std::runtime_error("Depth Pro fourth live lease was not rejected");
    const auto reusable_handle = retained[0].output.resource.native_handle;
    api.output_release(retained[0].lease);
    retained[0].lease = nullptr;
    Submitted reused = submit(
        api, model, dropped, width, height, 1003u, parameters);
    if (reused.output.resource.native_handle != reusable_handle)
        throw std::runtime_error("Depth Pro output slot handle was not reused");
    wait_fence(selected.device.Get(), reused.output.ready);
    std::vector<float> gpu = read_output(selected.device.Get(), queue.Get(), reused.output);
    const auto gpu_bounds = std::minmax_element(gpu.begin(), gpu.end());
    const auto reference_bounds =
        std::minmax_element(reference.begin(), reference.end());
    std::cout << "GPU raw min/max=" << *gpu_bounds.first << "/"
              << *gpu_bounds.second << " reference normalized min/max="
              << *reference_bounds.first << "/" << *reference_bounds.second
              << '\n';
    normalize(gpu);
    depth_pro_transfer_counters gpu_after{
        sizeof(gpu_after), DEPTH_PRO_ABI_VERSION, 0u, 0u};
    check(depth_pro_get_transfer_counters(&gpu_after), "GPU transfer counters after");
    if (gpu_after.tensor_upload_bytes != before.tensor_upload_bytes ||
        gpu_after.tensor_download_bytes != before.tensor_download_bytes)
        throw std::runtime_error("Depth Pro GPU path performed host tensor staging");
    float maximum_difference = 0.0f;
    for (std::size_t index = 0; index < gpu.size(); ++index)
        maximum_difference = std::max(
            maximum_difference, std::abs(gpu[index] - reference[index]));
    std::cout << "CPU correlation max/range=" << maximum_difference << '\n';
    if (maximum_difference >= 0.01f)
        throw std::runtime_error("Depth Pro GPU output exceeds the 1% CPU gate");

    Capture blocked = upload_texture(
        selected.device.Get(), queue.Get(), pixels(width, height, 9), width, height, false);
    api.output_release(retained[1].lease);
    retained[1].lease = nullptr;
    Submitted cancelled = submit(
        api, model, blocked, width, height, 2000u, parameters, false);
    // Make the imported producer wait runnable before cancelling. Cancellation
    // remains a job-state/lifetime test; leaving an external fence deliberately
    // unsignalled exercises driver-specific semaphore abandonment instead.
    check(queue->Signal(blocked.fence.Get(), blocked.value),
          "Signal(cancelled input)");
    check(api.job_cancel(cancelled.job), "job_cancel");
    ibrh_job_status cancelled_status{};
    check(api.job_poll(cancelled.job, sizeof(cancelled_status), &cancelled_status),
          "job_poll(cancelled)");
    if (cancelled_status.state != IBRH_JOB_CANCELLED)
        throw std::runtime_error("Depth Pro cancellation state failed");
    api.job_release(cancelled.job);

    api.output_release(retained[2].lease);
    api.output_release(reused.lease);
    api.job_release(reused.job);
    // One final lease must survive complete public object shutdown.
    Capture final_capture = upload_texture(
        selected.device.Get(), queue.Get(), pixels(width, height, 12), width, height);
    Submitted final_job = submit(
        api, model, final_capture, width, height, 3000u, parameters);
    api.job_release(final_job.job);
    api.model_unload(model);
    api.runtime_destroy(runtime);
    std::vector<float> final_depth = read_output(
        selected.device.Get(), queue.Get(), final_job.output);
    normalize(final_depth);
    api.output_release(final_job.lease);
    std::cout << "Depth Pro " << (learned_fov ? "learned" : "forced")
              << "-FOV common D3D12/Vulkan full graph passed; zero transfers; "
                 "three leases; reuse; cancellation; shutdown lease; "
              << "maximumSubmitReturnMs=" << maximum_submit_return_ms << '\n';
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
