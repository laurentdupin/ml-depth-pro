
#include "inferbridge_harness.h"

#include "depth_pro_native.h"
#include <inferbridge/native_harness_precision.h>
#if defined(DEPTH_PRO_WITH_VULKAN) || defined(DEPTH_PRO_WITH_METAL)
#include "external_gpu.h"
#endif
#if defined(DEPTH_PRO_WITH_METAL)
#include "depth_pro_internal.h"
#endif

#if (defined(DEPTH_PRO_WITH_VULKAN) && defined(_WIN32)) || \
    (defined(DEPTH_PRO_WITH_METAL) && defined(__APPLE__))
#define DEPTH_PRO_WITH_EXTERNAL_GPU 1
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <mutex>
#include <memory>
#include <new>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

class DepthProGpuWorker;
struct DepthProGpuAdmission;

struct ibrh_runtime {
    std::string error;
    int32_t vulkan_device_index = 0;
    uint64_t adapter_luid = 0u;
    bool force_host_transfers = false;
};

struct ibrh_model {
    ibrh_runtime* runtime = nullptr;
    depth_pro_context* context = nullptr;
    std::string model_path;
#if defined(DEPTH_PRO_WITH_EXTERNAL_GPU)
    std::shared_ptr<depth_pro_native::ExternalGpu> external_gpu;
    std::shared_ptr<DepthProGpuWorker> gpu_worker;
    std::shared_ptr<std::atomic<uint32_t>> gpu_admissions =
        std::make_shared<std::atomic<uint32_t>>(0u);
#endif
    float forced_fov_degrees = 63.0f;
    std::mutex submit_mutex;
};

struct ibrh_job {
    std::atomic<uint32_t> references{1u};
#if defined(DEPTH_PRO_WITH_EXTERNAL_GPU)
    mutable std::mutex gpu_mutex;
    std::shared_ptr<depth_pro_native::ExternalJob> gpu_job;
    std::shared_ptr<DepthProGpuAdmission> gpu_admission;
    std::weak_ptr<DepthProGpuWorker> gpu_worker;
    std::atomic<uint32_t> gpu_state{IBRH_JOB_QUEUED};
    std::atomic<bool> cancel_requested{false};
    std::string gpu_error;
    std::uintptr_t input_texture_handle = 0u;
    std::uint64_t input_texture_identity = 0u;
    std::uintptr_t input_fence_handle = 0u;
    std::uint64_t input_fence_value = 0u;
    std::uintptr_t output_texture_handle = 0u;
    std::uint64_t output_texture_identity = 0u;
    std::uintptr_t output_fence_handle = 0u;
    std::uint64_t output_fence_value = 0u;
#endif
    uint64_t source_frame_id = 0u;
    uint64_t timestamp_ns = 0u;
    uint32_t width = 0u;
    uint32_t height = 0u;
    std::vector<float> depth;
    ~ibrh_job() {
#if defined(DEPTH_PRO_WITH_EXTERNAL_GPU)
        gpu_job.reset(); gpu_admission.reset();
#endif
    }
};

namespace {

thread_local std::string g_last_error;
constexpr char kHarnessId[] = "inferbridge.depth-pro.native";
constexpr char kHarnessVersion[] = "1.3.0";

ibrh_result fail(
    ibrh_runtime* runtime, ibrh_result result, const std::string& message) {
    g_last_error = message;
    if (runtime != nullptr) runtime->error = message;
    return result;
}
std::string copy_string(ibrh_string_view value) {
    return value.size == 0u ? std::string() :
        std::string(value.data, value.size);
}

bool valid_string(ibrh_string_view value) {
    return value.data != nullptr && value.size != 0u &&
        std::memchr(value.data, '\0', value.size) == nullptr;
}

bool json_string(
    const std::string& json, const std::string& key, std::string& value) {
    const std::string marker = "\"" + key + "\"";
    size_t position = json.find(marker);
    if (position == std::string::npos) return false;
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) return false;
    position = json.find_first_not_of(" \t\r\n", position + 1u);
    if (position == std::string::npos || json[position] != '"') return false;
    const size_t end = json.find('"', position + 1u);
    if (end == std::string::npos) return false;
    value = json.substr(position + 1u, end - position - 1u);
    return true;
}

bool json_float(
    const std::string& json, const std::string& key, float& value) {
    std::string text;
    if (json_string(json, key, text)) {
        char* end = nullptr;
        errno = 0;
        const float parsed = std::strtof(text.c_str(), &end);
        if (errno != 0 || end == text.c_str() || *end != '\0' ||
            !std::isfinite(parsed))
            return false;
        value = parsed;
        return true;
    }
    const std::string marker = "\"" + key + "\"";
    size_t position = json.find(marker);
    if (position == std::string::npos) return false;
    position = json.find(':', position + marker.size());
    if (position == std::string::npos) return false;
    position = json.find_first_not_of(" \t\r\n", position + 1u);
    if (position == std::string::npos) return false;
    char* end = nullptr;
    errno = 0;
    const float parsed = std::strtof(json.c_str() + position, &end);
    if (errno != 0 || end == json.c_str() + position ||
        !std::isfinite(parsed))
        return false;
    value = parsed;
    return true;
}

bool forced_fov(
    const std::string& json, float fallback, float& value) {
    value = fallback;
    std::string estimation;
    if (json_string(json, "FovEstimation", estimation)) {
        if (estimation == "YES") {
            value = 0.0f;
            return true;
        }
        if (estimation != "NO") return false;
        if (value == 0.0f) value = 63.0f;
    }
    float parsed = 0.0f;
    if (json.find("\"FovForcedValue\"") != std::string::npos) {
        if (!json_float(json, "FovForcedValue", parsed) ||
            !(parsed > 0.0f && parsed < 180.0f))
            return false;
        if (estimation != "YES") value = parsed;
    }
    return value >= 0.0f && value < 180.0f;
}

bool parse_luid(const std::string& value, uint64_t& result) {
    if (value.size() != 16u) return false;
    const auto nibble = [](char character) -> int {
        if (character >= '0' && character <= '9') return character - '0';
        if (character >= 'a' && character <= 'f') return character - 'a' + 10;
        if (character >= 'A' && character <= 'F') return character - 'A' + 10;
        return -1;
    };
    uint8_t bytes[8]{};
    for (size_t index = 0; index < 8u; ++index) {
        const int high = nibble(value[index * 2u]);
        const int low = nibble(value[index * 2u + 1u]);
        if (high < 0 || low < 0) return false;
        bytes[index] = static_cast<uint8_t>((high << 4) | low);
    }
    std::memcpy(&result, bytes, sizeof(result));
    return true;
}

bool device_index_for_luid(uint64_t luid, int32_t& device_index) {
#if defined(DEPTH_PRO_WITH_VULKAN) && defined(_WIN32)
    for (int32_t index = 0; index < 32; ++index) {
        try {
            const auto capabilities = depth_pro_native::probe_external_gpu(
                static_cast<uint32_t>(index));
            if (capabilities.available && capabilities.adapter_luid == luid) {
                device_index = index;
                return true;
            }
        } catch (...) {
            if (index == 0) return false;
            break;
        }
    }
#else
    (void)luid;
    (void)device_index;
#endif
    return false;
}

ibrh_result status_result(depth_pro_status status) {
    switch (status) {
        case DEPTH_PRO_STATUS_OK: return IBRH_OK;
        case DEPTH_PRO_STATUS_INVALID_ARGUMENT:
            return IBRH_ERROR_INVALID_ARGUMENT;
        case DEPTH_PRO_STATUS_OUT_OF_MEMORY:
        case DEPTH_PRO_STATUS_INTERNAL_ERROR:
        default:
            return IBRH_ERROR_INTERNAL;
    }
}

void retain_job(ibrh_job* job) {
    (void)job->references.fetch_add(1u);
}

void release_job(ibrh_job* job) {
    if (job != nullptr && job->references.fetch_sub(1u) == 1u) delete job;
}

} // namespace

#if defined(DEPTH_PRO_WITH_EXTERNAL_GPU)
struct DepthProGpuAdmission {
    explicit DepthProGpuAdmission(
        std::shared_ptr<std::atomic<uint32_t>> value)
        : count(std::move(value)) {}
    ~DepthProGpuAdmission() { count->fetch_sub(1u); }
    std::shared_ptr<std::atomic<uint32_t>> count;
};

class DepthProGpuWorker {
public:
    explicit DepthProGpuWorker(
        std::shared_ptr<depth_pro_native::ExternalGpu> external)
        : external_(std::move(external)), thread_([this] { run(); }) {}
    ~DepthProGpuWorker() { stop(); }

    void enqueue(ibrh_job* job) {
        retain_job(job);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_) {
                release_job(job);
                throw std::runtime_error("Depth Pro GPU worker is stopping");
            }
            queue_.push_back(job);
        }
        condition_.notify_one();
    }

    bool cancel_queued(ibrh_job* job) noexcept {
        bool removed = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto queued = std::find(queue_.begin(), queue_.end(), job);
            if (queued != queue_.end()) {
                queue_.erase(queued);
                removed = true;
            }
        }
        if (removed) {
            job->cancel_requested.store(true);
            job->gpu_state.store(IBRH_JOB_CANCELLED);
            release_job(job);
        }
        return removed;
    }

    void stop() noexcept {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        condition_.notify_all();
        if (thread_.joinable()) thread_.join();
    }

private:
    void run() noexcept {
        for (;;) {
            ibrh_job* job = nullptr;
            bool stop_requested = false;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [&] {
                    return stopping_ || !queue_.empty();
                });
                if (queue_.empty()) {
                    if (stopping_) return;
                    continue;
                }
                job = queue_.front();
                queue_.pop_front();
                stop_requested = stopping_;
            }
            if (stop_requested || job->cancel_requested.load()) {
                job->gpu_state.store(IBRH_JOB_CANCELLED);
                release_job(job);
                continue;
            }
            try {
                auto native = external_->submit_texture({
                    job->input_texture_handle,
                    job->input_texture_identity,
                    job->width,
                    job->height,
                    job->input_fence_handle,
                    job->input_fence_value,
                    job->output_texture_handle,
                    job->output_texture_identity,
                    job->width, job->height,
                    job->output_fence_handle,
                    job->output_fence_value,
                    job->source_frame_id,
                    job->timestamp_ns});
                if (job->cancel_requested.load()) native->cancel();
                {
                    std::lock_guard<std::mutex> lock(job->gpu_mutex);
                    job->gpu_job = std::move(native);
                }
                job->gpu_state.store(
                    job->cancel_requested.load() ?
                        IBRH_JOB_CANCELLED : IBRH_JOB_RUNNING);
            } catch (const std::exception& error) {
                {
                    std::lock_guard<std::mutex> lock(job->gpu_mutex);
                    job->gpu_error = error.what();
                }
                job->gpu_state.store(
                    job->cancel_requested.load() ?
                        IBRH_JOB_CANCELLED : IBRH_JOB_FAILED);
            } catch (...) {
                job->gpu_state.store(IBRH_JOB_FAILED);
            }
            release_job(job);
        }
    }

    std::shared_ptr<depth_pro_native::ExternalGpu> external_;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<ibrh_job*> queue_;
    bool stopping_ = false;
    std::thread thread_;
};
#else
struct DepthProGpuAdmission {};
#endif

namespace {

ibrh_result IBRH_CALL query_capabilities(
    size_t capabilities_size, ibrh_capabilities* capabilities) {
    if (capabilities == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (capabilities_size < sizeof(*capabilities))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    *capabilities = {};
    capabilities->struct_size = sizeof(*capabilities);
    capabilities->api_version = IBRH_CURRENT_API_VERSION;
    capabilities->flags = IBRH_CAP_HOST_MEMORY;
    capabilities->input_domain_mask =
        1ull << IBRH_RESOURCE_DOMAIN_HOST;
    capabilities->output_domain_mask =
        1ull << IBRH_RESOURCE_DOMAIN_HOST;
    capabilities->maximum_inputs = 1u;
    capabilities->maximum_outputs = 1u;
    capabilities->maximum_in_flight_jobs = 1u;
#if defined(DEPTH_PRO_WITH_VULKAN) && defined(_WIN32)
    try {
        if (depth_pro_native::probe_external_gpu(0u).available) {
            capabilities->flags |=
                IBRH_CAP_ASYNC_SUBMIT | IBRH_CAP_CANCELLATION |
                IBRH_CAP_GPU_RESOURCES | IBRH_CAP_EXTERNAL_SYNCHRONIZATION |
                IBRH_CAP_GPU_RESIDENT_OUTPUT;
            capabilities->input_domain_mask |=
                1ull << IBRH_RESOURCE_DOMAIN_D3D12;
            capabilities->output_domain_mask |=
                1ull << IBRH_RESOURCE_DOMAIN_D3D12;
            capabilities->synchronization_mask =
                1ull << IBRH_SYNC_D3D12_FENCE;
            capabilities->maximum_in_flight_jobs = 3u;
        }
    } catch (...) {
    }
#endif
#if defined(DEPTH_PRO_WITH_METAL) && defined(__APPLE__)
    capabilities->flags |=
        IBRH_CAP_ASYNC_SUBMIT | IBRH_CAP_CANCELLATION |
        IBRH_CAP_GPU_RESOURCES | IBRH_CAP_EXTERNAL_SYNCHRONIZATION |
        IBRH_CAP_GPU_RESIDENT_OUTPUT;
    capabilities->input_domain_mask |=
        1ull << IBRH_RESOURCE_DOMAIN_METAL;
    capabilities->output_domain_mask |=
        1ull << IBRH_RESOURCE_DOMAIN_METAL;
    capabilities->synchronization_mask =
        1ull << IBRH_SYNC_METAL_SHARED_EVENT;
    capabilities->maximum_in_flight_jobs = 3u;
#endif
    capabilities->harness_id = {kHarnessId, sizeof(kHarnessId) - 1u};
    capabilities->harness_version = {
        kHarnessVersion, sizeof(kHarnessVersion) - 1u};
    return IBRH_OK;
}

ibrh_result IBRH_CALL runtime_create(
    size_t request_size, const ibrh_runtime_create_request* request,
    ibrh_runtime** output) {
    if (request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    auto* runtime = new (std::nothrow) ibrh_runtime();
    if (runtime == nullptr) return IBRH_ERROR_INTERNAL;
    const std::string device = copy_string(request->requested_device_json);
    std::string transfer_mode;
    runtime->force_host_transfers =
        json_string(device, "transfer_mode", transfer_mode) &&
        transfer_mode == "host";
    uint32_t index = 0u;
    float parsed_index = 0.0f;
    if (json_float(device, "index", parsed_index)) {
        if (parsed_index < 0.0f ||
            parsed_index >
                static_cast<float>(std::numeric_limits<int32_t>::max()) ||
            std::floor(parsed_index) != parsed_index) {
            delete runtime;
            return fail(
                nullptr, IBRH_ERROR_INVALID_ARGUMENT,
                "Depth Pro requested device index is out of range");
        }
        index = static_cast<uint32_t>(parsed_index);
        if (index > static_cast<uint32_t>(std::numeric_limits<int32_t>::max())) {
            delete runtime;
            return fail(
                nullptr, IBRH_ERROR_INVALID_ARGUMENT,
                "Depth Pro requested device index is out of range");
        }
        runtime->vulkan_device_index = static_cast<int32_t>(index);
    }
    std::string luid_text;
    if (json_string(device, "luid", luid_text) && !luid_text.empty()) {
        uint64_t luid = 0u;
        if (!parse_luid(luid_text, luid) ||
            !device_index_for_luid(luid, runtime->vulkan_device_index)) {
            delete runtime;
            return fail(
                nullptr, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                "Depth Pro could not match the requested GPU LUID");
        }
        runtime->adapter_luid = luid;
    }
    *output = runtime;
    return IBRH_OK;
}

void IBRH_CALL runtime_destroy(ibrh_runtime* runtime) {
    delete runtime;
}

ibrh_result IBRH_CALL model_load(
    ibrh_runtime* runtime, size_t request_size,
    const ibrh_model_load_request* request, ibrh_model** output) {
    if (runtime == nullptr || request == nullptr || output == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    *output = nullptr;
    if (request_size < sizeof(*request) ||
        request->struct_size < sizeof(*request))
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    if (!valid_string(request->model_path))
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "Depth Pro model path is missing");
    const std::string path = copy_string(request->model_path);
    const std::string parameters = copy_string(request->parameters_json);
    inferbridge::native::ScopedPrecisionRequest precision_request(
        inferbridge::native::precision_from_parameters_json(parameters));
    auto* model = new (std::nothrow) ibrh_model();
    if (model == nullptr) return IBRH_ERROR_INTERNAL;
    model->runtime = runtime;
    model->model_path = path;
    if (!forced_fov(
            parameters, model->forced_fov_degrees,
            model->forced_fov_degrees)) {
        delete model;
        return fail(
            runtime, IBRH_ERROR_INVALID_ARGUMENT,
            "Depth Pro FOV parameters are invalid");
    }
#if defined(DEPTH_PRO_WITH_VULKAN) && defined(_WIN32)
    if (runtime->adapter_luid != 0u && !runtime->force_host_transfers) {
        try {
            model->external_gpu = depth_pro_native::create_external_gpu(
                path, model->forced_fov_degrees,
                static_cast<uint32_t>(runtime->vulkan_device_index));
            const auto capabilities = model->external_gpu->capabilities();
            if (!capabilities.available ||
                capabilities.adapter_luid != runtime->adapter_luid)
                throw std::runtime_error(
                    "Depth Pro loaded on a GPU other than the requested LUID");
            model->gpu_worker = std::make_shared<DepthProGpuWorker>(
                model->external_gpu);
        } catch (const std::exception& error) {
            delete model;
            return fail(runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY, error.what());
        }
    } else
#endif
    {
        const depth_pro_status status = depth_pro_create_vulkan(
            path.c_str(),
            static_cast<uint32_t>(runtime->vulkan_device_index),
            &model->context);
        if (status != DEPTH_PRO_STATUS_OK) {
            const std::string message =
                std::string("Depth Pro model load failed: ") +
                depth_pro_last_error();
            delete model;
            return fail(runtime, status_result(status), message);
        }
#if defined(DEPTH_PRO_WITH_METAL) && defined(__APPLE__)
        if (!runtime->force_host_transfers) {
            try {
                model->external_gpu = depth_pro_native::create_metal_external_gpu(
                    model->context, model->forced_fov_degrees);
                model->gpu_worker = std::make_shared<DepthProGpuWorker>(
                    model->external_gpu);
            } catch (const std::exception& error) {
                depth_pro_destroy(model->context);
                delete model;
                return fail(runtime, IBRH_ERROR_UNSUPPORTED_CAPABILITY,
                    error.what());
            }
        }
#endif
    }
    *output = model;
    return IBRH_OK;
}

void IBRH_CALL model_unload(ibrh_model* model) {
    if (model == nullptr) return;
#if defined(DEPTH_PRO_WITH_EXTERNAL_GPU)
    if (model->gpu_worker) model->gpu_worker->stop();
    model->gpu_worker.reset();
    model->external_gpu.reset();
#endif
    depth_pro_destroy(model->context);
    delete model;
}

ibrh_result IBRH_CALL model_describe_io(const ibrh_model*m,size_t n,ibrh_model_io_descriptor*o){if(!m||!o)return IBRH_ERROR_INVALID_ARGUMENT;if(n<sizeof(*o))return IBRH_ERROR_STRUCT_TOO_SMALL;*o={};o->struct_size=sizeof(*o);o->api_version=IBRH_CURRENT_API_VERSION;o->input_count=o->output_count=1;return IBRH_OK;}
ibrh_result IBRH_CALL model_get_port(const ibrh_model*m,uint32_t d,uint32_t i,size_t n,ibrh_port_descriptor*o){if(!m||!o)return IBRH_ERROR_INVALID_ARGUMENT;if(n<sizeof(*o))return IBRH_ERROR_STRUCT_TOO_SMALL;if(i||(d!=IBRH_PORT_INPUT&&d!=IBRH_PORT_OUTPUT))return IBRH_ERROR_NOT_FOUND;*o={};o->struct_size=sizeof(*o);o->api_version=IBRH_CURRENT_API_VERSION;o->direction=d;o->semantic=d==IBRH_PORT_INPUT?IBRH_SEMANTIC_IMAGE:IBRH_SEMANTIC_DEPTH;o->payload_type=d==IBRH_PORT_INPUT?IBRH_PIXEL_BGRA8:IBRH_PIXEL_DEPTH_METRIC_FLOAT32;o->pixel_format=o->payload_type;o->accepted_pixel_format_mask=1ull<<o->pixel_format;o->resource_kind=IBRH_RESOURCE_KIND_IMAGE_2D;o->depth=1;o->flags=IBRH_DESCRIPTOR_DYNAMIC_WIDTH|IBRH_DESCRIPTOR_DYNAMIC_HEIGHT;return IBRH_OK;}
ibrh_result IBRH_CALL model_plan_outputs(const ibrh_model*m,size_t n,const ibrh_output_plan_request*r,uint32_t c,ibrh_port_descriptor*o){if(!m||!r||!o)return IBRH_ERROR_INVALID_ARGUMENT;if(n<sizeof(*r)||r->struct_size<sizeof(*r)||c<1)return IBRH_ERROR_STRUCT_TOO_SMALL;if(r->input_count!=1||!r->inputs)return IBRH_ERROR_INVALID_ARGUMENT;auto x=model_get_port(m,IBRH_PORT_OUTPUT,0,sizeof(o[0]),&o[0]);if(x!=IBRH_OK)return x;o[0].width=r->inputs[0].width;o[0].height=r->inputs[0].height;o[0].flags=0;return IBRH_OK;}

ibrh_result IBRH_CALL submit(ibrh_model*model,size_t n,const ibrh_submit_request*r,ibrh_job**out){if(!model||!r||!out)return IBRH_ERROR_INVALID_ARGUMENT;*out=nullptr;if(n<sizeof(*r)||r->struct_size<sizeof(*r))return IBRH_ERROR_STRUCT_TOO_SMALL;if(r->input_count!=1||!r->inputs||r->output_count!=1||!r->outputs)return IBRH_ERROR_INVALID_ARGUMENT;const auto&s=r->inputs[0];const auto&t=r->outputs[0];const auto&i=s.resource;const auto&o=t.resource;float fov=model->forced_fov_degrees;if(!forced_fov(copy_string(r->parameters_json),fov,fov))return IBRH_ERROR_INVALID_ARGUMENT;if(!i.width||!i.height||o.width!=i.width||o.height!=i.height||o.pixel_format!=IBRH_PIXEL_DEPTH_METRIC_FLOAT32)return IBRH_ERROR_INVALID_ARGUMENT;
#if defined(DEPTH_PRO_WITH_VULKAN) && defined(_WIN32)
if(i.domain==IBRH_RESOURCE_DOMAIN_D3D12){if(fov!=model->forced_fov_degrees||o.domain!=IBRH_RESOURCE_DOMAIN_D3D12||i.pixel_format!=IBRH_PIXEL_BGRA8||i.native_handle_type!=IBRH_NATIVE_HANDLE_WIN32_SHARED||o.native_handle_type!=IBRH_NATIVE_HANDLE_WIN32_SHARED||s.synchronization.kind!=IBRH_SYNC_D3D12_FENCE||s.synchronization.operation!=IBRH_SYNC_WAIT||t.synchronization.kind!=IBRH_SYNC_D3D12_FENCE||t.synchronization.operation!=IBRH_SYNC_SIGNAL)return IBRH_ERROR_UNSUPPORTED_CAPABILITY;uint32_t admitted=model->gpu_admissions->load();while(admitted<3&&!model->gpu_admissions->compare_exchange_weak(admitted,admitted+1)){}if(admitted>=3)return IBRH_ERROR_INVALID_STATE;auto*j=new(std::nothrow)ibrh_job();if(!j){model->gpu_admissions->fetch_sub(1);return IBRH_ERROR_INTERNAL;}try{j->gpu_admission=std::make_shared<DepthProGpuAdmission>(model->gpu_admissions);}catch(...){model->gpu_admissions->fetch_sub(1);delete j;return IBRH_ERROR_INTERNAL;}j->input_texture_handle=i.native_handle;j->input_texture_identity=i.auxiliary_handle;j->input_fence_handle=s.synchronization.native_handle;j->input_fence_value=s.synchronization.value;j->output_texture_handle=o.native_handle;j->output_texture_identity=o.auxiliary_handle;j->output_fence_handle=t.synchronization.native_handle;j->output_fence_value=t.synchronization.value;j->source_frame_id=r->source_frame_id;j->timestamp_ns=r->timestamp_ns;j->width=i.width;j->height=i.height;try{std::lock_guard<std::mutex>l(model->submit_mutex);if(!model->external_gpu){depth_pro_destroy(model->context);model->context=nullptr;model->external_gpu=depth_pro_native::create_external_gpu(model->model_path,model->forced_fov_degrees,model->runtime->vulkan_device_index);model->gpu_worker=std::make_shared<DepthProGpuWorker>(model->external_gpu);}j->gpu_worker=model->gpu_worker;model->gpu_worker->enqueue(j);}catch(const std::exception&e){delete j;return fail(model->runtime,IBRH_ERROR_UNSUPPORTED_CAPABILITY,e.what());}*out=j;return IBRH_OK;}
#endif
#if defined(DEPTH_PRO_WITH_METAL) && defined(__APPLE__)
if(i.domain==IBRH_RESOURCE_DOMAIN_METAL){const auto&wait=s.synchronization;const auto&signal=t.synchronization;const bool no_wait=wait.kind==IBRH_SYNC_NONE;const bool event_wait=wait.kind==IBRH_SYNC_METAL_SHARED_EVENT&&wait.operation==IBRH_SYNC_WAIT&&wait.native_handle_type==IBRH_NATIVE_HANDLE_METAL_SHARED_EVENT&&wait.native_handle!=0u;if(fov!=model->forced_fov_degrees||!model->external_gpu||o.domain!=IBRH_RESOURCE_DOMAIN_METAL||i.pixel_format!=IBRH_PIXEL_BGRA8||i.native_handle_type!=IBRH_NATIVE_HANDLE_METAL_TEXTURE||!i.native_handle||o.native_handle_type!=IBRH_NATIVE_HANDLE_METAL_TEXTURE||!o.native_handle||(!no_wait&&!event_wait)||signal.kind!=IBRH_SYNC_METAL_SHARED_EVENT||signal.operation!=IBRH_SYNC_SIGNAL||signal.native_handle_type!=IBRH_NATIVE_HANDLE_METAL_SHARED_EVENT||!signal.native_handle||!signal.value)return IBRH_ERROR_UNSUPPORTED_CAPABILITY;uint32_t admitted=model->gpu_admissions->load();while(admitted<3&&!model->gpu_admissions->compare_exchange_weak(admitted,admitted+1)){}if(admitted>=3)return IBRH_ERROR_INVALID_STATE;auto*j=new(std::nothrow)ibrh_job();if(!j){model->gpu_admissions->fetch_sub(1);return IBRH_ERROR_INTERNAL;}try{j->gpu_admission=std::make_shared<DepthProGpuAdmission>(model->gpu_admissions);}catch(...){model->gpu_admissions->fetch_sub(1);delete j;return IBRH_ERROR_INTERNAL;}j->input_texture_handle=i.native_handle;j->input_texture_identity=i.auxiliary_handle;j->input_fence_handle=event_wait?wait.native_handle:0u;j->input_fence_value=event_wait?wait.value:0u;j->output_texture_handle=o.native_handle;j->output_texture_identity=o.auxiliary_handle;j->output_fence_handle=signal.native_handle;j->output_fence_value=signal.value;j->source_frame_id=r->source_frame_id;j->timestamp_ns=r->timestamp_ns;j->width=i.width;j->height=i.height;try{j->gpu_worker=model->gpu_worker;model->gpu_worker->enqueue(j);}catch(const std::exception&e){delete j;return fail(model->runtime,IBRH_ERROR_UNSUPPORTED_CAPABILITY,e.what());}*out=j;return IBRH_OK;}
#endif
if(i.domain!=IBRH_RESOURCE_DOMAIN_HOST||o.domain!=IBRH_RESOURCE_DOMAIN_HOST||i.pixel_format!=IBRH_PIXEL_BGRA8||i.native_handle_type!=IBRH_NATIVE_HANDLE_HOST_POINTER||o.native_handle_type!=IBRH_NATIVE_HANDLE_HOST_POINTER||s.synchronization.kind!=IBRH_SYNC_NONE||t.synchronization.kind!=IBRH_SYNC_NONE)return IBRH_ERROR_UNSUPPORTED_CAPABILITY;const auto*bgra=reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(i.native_handle))+i.byte_offset;auto*depth=reinterpret_cast<float*>(static_cast<uintptr_t>(o.native_handle)+o.byte_offset);float focal=0;depth_pro_status q;{std::lock_guard<std::mutex>l(model->submit_mutex);q=depth_pro_infer_bgra8_f32(model->context,bgra,i.row_stride_bytes,i.width,i.height,fov,depth,static_cast<size_t>(i.width)*i.height,&focal);}if(q!=DEPTH_PRO_STATUS_OK)return fail(model->runtime,status_result(q),depth_pro_last_error());auto*j=new(std::nothrow)ibrh_job();if(!j)return IBRH_ERROR_INTERNAL;j->source_frame_id=r->source_frame_id;j->timestamp_ns=r->timestamp_ns;j->width=i.width;j->height=i.height;*out=j;return IBRH_OK;}
ibrh_result IBRH_CALL job_poll(
    const ibrh_job* job, size_t status_size, ibrh_job_status* status) {
    if (job == nullptr || status == nullptr)
        return IBRH_ERROR_INVALID_ARGUMENT;
    if (status_size < sizeof(*status)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    *status = {};
    status->struct_size = sizeof(*status);
#if defined(DEPTH_PRO_WITH_EXTERNAL_GPU)
    if (job->gpu_admission) {
        std::shared_ptr<depth_pro_native::ExternalJob> gpu_job;
        {
            std::lock_guard<std::mutex> lock(job->gpu_mutex);
            gpu_job = job->gpu_job;
        }
        if (gpu_job) {
            switch (gpu_job->state()) {
                case depth_pro_native::ExternalJobState::running:
                    status->state = IBRH_JOB_RUNNING; break;
                case depth_pro_native::ExternalJobState::complete:
                    status->state = IBRH_JOB_COMPLETE; break;
                case depth_pro_native::ExternalJobState::cancelled:
                    status->state = IBRH_JOB_CANCELLED; break;
            }
        } else {
            status->state = job->gpu_state.load();
        }
    } else {
        status->state = IBRH_JOB_COMPLETE;
    }
#else
    status->state = IBRH_JOB_COMPLETE;
#endif
    status->output_count = 1u;
    status->source_frame_id = job->source_frame_id;
    return IBRH_OK;
}

ibrh_result IBRH_CALL job_cancel(ibrh_job* job) {
    if (job == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
#if defined(DEPTH_PRO_WITH_EXTERNAL_GPU)
    job->cancel_requested.store(true);
    if (auto worker = job->gpu_worker.lock();
        worker && worker->cancel_queued(job))
        return IBRH_OK;
    std::shared_ptr<depth_pro_native::ExternalJob> gpu_job;
    {
        std::lock_guard<std::mutex> lock(job->gpu_mutex);
        gpu_job = job->gpu_job;
    }
    if (gpu_job) {
        gpu_job->cancel();
        job->gpu_state.store(IBRH_JOB_CANCELLED);
        return IBRH_OK;
    }
    const uint32_t state = job->gpu_state.load();
    if (state == IBRH_JOB_QUEUED || state == IBRH_JOB_RUNNING) {
        job->gpu_state.store(IBRH_JOB_CANCELLED);
        return IBRH_OK;
    }
#endif
    return IBRH_ERROR_INVALID_STATE;
}

void IBRH_CALL job_release(ibrh_job* job) {
    release_job(job);
}

ibrh_result IBRH_CALL get_last_error(
    const void* object, char* destination, size_t destination_size,
    size_t* required_size) {
    const auto* runtime = static_cast<const ibrh_runtime*>(object);
    const std::string& message =
        runtime != nullptr && !runtime->error.empty() ?
        runtime->error : g_last_error;
    const size_t required = message.size() + 1u;
    if (required_size != nullptr) *required_size = required;
    if (destination == nullptr || destination_size < required)
        return IBRH_ERROR_STRUCT_TOO_SMALL;
    std::memcpy(destination, message.c_str(), required);
    return IBRH_OK;
}

}  // namespace

extern "C" IBRH_API ibrh_result IBRH_CALL ibrh_get_api(
    uint32_t requested_api_version, size_t api_size, ibrh_api* api) {
    if (api == nullptr) return IBRH_ERROR_INVALID_ARGUMENT;
    if (api_size < sizeof(*api)) return IBRH_ERROR_STRUCT_TOO_SMALL;
    if ((requested_api_version >> 16u) != IBRH_API_VERSION_MAJOR)
        return IBRH_ERROR_UNSUPPORTED_API;
    *api = {};
    api->struct_size = sizeof(*api);
    api->api_version = IBRH_CURRENT_API_VERSION;
    api->query_capabilities = query_capabilities;
    api->runtime_create = runtime_create;
    api->runtime_destroy = runtime_destroy;
    api->model_load = model_load;
    api->model_unload=model_unload;api->model_describe_io=model_describe_io;api->model_get_port=model_get_port;api->model_plan_outputs=model_plan_outputs;api->submit=submit;
    api->job_poll = job_poll;
    api->job_cancel = job_cancel;
    api->job_release = job_release;
    api->get_last_error = get_last_error;
    return IBRH_OK;
}
