#include "encoder_gpu.h"
#include "gpu_model.h"
#include "model.h"
#include "operators.h"
#include "vulkan.h"
#include "inferbridge/native_harness_precision.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void environment(const char* name, const char* value) {
#if defined(_WIN32)
    if (_putenv_s(name, value) != 0)
#else
    if (setenv(name, value, 1) != 0)
#endif
        throw std::runtime_error("could not set probe environment");
}

std::vector<float> run(const depth_pro_native::ModelFile& model,
    std::uint32_t device, std::uint32_t rows, std::uint32_t iterations,
    bool fused, const std::string& dump_prefix) {
    using namespace depth_pro_native;
    environment("DPRO_ENABLE_FC1_GELU_EPILOGUE", fused ? "1" : "0");
    environment("DPRO_TRACE_FC1_GELU_EPILOGUE", "1");
    const inferbridge::native::ScopedPrecisionRequest precision(
        inferbridge::native::Precision::fp32);
    const auto& weight = model.tensor("encoder.patch_encoder.blocks.0.mlp.fc1.weight");
    const auto& bias = model.tensor("encoder.patch_encoder.blocks.0.mlp.fc1.bias");
    if (weight.rank != 2 || weight.dimensions[0] != 4096 ||
        weight.dimensions[1] != 1024 || bias.elements != 4096)
        throw std::runtime_error("unexpected canonical FC1 shape");
    VulkanContext context(device);
    VulkanOperators operators(context);
    std::fprintf(stderr, "DPRO_FC1_PROBE mode=%s device=%s rows=%u\n",
        fused ? "candidate" : "control", context.device_name().c_str(), rows);
    std::vector<float> inputs(std::uint64_t(rows) * 1024);
    std::uint32_t random = 0x12345678u;
    for (float& value : inputs) {
        random = random * 1664525u + 1013904223u;
        value = (float((random >> 8u) & 65535u) / 32768.0f - 1.0f) * 2.0f;
    }
    std::vector<float> biases(4096);
    for (std::size_t i = 0; i < biases.size(); ++i)
        biases[i] = half_to_float(bias.data[i]);
    auto input_gpu = context.create_device_buffer(inputs.size() * sizeof(float));
    auto weight_gpu = context.create_device_buffer(weight.elements * sizeof(std::uint16_t));
    auto bias_gpu = context.create_device_buffer(biases.size() * sizeof(float));
    auto output_gpu = context.create_device_buffer(std::uint64_t(rows) * 4096 * sizeof(float));
    context.upload(input_gpu, inputs.data(), inputs.size() * sizeof(float));
    context.upload(weight_gpu, weight.data, weight.elements * sizeof(std::uint16_t));
    context.upload(bias_gpu, biases.data(), biases.size() * sizeof(float));
    std::vector<double> elapsed;
    for (std::uint32_t i = 0; i < iterations + 2u; ++i) {
        const auto begin = std::chrono::steady_clock::now();
        // Batching matches the production operator boundary. The existing
        // DPRO_VULKAN_PROFILE diagnostic deliberately splits it to timestamp
        // individual kernels; never use that mode for end-to-end A/B timing.
        context.batch([&] {
            operators.linear(output_gpu, input_gpu, weight_gpu, bias_gpu,
                rows, 1024, 4096, true, false, true, true, 8);
        });
        if (i >= 2u) elapsed.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - begin).count());
    }
    std::vector<float> output(std::uint64_t(rows) * 4096);
    context.download(output_gpu, output.data(), output.size() * sizeof(float));
    for (float value : output)
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite FC1 output");
    std::sort(elapsed.begin(), elapsed.end());
    const double median = (elapsed[(elapsed.size() - 1) / 2] +
        elapsed[elapsed.size() / 2]) * 0.5;
    std::cout << "FC1_TIMING_JSON:{\"mode\":\"" << (fused ? "candidate" : "control")
        << "\",\"rows\":" << rows << ",\"iterations\":" << iterations
        << ",\"warmup\":2,\"host_completed_batch_median_ms\":" << median << "}\n";
    if (!dump_prefix.empty()) {
        const std::string path = dump_prefix + (fused ? ".candidate.f32" : ".control.f32");
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(output.data()),
            static_cast<std::streamsize>(output.size() * sizeof(float)));
        if (!stream) throw std::runtime_error("could not write FC1 dump");
    }
    return output;
}

std::vector<float> run_encoder(const depth_pro_native::ModelFile& file,
    std::uint32_t device, std::uint32_t rows, bool fused,
    const std::string& dump_prefix) {
    using namespace depth_pro_native;
    environment("DPRO_ENABLE_FC1_GELU_EPILOGUE", fused ? "1" : "0");
    environment("DPRO_TRACE_FC1_GELU_EPILOGUE", "1");
    const inferbridge::native::ScopedPrecisionRequest precision(
        inferbridge::native::Precision::fp32);
    VulkanContext context(device);
    GpuModel model(file, context, false);
    // Match the external FP32 path: native-half weights and fixed vec8,
    // without running the older synchronous linear autotuner.
    model.set_linear_tuning(false, true, 8);
    VulkanOperators operators(context);
    const std::uint32_t batches = rows / 577u;
    std::fprintf(stderr,
        "DPRO_ENCODER_PROBE mode=%s device=%s rows=%u native_half=%u vector_tile=8\n",
        fused ? "candidate" : "control", context.device_name().c_str(), rows,
        model.uses_half_weights() ? 1u : 0u);
    std::vector<float> inputs(std::uint64_t(batches) * 3 * 384 * 384);
    std::uint32_t random = 0x12345678u;
    for (float& value : inputs) {
        random = random * 1664525u + 1013904223u;
        value = float((random >> 8u) & 65535u) / 32768.0f - 1.0f;
    }
    auto image = context.create_device_buffer(inputs.size() * sizeof(float));
    context.upload(image, inputs.data(), inputs.size() * sizeof(float));
    auto encoded = encoder_gpu(context, model, operators,
        "encoder.patch_encoder.", image, batches);
    const std::size_t tap_elements = std::size_t(rows) * 1024u;
    std::vector<float> output(tap_elements * 3u);
    const VulkanBuffer* taps[] = {&encoded.capture5, &encoded.capture11, &encoded.final};
    for (std::size_t i = 0; i < 3; ++i)
        context.download(*taps[i], output.data() + i * tap_elements,
            tap_elements * sizeof(float));
    for (float value : output)
        if (!std::isfinite(value)) throw std::runtime_error("nonfinite encoder tap");
    if (!dump_prefix.empty()) {
        const std::string path = dump_prefix +
            (fused ? ".encoder.candidate.f32" : ".encoder.control.f32");
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(output.data()),
            static_cast<std::streamsize>(output.size() * sizeof(float)));
        if (!stream) throw std::runtime_error("could not write encoder dump");
    }
    return output;
}
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "depth_pro_fc1_gelu_probe model device rows iterations control|candidate|compare|encoder-compare [dump-prefix]\n"
            "DPRO_VULKAN_PROFILE=1 enables diagnostic per-kernel GPU timestamps.\n"
            "encoder-compare requires rows divisible by 577 and iterations=1; dumps capture5, capture11, final in order.\n";
        return 0;
    }
    if (argc < 6 || argc > 7) {
        std::cerr << "Use --help for probe arguments.\n";
        return 2;
    }
    try {
        const auto device = static_cast<std::uint32_t>(std::stoul(argv[2]));
        const auto rows = static_cast<std::uint32_t>(std::stoul(argv[3]));
        const auto iterations = static_cast<std::uint32_t>(std::stoul(argv[4]));
        const std::string mode = argv[5];
        if (rows == 0 || rows > 35u * 577u || iterations == 0 || iterations > 1000 ||
            (mode != "control" && mode != "candidate" && mode != "compare" &&
             mode != "encoder-compare"))
            throw std::invalid_argument("invalid rows, iterations or mode");
        const bool encoder = mode == "encoder-compare";
        if (encoder && (rows % 577u != 0 || iterations != 1))
            throw std::invalid_argument("encoder-compare requires rows divisible by 577 and iterations=1");
        const std::string dump_prefix = argc == 7 ? argv[6] : "";
        const depth_pro_native::ModelFile model(argv[1]);
        std::vector<float> control;
        if (mode != "candidate") control = encoder
            ? run_encoder(model, device, rows, false, dump_prefix)
            : run(model, device, rows, iterations, false, dump_prefix);
        if (mode == "control") return 0;
        const auto candidate = encoder
            ? run_encoder(model, device, rows, true, dump_prefix)
            : run(model, device, rows, iterations, true, dump_prefix);
        if (mode == "candidate") return 0;
        std::uint64_t different = 0;
        double absolute_sum = 0.0, reference_sum = 0.0, maximum = 0.0;
        for (std::size_t i = 0; i < control.size(); ++i) {
            if (std::memcmp(&control[i], &candidate[i], sizeof(float)) != 0) ++different;
            const double error = std::abs(double(control[i]) - candidate[i]);
            absolute_sum += error;
            reference_sum += std::abs(double(control[i]));
            maximum = std::max(maximum, error);
        }
        std::cout << (encoder ? "ENCODER_PARITY_JSON:" : "FC1_PARITY_JSON:")
            << "{\"elements\":" << control.size()
            << ",\"bitwise_different\":" << different << ",\"maximum_absolute\":" << maximum
            << ",\"relative_l1\":" << absolute_sum / std::max(reference_sum, 1e-30) << "}\n";
        return different == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
