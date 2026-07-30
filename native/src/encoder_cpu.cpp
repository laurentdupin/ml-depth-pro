#include "encoder_cpu.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace depth_pro_native {
namespace {

constexpr std::uint32_t embedding = 1024;
constexpr std::uint32_t heads = 16;
constexpr std::uint32_t head_channels = 64;

template <typename Function>
void parallel_for(std::uint32_t count, Function function) {
    if (count == 0) {
        return;
    }
    const std::uint32_t workers = std::min(
        count, std::max(
            1u, std::min(16u, std::thread::hardware_concurrency())));
    if (workers == 1) {
        function(0, count);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (std::uint32_t worker = 0; worker < workers; ++worker) {
        const std::uint32_t begin =
            static_cast<std::uint32_t>(
                std::uint64_t(count) * worker / workers);
        const std::uint32_t end =
            static_cast<std::uint32_t>(
                std::uint64_t(count) * (worker + 1) / workers);
        threads.emplace_back(function, begin, end);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
}

const TensorView& tensor(
    const ModelFile& model,
    const std::string& name,
    std::uint32_t rank) {
    const TensorView& result = model.tensor(name);
    if (result.rank != rank) {
        throw std::runtime_error(
            "unexpected Depth Pro encoder tensor rank: " + name);
    }
    return result;
}

std::vector<float> decode(const TensorView& tensor) {
    std::vector<float> output(
        static_cast<std::size_t>(tensor.elements));
    parallel_for(
        static_cast<std::uint32_t>(output.size()),
        [&](std::uint32_t begin, std::uint32_t end) {
            for (std::uint32_t index = begin; index < end; ++index) {
                output[index] = half_to_float(tensor.data[index]);
            }
        });
    return output;
}

void linear(
    const std::vector<float>& input,
    std::uint32_t rows,
    std::uint32_t input_channels,
    const TensorView& weight_view,
    const TensorView& bias_view,
    std::vector<float>& output) {
    if (weight_view.rank != 2 || bias_view.rank != 1 ||
        weight_view.dimensions[1] != input_channels ||
        bias_view.dimensions[0] != weight_view.dimensions[0] ||
        input.size() != std::uint64_t(rows) * input_channels) {
        throw std::runtime_error(
            "Depth Pro encoder linear shape mismatch");
    }
    const std::uint32_t output_channels =
        static_cast<std::uint32_t>(weight_view.dimensions[0]);
    const std::vector<float> weight = decode(weight_view);
    const std::vector<float> bias = decode(bias_view);
    output.resize(std::uint64_t(rows) * output_channels);
    parallel_for(rows, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t row = begin; row < end; ++row) {
            const float* source =
                input.data() + std::uint64_t(row) * input_channels;
            float* destination =
                output.data() + std::uint64_t(row) * output_channels;
            for (std::uint32_t out = 0;
                 out < output_channels; ++out) {
                const float* kernel =
                    weight.data() +
                    std::uint64_t(out) * input_channels;
                float value = bias[out];
                for (std::uint32_t in = 0;
                     in < input_channels; ++in) {
                    value += source[in] * kernel[in];
                }
                destination[out] = value;
            }
        }
    });
}

void layer_norm(
    const std::vector<float>& input,
    std::uint32_t rows,
    const TensorView& scale_view,
    const TensorView& bias_view,
    std::vector<float>& output) {
    if (input.size() != std::uint64_t(rows) * embedding ||
        scale_view.rank != 1 || bias_view.rank != 1 ||
        scale_view.dimensions[0] != embedding ||
        bias_view.dimensions[0] != embedding) {
        throw std::runtime_error(
            "Depth Pro layer norm shape mismatch");
    }
    const std::vector<float> scale = decode(scale_view);
    const std::vector<float> bias = decode(bias_view);
    output.resize(input.size());
    parallel_for(rows, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t row = begin; row < end; ++row) {
            const float* source =
                input.data() + std::uint64_t(row) * embedding;
            float* destination =
                output.data() + std::uint64_t(row) * embedding;
            float mean = 0.0f;
            for (std::uint32_t channel = 0;
                 channel < embedding; ++channel) {
                mean += source[channel];
            }
            mean /= static_cast<float>(embedding);
            float variance = 0.0f;
            for (std::uint32_t channel = 0;
                 channel < embedding; ++channel) {
                const float difference = source[channel] - mean;
                variance += difference * difference;
            }
            variance /= static_cast<float>(embedding);
            const float inverse =
                1.0f / std::sqrt(variance + 1.0e-6f);
            for (std::uint32_t channel = 0;
                 channel < embedding; ++channel) {
                destination[channel] =
                    (source[channel] - mean) * inverse *
                        scale[channel] +
                    bias[channel];
            }
        }
    });
}

void attention(
    const ModelFile& model,
    const std::string& base,
    const std::vector<float>& input,
    std::uint32_t tokens,
    std::vector<float>& output) {
    std::vector<float> qkv;
    linear(
        input, tokens, embedding,
        tensor(model, base + "qkv.weight", 2),
        tensor(model, base + "qkv.bias", 1), qkv);
    std::vector<float> attended(
        std::uint64_t(tokens) * embedding);
    parallel_for(
        heads * tokens,
        [&](std::uint32_t begin, std::uint32_t end) {
            std::vector<float> scores(tokens);
            for (std::uint32_t task = begin; task < end; ++task) {
                const std::uint32_t head = task / tokens;
                const std::uint32_t query = task % tokens;
                const float* q = qkv.data() +
                    std::uint64_t(query) * 3 * embedding +
                    head * head_channels;
                float maximum =
                    -std::numeric_limits<float>::infinity();
                for (std::uint32_t key = 0; key < tokens; ++key) {
                    const float* k = qkv.data() +
                        std::uint64_t(key) * 3 * embedding +
                        embedding + head * head_channels;
                    float score = 0.0f;
                    for (std::uint32_t channel = 0;
                         channel < head_channels; ++channel) {
                        score +=
                            (q[channel] * 0.125f) * k[channel];
                    }
                    scores[key] = score;
                    maximum = std::max(maximum, score);
                }
                float denominator = 0.0f;
                for (float& score : scores) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                float* destination = attended.data() +
                    std::uint64_t(query) * embedding +
                    head * head_channels;
                std::fill_n(destination, head_channels, 0.0f);
                for (std::uint32_t key = 0; key < tokens; ++key) {
                    const float probability =
                        scores[key] / denominator;
                    const float* value = qkv.data() +
                        std::uint64_t(key) * 3 * embedding +
                        2 * embedding + head * head_channels;
                    for (std::uint32_t channel = 0;
                         channel < head_channels; ++channel) {
                        destination[channel] +=
                            probability * value[channel];
                    }
                }
            }
        });
    linear(
        attended, tokens, embedding,
        tensor(model, base + "proj.weight", 2),
        tensor(model, base + "proj.bias", 1), output);
}

void block(
    const ModelFile& model,
    const std::string& prefix,
    std::uint32_t block_index,
    std::uint32_t tokens,
    std::vector<float>& state) {
    const std::string base =
        prefix + "blocks." + std::to_string(block_index) + ".";
    std::vector<float> normalized;
    layer_norm(
        state, tokens,
        tensor(model, base + "norm1.weight", 1),
        tensor(model, base + "norm1.bias", 1), normalized);
    std::vector<float> branch;
    attention(
        model, base + "attn.", normalized, tokens, branch);
    const std::vector<float> gamma1 =
        decode(tensor(model, base + "ls1.gamma", 1));
    parallel_for(tokens, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t row = begin; row < end; ++row) {
            for (std::uint32_t channel = 0;
                 channel < embedding; ++channel) {
                state[std::uint64_t(row) * embedding + channel] +=
                    branch[
                        std::uint64_t(row) * embedding + channel] *
                    gamma1[channel];
            }
        }
    });
    layer_norm(
        state, tokens,
        tensor(model, base + "norm2.weight", 1),
        tensor(model, base + "norm2.bias", 1), normalized);
    linear(
        normalized, tokens, embedding,
        tensor(model, base + "mlp.fc1.weight", 2),
        tensor(model, base + "mlp.fc1.bias", 1), branch);
    parallel_for(
        static_cast<std::uint32_t>(branch.size()),
        [&](std::uint32_t begin, std::uint32_t end) {
            for (std::uint32_t index = begin; index < end; ++index) {
                const float value = branch[index];
                branch[index] = 0.5f * value *
                    (1.0f + std::erf(
                        value * 0.7071067811865475244f));
            }
        });
    std::vector<float> projected;
    linear(
        branch, tokens, 4096,
        tensor(model, base + "mlp.fc2.weight", 2),
        tensor(model, base + "mlp.fc2.bias", 1), projected);
    const std::vector<float> gamma2 =
        decode(tensor(model, base + "ls2.gamma", 1));
    parallel_for(tokens, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t row = begin; row < end; ++row) {
            for (std::uint32_t channel = 0;
                 channel < embedding; ++channel) {
                state[std::uint64_t(row) * embedding + channel] +=
                    projected[
                        std::uint64_t(row) * embedding + channel] *
                    gamma2[channel];
            }
        }
    });
}

}  // namespace

EncoderOutput encoder_cpu(
    const ModelFile& model,
    std::string_view encoder_prefix,
    const float* input,
    std::uint32_t width,
    std::uint32_t height) {
    if (!input || width != 384 || height != 384) {
        throw std::invalid_argument(
            "Depth Pro encoder currently requires 384x384 input");
    }
    const std::string prefix(encoder_prefix);
    const std::uint32_t patch_width = width / 16;
    const std::uint32_t patch_height = height / 16;
    const std::uint32_t patches =
        patch_width * patch_height;
    const std::uint32_t tokens = 1 + patches;
    const TensorView& weight_view = tensor(
        model, prefix + "patch_embed.proj.weight", 4);
    const TensorView& bias_view = tensor(
        model, prefix + "patch_embed.proj.bias", 1);
    if (weight_view.dimensions[0] != embedding ||
        weight_view.dimensions[1] != 3 ||
        weight_view.dimensions[2] != 16 ||
        weight_view.dimensions[3] != 16 ||
        bias_view.dimensions[0] != embedding) {
        throw std::runtime_error(
            "unexpected Depth Pro patch embedding shape");
    }
    const std::vector<float> weight = decode(weight_view);
    const std::vector<float> bias = decode(bias_view);
    std::vector<float> state(
        std::uint64_t(tokens) * embedding);
    const std::vector<float> cls =
        decode(tensor(model, prefix + "cls_token", 3));
    const std::vector<float> position =
        decode(tensor(model, prefix + "pos_embed", 3));
    std::copy_n(cls.data(), embedding, state.data());
    parallel_for(
        patches * embedding,
        [&](std::uint32_t begin, std::uint32_t end) {
            for (std::uint32_t task = begin; task < end; ++task) {
                const std::uint32_t patch = task / embedding;
                const std::uint32_t out = task % embedding;
                const std::uint32_t py = patch / patch_width;
                const std::uint32_t px = patch % patch_width;
                const float* kernel =
                    weight.data() +
                    std::uint64_t(out) * 3 * 16 * 16;
                float value = bias[out];
                for (std::uint32_t channel = 0;
                     channel < 3; ++channel) {
                    for (std::uint32_t ky = 0; ky < 16; ++ky) {
                        const float* source = input +
                            std::uint64_t(channel) * width * height +
                            std::uint64_t(py * 16 + ky) * width +
                            px * 16;
                        const float* row = kernel +
                            std::uint64_t(channel) * 16 * 16 +
                            ky * 16;
                        for (std::uint32_t kx = 0; kx < 16; ++kx) {
                            value += source[kx] * row[kx];
                        }
                    }
                }
                state[
                    std::uint64_t(1 + patch) * embedding + out] =
                    value;
            }
        });
    parallel_for(
        static_cast<std::uint32_t>(state.size()),
        [&](std::uint32_t begin, std::uint32_t end) {
            for (std::uint32_t index = begin; index < end; ++index) {
                state[index] += position[index];
            }
        });

    EncoderOutput output;
    output.patch_width = patch_width;
    output.patch_height = patch_height;
    for (std::uint32_t block_index = 0;
         block_index < 24; ++block_index) {
        block(
            model, prefix, block_index, tokens, state);
        if (block_index == 5 || block_index == 11 ||
            block_index == 17 || block_index == 23) {
            output.captures.push_back(state);
        }
    }
    return output;
}

}  // namespace depth_pro_native

