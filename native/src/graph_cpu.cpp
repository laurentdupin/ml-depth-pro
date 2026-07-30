#include "graph_cpu.h"

#include "encoder_cpu.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace depth_pro_native {
namespace {

struct Image {
    std::uint32_t channels = 0;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
    std::vector<float> values;
};

std::uint64_t count(const Image& image) {
    return std::uint64_t(image.channels) *
        image.height * image.width;
}

template <typename Function>
void parallel_for(std::uint32_t tasks, Function function) {
    const std::uint32_t workers = std::min(
        tasks, std::max(
            1u, std::min(16u, std::thread::hardware_concurrency())));
    std::vector<std::thread> threads;
    for (std::uint32_t worker = 0; worker < workers; ++worker) {
        const std::uint32_t begin =
            static_cast<std::uint32_t>(
                std::uint64_t(tasks) * worker / workers);
        const std::uint32_t end =
            static_cast<std::uint32_t>(
                std::uint64_t(tasks) * (worker + 1) / workers);
        threads.emplace_back(function, begin, end);
    }
    for (std::thread& thread : threads) {
        thread.join();
    }
}

std::vector<float> decode(const TensorView& tensor) {
    std::vector<float> output(
        static_cast<std::size_t>(tensor.elements));
    parallel_for(
        static_cast<std::uint32_t>(output.size()),
        [&](std::uint32_t begin, std::uint32_t end) {
            for (std::uint32_t i = begin; i < end; ++i) {
                output[i] = half_to_float(tensor.data[i]);
            }
        });
    return output;
}

Image resize(
    const Image& input,
    std::uint32_t output_height,
    std::uint32_t output_width) {
    Image output{
        input.channels, output_height, output_width, {}};
    output.values.resize(static_cast<std::size_t>(count(output)));
    parallel_for(input.channels, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t c = begin; c < end; ++c) {
            for (std::uint32_t oy = 0; oy < output_height; ++oy) {
                const float sy_raw =
                    (static_cast<float>(oy) + 0.5f) *
                        input.height / output_height -
                    0.5f;
                const float sy = std::clamp(
                    sy_raw, 0.0f,
                    static_cast<float>(input.height - 1));
                const std::uint32_t y0 =
                    static_cast<std::uint32_t>(sy);
                const std::uint32_t y1 =
                    std::min(y0 + 1, input.height - 1);
                const float wy = sy - y0;
                for (std::uint32_t ox = 0; ox < output_width; ++ox) {
                    const float sx_raw =
                        (static_cast<float>(ox) + 0.5f) *
                            input.width / output_width -
                        0.5f;
                    const float sx = std::clamp(
                        sx_raw, 0.0f,
                        static_cast<float>(input.width - 1));
                    const std::uint32_t x0 =
                        static_cast<std::uint32_t>(sx);
                    const std::uint32_t x1 =
                        std::min(x0 + 1, input.width - 1);
                    const float wx = sx - x0;
                    const auto at = [&](std::uint32_t y, std::uint32_t x) {
                        return input.values[
                            (std::uint64_t(c) * input.height + y) *
                                input.width +
                            x];
                    };
                    output.values[
                        (std::uint64_t(c) * output_height + oy) *
                            output_width +
                        ox] =
                        (at(y0, x0) * (1.0f - wx) +
                         at(y0, x1) * wx) *
                            (1.0f - wy) +
                        (at(y1, x0) * (1.0f - wx) +
                         at(y1, x1) * wx) *
                            wy;
                }
            }
        }
    });
    return output;
}

Image crop(
    const Image& input,
    std::uint32_t top,
    std::uint32_t left,
    std::uint32_t size) {
    Image output{input.channels, size, size, {}};
    output.values.resize(static_cast<std::size_t>(count(output)));
    parallel_for(input.channels, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t c = begin; c < end; ++c) {
            for (std::uint32_t y = 0; y < size; ++y) {
                const float* source = input.values.data() +
                    (std::uint64_t(c) * input.height + top + y) *
                        input.width +
                    left;
                float* destination = output.values.data() +
                    (std::uint64_t(c) * size + y) * size;
                std::copy_n(source, size, destination);
            }
        }
    });
    return output;
}

Image conv(
    const ModelFile& model,
    const Image& input,
    const std::string& weight_name,
    const std::string& bias_name,
    std::uint32_t stride,
    std::uint32_t padding) {
    const TensorView& weight_view = model.tensor(weight_name);
    if (weight_view.rank != 4 ||
        weight_view.dimensions[1] != input.channels ||
        weight_view.dimensions[2] != weight_view.dimensions[3]) {
        throw std::runtime_error(
            "Depth Pro convolution shape mismatch: " + weight_name);
    }
    const std::vector<float> weight = decode(weight_view);
    std::vector<float> bias;
    if (!bias_name.empty()) {
        bias = decode(model.tensor(bias_name));
    }
    const std::uint32_t out_channels =
        static_cast<std::uint32_t>(weight_view.dimensions[0]);
    const std::uint32_t kernel =
        static_cast<std::uint32_t>(weight_view.dimensions[2]);
    Image output{
        out_channels,
        (input.height + 2 * padding - kernel) / stride + 1,
        (input.width + 2 * padding - kernel) / stride + 1,
        {}};
    output.values.resize(static_cast<std::size_t>(count(output)));
    parallel_for(out_channels, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t oc = begin; oc < end; ++oc) {
            float* output_plane = output.values.data() +
                std::uint64_t(oc) * output.height * output.width;
            std::fill_n(
                output_plane,
                std::uint64_t(output.height) * output.width,
                bias.empty() ? 0.0f : bias[oc]);
            for (std::uint32_t ic = 0;
                 ic < input.channels; ++ic) {
                const float* input_plane = input.values.data() +
                    std::uint64_t(ic) * input.height * input.width;
                for (std::uint32_t ky = 0; ky < kernel; ++ky) {
                    int first_y = 0;
                    while (first_y < static_cast<int>(output.height) &&
                           first_y * static_cast<int>(stride) +
                                   static_cast<int>(ky) <
                               static_cast<int>(padding)) {
                        ++first_y;
                    }
                    int last_y = static_cast<int>(output.height);
                    while (last_y > first_y &&
                           (last_y - 1) * static_cast<int>(stride) +
                                   static_cast<int>(ky) -
                                   static_cast<int>(padding) >=
                               static_cast<int>(input.height)) {
                        --last_y;
                    }
                    for (std::uint32_t kx = 0; kx < kernel; ++kx) {
                        int first_x = 0;
                        while (first_x < static_cast<int>(output.width) &&
                               first_x * static_cast<int>(stride) +
                                       static_cast<int>(kx) <
                                   static_cast<int>(padding)) {
                            ++first_x;
                        }
                        int last_x = static_cast<int>(output.width);
                        while (last_x > first_x &&
                               (last_x - 1) *
                                           static_cast<int>(stride) +
                                       static_cast<int>(kx) -
                                       static_cast<int>(padding) >=
                                   static_cast<int>(input.width)) {
                            --last_x;
                        }
                        const float coefficient = weight[
                            ((std::uint64_t(oc) * input.channels + ic) *
                                 kernel +
                             ky) *
                                kernel +
                            kx];
                        for (int oy = first_y; oy < last_y; ++oy) {
                            const std::uint32_t iy =
                                static_cast<std::uint32_t>(
                                    oy * static_cast<int>(stride) +
                                    static_cast<int>(ky) -
                                    static_cast<int>(padding));
                            float* destination =
                                output_plane +
                                std::uint64_t(oy) * output.width +
                                first_x;
                            const float* source =
                                input_plane +
                                std::uint64_t(iy) * input.width +
                                static_cast<std::uint32_t>(
                                    first_x *
                                        static_cast<int>(stride) +
                                    static_cast<int>(kx) -
                                    static_cast<int>(padding));
                            for (int ox = first_x; ox < last_x; ++ox) {
                                *destination++ += *source * coefficient;
                                source += stride;
                            }
                        }
                    }
                }
            }
        }
    });
    return output;
}

Image deconv(
    const ModelFile& model,
    const Image& input,
    const std::string& weight_name,
    const std::string& bias_name) {
    const TensorView& view = model.tensor(weight_name);
    if (view.rank != 4 || view.dimensions[0] != input.channels ||
        view.dimensions[2] != view.dimensions[3]) {
        throw std::runtime_error(
            "Depth Pro transpose convolution mismatch");
    }
    const std::vector<float> weight = decode(view);
    const std::uint32_t outputs =
        static_cast<std::uint32_t>(view.dimensions[1]);
    const std::uint32_t kernel =
        static_cast<std::uint32_t>(view.dimensions[2]);
    std::vector<float> bias;
    if (!bias_name.empty()) {
        bias = decode(model.tensor(bias_name));
    }
    Image output{
        outputs, input.height * kernel, input.width * kernel, {}};
    output.values.resize(static_cast<std::size_t>(count(output)));
    parallel_for(outputs, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t oc = begin; oc < end; ++oc) {
            std::fill(
                output.values.begin() +
                    std::uint64_t(oc) * output.height * output.width,
                output.values.begin() +
                    std::uint64_t(oc + 1) *
                        output.height * output.width,
                bias.empty() ? 0.0f : bias[oc]);
            for (std::uint32_t ic = 0; ic < input.channels; ++ic) {
                for (std::uint32_t iy = 0; iy < input.height; ++iy) {
                    for (std::uint32_t ix = 0; ix < input.width; ++ix) {
                        const float source = input.values[
                            (std::uint64_t(ic) * input.height + iy) *
                                input.width +
                            ix];
                        for (std::uint32_t ky = 0; ky < kernel; ++ky) {
                            for (std::uint32_t kx = 0;
                                 kx < kernel; ++kx) {
                                output.values[
                                    (std::uint64_t(oc) * output.height +
                                     iy * kernel + ky) *
                                        output.width +
                                    ix * kernel + kx] +=
                                    source * weight[
                                        ((std::uint64_t(ic) * outputs + oc) *
                                             kernel +
                                         ky) *
                                            kernel +
                                        kx];
                            }
                        }
                    }
                }
            }
        }
    });
    return output;
}

void relu(Image& image) {
    parallel_for(image.channels, [&](std::uint32_t begin, std::uint32_t end) {
        const std::uint64_t plane =
            std::uint64_t(image.height) * image.width;
        for (std::uint32_t c = begin; c < end; ++c) {
            for (std::uint64_t i = c * plane;
                 i < (c + 1) * plane; ++i) {
                image.values[static_cast<std::size_t>(i)] =
                    std::max(
                        image.values[static_cast<std::size_t>(i)], 0.0f);
            }
        }
    });
}

Image add(Image first, const Image& second) {
    if (first.channels != second.channels ||
        first.height != second.height || first.width != second.width) {
        throw std::runtime_error("Depth Pro addition shape mismatch");
    }
    parallel_for(first.channels, [&](std::uint32_t begin, std::uint32_t end) {
        const std::uint64_t plane =
            std::uint64_t(first.height) * first.width;
        for (std::uint64_t i = std::uint64_t(begin) * plane;
             i < std::uint64_t(end) * plane; ++i) {
            first.values[static_cast<std::size_t>(i)] +=
                second.values[static_cast<std::size_t>(i)];
        }
    });
    return first;
}

Image concatenate(const Image& a, const Image& b) {
    if (a.height != b.height || a.width != b.width) {
        throw std::runtime_error(
            "Depth Pro concatenation shape mismatch");
    }
    Image output{a.channels + b.channels, a.height, a.width, {}};
    output.values = a.values;
    output.values.insert(
        output.values.end(), b.values.begin(), b.values.end());
    return output;
}

Image token_image(
    const std::vector<float>& tokens,
    std::uint32_t channels = 1024) {
    if (tokens.size() != std::uint64_t(577) * channels) {
        throw std::runtime_error("invalid Depth Pro token tensor");
    }
    Image image{channels, 24, 24, {}};
    image.values.resize(static_cast<std::size_t>(count(image)));
    parallel_for(channels, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t c = begin; c < end; ++c) {
            for (std::uint32_t p = 0; p < 576; ++p) {
                image.values[std::uint64_t(c) * 576 + p] =
                    tokens[std::uint64_t(1 + p) * channels + c];
            }
        }
    });
    return image;
}

Image merge(
    const std::vector<Image>& patches,
    std::uint32_t steps,
    std::uint32_t padding) {
    if (patches.size() != std::uint64_t(steps) * steps) {
        throw std::runtime_error("invalid Depth Pro merge patch count");
    }
    const std::uint32_t tile = patches[0].width;
    const std::uint32_t stride = tile - 2 * padding;
    Image output{
        patches[0].channels,
        tile + (steps - 1) * stride,
        tile + (steps - 1) * stride,
        {}};
    output.values.resize(static_cast<std::size_t>(count(output)));
    parallel_for(output.channels, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t c = begin; c < end; ++c) {
            for (std::uint32_t j = 0; j < steps; ++j) {
                for (std::uint32_t i = 0; i < steps; ++i) {
                    const Image& patch = patches[j * steps + i];
                    const std::uint32_t top = j ? padding : 0;
                    const std::uint32_t left = i ? padding : 0;
                    const std::uint32_t bottom =
                        j + 1 < steps ? padding : 0;
                    const std::uint32_t right =
                        i + 1 < steps ? padding : 0;
                    for (std::uint32_t y = top;
                         y < tile - bottom; ++y) {
                        for (std::uint32_t x = left;
                             x < tile - right; ++x) {
                            output.values[
                                (std::uint64_t(c) * output.height +
                                 j * stride + y - top) *
                                    output.width +
                                i * stride + x - left] =
                                patch.values[
                                    (std::uint64_t(c) * tile + y) *
                                        tile +
                                    x];
                        }
                    }
                }
            }
        }
    });
    return output;
}

Image residual(
    const ModelFile& model,
    const Image& input,
    const std::string& base) {
    Image branch = input;
    relu(branch);
    branch = conv(
        model, branch, base + ".1.weight",
        base + ".1.bias", 1, 1);
    relu(branch);
    branch = conv(
        model, branch, base + ".3.weight",
        base + ".3.bias", 1, 1);
    return add(std::move(branch), input);
}

Image fusion(
    const ModelFile& model,
    Image path,
    const Image* skip,
    std::uint32_t level) {
    const std::string base =
        "decoder.fusions." + std::to_string(level);
    if (skip) {
        path = add(
            std::move(path),
            residual(model, *skip, base + ".resnet1.residual"));
    }
    path = residual(model, path, base + ".resnet2.residual");
    if (level != 0) {
        path = deconv(
            model, path, base + ".deconv.weight", "");
    }
    return conv(
        model, path, base + ".out_conv.weight",
        base + ".out_conv.bias", 1, 0);
}

Image project_upsample(
    const ModelFile& model,
    Image image,
    const std::string& base,
    std::uint32_t layers) {
    image = conv(model, image, base + ".0.weight", "", 1, 0);
    for (std::uint32_t i = 0; i < layers; ++i) {
        image = deconv(
            model, image,
            base + "." + std::to_string(i + 1) + ".weight", "");
    }
    return image;
}

}  // namespace

InferenceOutput infer_cpu(
    const ModelFile& model,
    const float* rgb,
    std::uint32_t width,
    std::uint32_t height,
    float forced_fov_degrees) {
    if (!rgb || width == 0 || height == 0) {
        throw std::invalid_argument("invalid Depth Pro image");
    }
    Image input{3, height, width, {}};
    input.values.resize(static_cast<std::size_t>(count(input)));
    for (std::size_t i = 0; i < input.values.size(); ++i) {
        input.values[i] = (rgb[i] - 0.5f) / 0.5f;
    }
    Image x0 = resize(input, 1536, 1536);
    Image x1 = resize(x0, 768, 768);
    Image x2 = resize(x0, 384, 384);

    std::vector<EncoderOutput> patch_encodings;
    patch_encodings.reserve(35);
    for (std::uint32_t j = 0; j < 5; ++j) {
        for (std::uint32_t i = 0; i < 5; ++i) {
            Image patch = crop(x0, j * 288, i * 288, 384);
            patch_encodings.push_back(encoder_cpu(
                model, "encoder.patch_encoder.",
                patch.values.data(), 384, 384));
        }
    }
    for (std::uint32_t j = 0; j < 3; ++j) {
        for (std::uint32_t i = 0; i < 3; ++i) {
            Image patch = crop(x1, j * 192, i * 192, 384);
            patch_encodings.push_back(encoder_cpu(
                model, "encoder.patch_encoder.",
                patch.values.data(), 384, 384));
        }
    }
    patch_encodings.push_back(encoder_cpu(
        model, "encoder.patch_encoder.",
        x2.values.data(), 384, 384));

    std::vector<Image> images;
    for (std::uint32_t i = 0; i < 25; ++i) {
        images.push_back(token_image(
            patch_encodings[i].captures[0]));
    }
    Image latent0 = merge(images, 5, 3);
    images.clear();
    for (std::uint32_t i = 0; i < 25; ++i) {
        images.push_back(token_image(
            patch_encodings[i].captures[1]));
    }
    Image latent1 = merge(images, 5, 3);
    images.clear();
    for (std::uint32_t i = 0; i < 25; ++i) {
        images.push_back(token_image(patch_encodings[i].final));
    }
    Image feature0 = merge(images, 5, 3);
    images.clear();
    for (std::uint32_t i = 25; i < 34; ++i) {
        images.push_back(token_image(patch_encodings[i].final));
    }
    Image feature1 = merge(images, 3, 6);
    Image feature2 = token_image(patch_encodings[34].final);
    patch_encodings.clear();

    const EncoderOutput image_encoded = encoder_cpu(
        model, "encoder.image_encoder.",
        x2.values.data(), 384, 384);
    Image global = token_image(image_encoded.final);
    latent0 = project_upsample(
        model, std::move(latent0),
        "encoder.upsample_latent0", 3);
    latent1 = project_upsample(
        model, std::move(latent1),
        "encoder.upsample_latent1", 2);
    feature0 = project_upsample(
        model, std::move(feature0), "encoder.upsample0", 1);
    feature1 = project_upsample(
        model, std::move(feature1), "encoder.upsample1", 1);
    feature2 = project_upsample(
        model, std::move(feature2), "encoder.upsample2", 1);
    global = deconv(
        model, global, "encoder.upsample_lowres.weight",
        "encoder.upsample_lowres.bias");
    global = conv(
        model, concatenate(feature2, global),
        "encoder.fuse_lowres.weight",
        "encoder.fuse_lowres.bias", 1, 0);

    Image path = conv(
        model, global, "decoder.convs.4.weight", "", 1, 1);
    const Image lowres = path;
    path = fusion(model, std::move(path), nullptr, 4);
    Image projected = conv(
        model, feature1, "decoder.convs.3.weight", "", 1, 1);
    path = fusion(model, std::move(path), &projected, 3);
    projected = conv(
        model, feature0, "decoder.convs.2.weight", "", 1, 1);
    path = fusion(model, std::move(path), &projected, 2);
    projected = conv(
        model, latent1, "decoder.convs.1.weight", "", 1, 1);
    path = fusion(model, std::move(path), &projected, 1);
    path = fusion(model, std::move(path), &latent0, 0);

    path = conv(model, path, "head.0.weight", "head.0.bias", 1, 1);
    path = deconv(model, path, "head.1.weight", "head.1.bias");
    path = conv(model, path, "head.2.weight", "head.2.bias", 1, 1);
    relu(path);
    Image inverse = conv(
        model, path, "head.4.weight", "head.4.bias", 1, 0);
    relu(inverse);

    float fov_degrees = forced_fov_degrees;
    if (!(fov_degrees > 0.0f && fov_degrees < 180.0f)) {
    const EncoderOutput fov_encoded = encoder_cpu(
        model, "fov.encoder.0.",
        x2.values.data(), 384, 384);
    const TensorView& fov_weight_view =
        model.tensor("fov.encoder.1.weight");
    const std::vector<float> fov_weight = decode(fov_weight_view);
    const std::vector<float> fov_bias =
        decode(model.tensor("fov.encoder.1.bias"));
    Image fov_tokens{128, 24, 24, {}};
    fov_tokens.values.resize(static_cast<std::size_t>(count(fov_tokens)));
    parallel_for(128, [&](std::uint32_t begin, std::uint32_t end) {
        for (std::uint32_t oc = begin; oc < end; ++oc) {
            for (std::uint32_t p = 0; p < 576; ++p) {
                float value = fov_bias[oc];
                for (std::uint32_t ic = 0; ic < 1024; ++ic) {
                    value += fov_encoded.final[
                        std::uint64_t(1 + p) * 1024 + ic] *
                        fov_weight[std::uint64_t(oc) * 1024 + ic];
                }
                fov_tokens.values[std::uint64_t(oc) * 576 + p] =
                    value;
            }
        }
    });
    Image fov = conv(
        model, lowres, "fov.downsample.0.weight",
        "fov.downsample.0.bias", 2, 1);
    relu(fov);
    fov = add(std::move(fov), fov_tokens);
    fov = conv(
        model, fov, "fov.head.0.weight", "fov.head.0.bias", 2, 1);
    relu(fov);
    fov = conv(
        model, fov, "fov.head.2.weight", "fov.head.2.bias", 2, 1);
    relu(fov);
    fov = conv(
        model, fov, "fov.head.4.weight", "fov.head.4.bias", 1, 0);
    fov_degrees = fov.values[0];
    }
    const float focal =
        0.5f * width /
        std::tan(
            0.5f * fov_degrees *
            3.14159265358979323846f / 180.0f);
    for (float& value : inverse.values) {
        value *= static_cast<float>(width) / focal;
    }
    inverse = resize(inverse, height, width);
    InferenceOutput output;
    output.focal_length_pixels = focal;
    output.depth.resize(std::uint64_t(width) * height);
    for (std::size_t i = 0; i < output.depth.size(); ++i) {
        output.depth[i] =
            1.0f / std::clamp(inverse.values[i], 1.0e-4f, 1.0e4f);
    }
    return output;
}

}  // namespace depth_pro_native
