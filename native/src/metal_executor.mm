#include "metal_executor.h"
#include "inferbridge/native_harness_metal_texture.h"
#include "inferbridge/native_harness_precision.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace depth_pro_native {
namespace {

MPSShape* shape(std::initializer_list<NSInteger> values) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:values.size()];
    for (NSInteger value : values) [result addObject:@(value)];
    return result;
}
MPSShape* shape(const TensorView& tensor) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:tensor.rank];
    for (std::uint32_t i = 0; i < tensor.rank; ++i)
        [result addObject:@(tensor.dimensions[i])];
    return result;
}
NSString* ns(const std::string& value) {
    return [NSString stringWithUTF8String:value.c_str()];
}

std::string hexadecimal(const std::array<std::uint8_t, 32>& bytes) {
    constexpr char digits[] = "0123456789abcdef";
    std::string result(bytes.size() * 2, '0');
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        result[i * 2] = digits[bytes[i] >> 4];
        result[i * 2 + 1] = digits[bytes[i] & 15];
    }
    return result;
}

struct HostImage {
    int channels;
    int height;
    int width;
    std::vector<float> values;
};

HostImage resize_host(
    const HostImage& input, int output_height, int output_width) {
    HostImage output{input.channels, output_height, output_width,
        std::vector<float>(static_cast<std::size_t>(input.channels) *
                           output_height * output_width)};
    for (int c = 0; c < input.channels; ++c) {
        for (int y = 0; y < output_height; ++y) {
            const float raw_y = (y + 0.5f) * input.height / output_height - 0.5f;
            const float sy = std::clamp(raw_y, 0.0f,
                                        static_cast<float>(input.height - 1));
            const int y0 = static_cast<int>(sy);
            const int y1 = std::min(y0 + 1, input.height - 1);
            const float fy = sy - y0;
            for (int x = 0; x < output_width; ++x) {
                const float raw_x = (x + 0.5f) * input.width / output_width - 0.5f;
                const float sx = std::clamp(raw_x, 0.0f,
                                            static_cast<float>(input.width - 1));
                const int x0 = static_cast<int>(sx);
                const int x1 = std::min(x0 + 1, input.width - 1);
                const float fx = sx - x0;
                const auto at = [&](int py, int px) {
                    return input.values[(static_cast<std::size_t>(c) *
                        input.height + py) * input.width + px];
                };
                output.values[(static_cast<std::size_t>(c) * output_height + y) *
                    output_width + x] =
                    (at(y0, x0) * (1.0f - fx) + at(y0, x1) * fx) *
                        (1.0f - fy) +
                    (at(y1, x0) * (1.0f - fx) + at(y1, x1) * fx) * fy;
            }
        }
    }
    return output;
}

void copy_crop(
    const HostImage& input, int top, int left, float* destination) {
    constexpr int size = 384;
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < size; ++y)
            std::copy_n(input.values.data() +
                (static_cast<std::size_t>(c) * input.height + top + y) *
                    input.width + left,
                size, destination + (static_cast<std::size_t>(c) * size + y) * size);
}

struct EncoderResult {
    std::array<MPSGraphTensor*, 4> captures{};
    MPSGraphTensor* final = nil;
};

class GraphBuilder {
public:
    GraphBuilder(
        const ModelFile& model, bool fp16, bool forced_fov)
        : model_(model), fp16_(fp16),
          forced_fov_(forced_fov), graph_([MPSGraph new]) {}

    void build() {
        patches_ = [graph_ placeholderWithShape:shape({35, 3, 384, 384})
                                        dataType:MPSDataTypeFloat32
                                            name:@"pyramid_patches"];
        image_ = [graph_ placeholderWithShape:shape({1, 3, 384, 384})
                                      dataType:MPSDataTypeFloat32
                                          name:@"image_patch"];
        if (forced_fov_)
            fov_input_ = [graph_ placeholderWithShape:shape({1})
                                              dataType:MPSDataTypeFloat32
                                                  name:@"forced_fov_degrees"];

        EncoderResult patch_encoded = encoder(
            internal(patches_), "encoder.patch_encoder.", 35);
        MPSGraphTensor* latent0 = merge(token_image(
            patch_encoded.captures[0], 35), 5, 3, 0);
        MPSGraphTensor* latent1 = merge(token_image(
            patch_encoded.captures[1], 35), 5, 3, 0);
        MPSGraphTensor* feature0 = merge(token_image(
            patch_encoded.final, 35), 5, 3, 0);
        MPSGraphTensor* feature1 = merge(token_image(
            patch_encoded.final, 35), 3, 6, 25);
        MPSGraphTensor* feature2 = [graph_ sliceTensor:
            token_image(patch_encoded.final, 35) dimension:0
            start:34 length:1 name:nil];

        EncoderResult image_encoded = encoder(
            internal(image_), "encoder.image_encoder.", 1);
        MPSGraphTensor* global = token_image(image_encoded.final, 1);
        latent0 = project_upsample(latent0, "encoder.upsample_latent0", 3);
        latent1 = project_upsample(latent1, "encoder.upsample_latent1", 2);
        feature0 = project_upsample(feature0, "encoder.upsample0", 1);
        feature1 = project_upsample(feature1, "encoder.upsample1", 1);
        feature2 = project_upsample(feature2, "encoder.upsample2", 1);
        global = deconv(global, "encoder.upsample_lowres");
        global = conv([graph_ concatTensors:@[feature2, global]
                                  dimension:1 name:nil],
                      "encoder.fuse_lowres", 1, 0);

        MPSGraphTensor* path = conv(global, "decoder.convs.4", 1, 1, false);
        MPSGraphTensor* lowres = path;
        path = fusion(path, nil, 4);
        MPSGraphTensor* projected = conv(
            feature1, "decoder.convs.3", 1, 1, false);
        path = fusion(path, projected, 3);
        projected = conv(feature0, "decoder.convs.2", 1, 1, false);
        path = fusion(path, projected, 2);
        projected = conv(latent1, "decoder.convs.1", 1, 1, false);
        path = fusion(path, projected, 1);
        path = fusion(path, latent0, 0);
        path = conv(path, "head.0", 1, 1);
        path = deconv(path, "head.1");
        path = [graph_ reLUWithTensor:conv(path, "head.2", 1, 1) name:nil];
        MPSGraphTensor* inverse = [graph_ reLUWithTensor:
            conv(path, "head.4", 1, 0) name:nil];

        MPSGraphTensor* fov = nil;
        if (forced_fov_) {
            fov = internal(fov_input_);
        } else {
            EncoderResult fov_encoded = encoder(
                internal(image_), "fov.encoder.0.", 1);
            MPSGraphTensor* fov_tokens = linear(
                fov_encoded.final, "fov.encoder.1");
            fov_tokens = token_image(fov_tokens, 1, 128);
            fov = [graph_ reLUWithTensor:
                conv(lowres, "fov.downsample.0", 2, 1) name:nil];
            fov = add(fov, fov_tokens);
            fov = [graph_ reLUWithTensor:
                conv(fov, "fov.head.0", 2, 1) name:nil];
            fov = [graph_ reLUWithTensor:
                conv(fov, "fov.head.2", 2, 1) name:nil];
            fov = conv(fov, "fov.head.4", 1, 0);
            fov = [graph_ reshapeTensor:fov withShape:shape({1}) name:nil];
        }
        // Keep the learned graph output at its native 1536 resolution. The
        // exact half-pixel presentation resize and reciprocal are performed
        // below on the host, matching the cross-backend reference indexing.
        depth_ = external(inverse);
        fov_ = external(fov);
    }

    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* patches() const { return patches_; }
    MPSGraphTensor* image() const { return image_; }
    MPSGraphTensor* fov_input() const { return fov_input_; }
    MPSGraphTensor* depth() const { return depth_; }
    MPSGraphTensor* fov() const { return fov_; }

private:
    MPSGraphTensor* internal(MPSGraphTensor* value) {
        return fp16_ ? [graph_ castTensor:value toType:MPSDataTypeFloat16 name:nil]
                     : value;
    }
    MPSGraphTensor* external(MPSGraphTensor* value) {
        return value.dataType == MPSDataTypeFloat32 ? value :
            [graph_ castTensor:value toType:MPSDataTypeFloat32 name:nil];
    }
    MPSGraphTensor* constant(const std::string& name) {
        const TensorView& tensor = model_.tensor(name);
        NSData* data = [NSData dataWithBytesNoCopy:
            const_cast<std::uint16_t*>(tensor.data)
            length:tensor.elements * sizeof(std::uint16_t) freeWhenDone:NO];
        MPSGraphTensor* value = [graph_ constantWithData:data
            shape:shape(tensor) dataType:MPSDataTypeFloat16];
        return fp16_ ? value :
            [graph_ castTensor:value toType:MPSDataTypeFloat32 name:nil];
    }
    MPSGraphTensor* scalar(float value) {
        return internal([graph_ constantWithScalar:value
                                          dataType:MPSDataTypeFloat32]);
    }
    MPSGraphTensor* add(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* multiply(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    MPSGraphTensor* linear(MPSGraphTensor* value, const std::string& prefix) {
        MPSGraphTensor* weight = [graph_ transposeTensor:
            constant(prefix + ".weight") dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* result = [graph_
            matrixMultiplicationWithPrimaryTensor:value
            secondaryTensor:weight name:ns(prefix)];
        return model_.contains(prefix + ".bias")
            ? add(result, constant(prefix + ".bias")) : result;
    }
    MPSGraphTensor* layer_norm(MPSGraphTensor* value, const std::string& prefix) {
        NSArray<NSNumber*>* axes = @[@(-1)];
        MPSGraphTensor* mean = [graph_ meanOfTensor:value axes:axes name:nil];
        MPSGraphTensor* variance = [graph_ varianceOfTensor:value
            meanTensor:mean axes:axes name:nil];
        return [graph_ normalizationWithTensor:value meanTensor:mean
            varianceTensor:variance gammaTensor:constant(prefix + ".weight")
            betaTensor:constant(prefix + ".bias") epsilon:1.0e-6f name:ns(prefix)];
    }
    MPSGraphTensor* gelu(MPSGraphTensor* value) {
        MPSGraphTensor* error = [graph_ erfWithTensor:
            multiply(value, scalar(static_cast<float>(M_SQRT1_2))) name:nil];
        return multiply(multiply(value, scalar(0.5f)),
                        add(error, scalar(1.0f)));
    }
    MPSGraphConvolution2DOpDescriptor* descriptor(int stride, int padding) {
        return [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride strideInY:stride
            dilationRateInX:1 dilationRateInY:1 groups:1
            paddingLeft:padding paddingRight:padding
            paddingTop:padding paddingBottom:padding
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    }
    MPSGraphTensor* conv(
        MPSGraphTensor* value, const std::string& prefix,
        int stride, int padding, bool bias = true) {
        MPSGraphTensor* result = [graph_ convolution2DWithSourceTensor:value
            weightsTensor:constant(prefix + ".weight")
            descriptor:descriptor(stride, padding) name:ns(prefix)];
        if (bias && model_.contains(prefix + ".bias")) {
            const int channels = static_cast<int>(
                model_.tensor(prefix + ".bias").elements);
            result = add(result, [graph_ reshapeTensor:constant(prefix + ".bias")
                withShape:shape({1, channels, 1, 1}) name:nil]);
        }
        return result;
    }
    MPSGraphTensor* deconv(MPSGraphTensor* value, const std::string& prefix) {
        const TensorView& weight = model_.tensor(prefix + ".weight");
        const int channels = static_cast<int>(weight.dimensions[1]);
        const int kernel = static_cast<int>(weight.dimensions[2]);
        const int input_channels = static_cast<int>(weight.dimensions[0]);
        const int batch = [value.shape[0] intValue];
        const int source_height = [value.shape[2] intValue];
        const int source_width = [value.shape[3] intValue];
        // These layers are stride==kernel, padding==0. The reference backend
        // scatters weight[in,out,ky,kx] directly, so express that exact indexing
        // as a 1x1 projection followed by pixel shuffle. A generic transpose
        // convolution may reverse the spatial kernel on some implementations.
        MPSGraphTensor* projected_weight = [graph_ transposeTensor:
            constant(prefix + ".weight")
            permutation:@[@1, @2, @3, @0] name:nil];
        projected_weight = [graph_ reshapeTensor:projected_weight
            withShape:shape({channels * kernel * kernel,
                             input_channels, 1, 1}) name:nil];
        MPSGraphTensor* result = [graph_ convolution2DWithSourceTensor:value
            weightsTensor:projected_weight descriptor:descriptor(1, 0)
            name:ns(prefix)];
        result = [graph_ reshapeTensor:result withShape:
            shape({batch, channels, kernel, kernel,
                   source_height, source_width}) name:nil];
        result = [graph_ transposeTensor:result
            permutation:@[@0, @1, @4, @2, @5, @3] name:nil];
        result = [graph_ reshapeTensor:result withShape:
            shape({batch, channels, source_height * kernel,
                   source_width * kernel}) name:nil];
        if (model_.contains(prefix + ".bias"))
            result = add(result, [graph_ reshapeTensor:constant(prefix + ".bias")
                withShape:shape({1, channels, 1, 1}) name:nil]);
        return result;
    }
    MPSGraphTensor* resize(
        MPSGraphTensor* value, int height, int width, bool align_corners) {
        return [graph_ resizeTensor:value size:shape({height, width})
            mode:MPSGraphResizeBilinear centerResult:align_corners ? NO : YES
            alignCorners:align_corners ? YES : NO
            layout:MPSGraphTensorNamedDataLayoutNCHW name:nil];
    }
    EncoderResult encoder(
        MPSGraphTensor* input, const std::string& prefix, int batch) {
        MPSGraphTensor* current = conv(input, prefix + "patch_embed.proj", 16, 0);
        current = [graph_ reshapeTensor:current
            withShape:shape({batch, 1024, 576}) name:nil];
        current = [graph_ transposeTensor:current dimension:1
                            withDimension:2 name:nil];
        MPSGraphTensor* cls = constant(prefix + "cls_token");
        if (batch != 1)
            cls = [graph_ tileTensor:cls withMultiplier:shape({batch, 1, 1})
                                name:nil];
        current = add([graph_ concatTensors:@[cls, current]
                                  dimension:1 name:nil],
                      constant(prefix + "pos_embed"));
        EncoderResult output;
        int capture = 0;
        for (int block = 0; block < 24; ++block) {
            const std::string base = prefix + "blocks." +
                std::to_string(block) + ".";
            MPSGraphTensor* normalized = layer_norm(current, base + "norm1");
            MPSGraphTensor* qkv = linear(normalized, base + "attn.qkv");
            qkv = [graph_ reshapeTensor:qkv
                withShape:shape({batch, 577, 3, 16, 64}) name:nil];
            qkv = [graph_ transposeTensor:qkv
                permutation:@[@2, @0, @3, @1, @4] name:nil];
            MPSGraphTensor* q = [graph_ sliceTensor:qkv dimension:0
                                              start:0 length:1 name:nil];
            MPSGraphTensor* k = [graph_ sliceTensor:qkv dimension:0
                                              start:1 length:1 name:nil];
            MPSGraphTensor* v = [graph_ sliceTensor:qkv dimension:0
                                              start:2 length:1 name:nil];
            q = [graph_ reshapeTensor:q withShape:shape({batch, 16, 577, 64})
                                 name:nil];
            k = [graph_ reshapeTensor:k withShape:shape({batch, 16, 577, 64})
                                 name:nil];
            v = [graph_ reshapeTensor:v withShape:shape({batch, 16, 577, 64})
                                 name:nil];
            q = multiply(q, scalar(0.125f));
            k = [graph_ transposeTensor:k dimension:2 withDimension:3 name:nil];
            MPSGraphTensor* scores = [graph_
                matrixMultiplicationWithPrimaryTensor:q secondaryTensor:k name:nil];
            scores = [graph_ softMaxWithTensor:scores axis:-1 name:nil];
            MPSGraphTensor* attention = [graph_
                matrixMultiplicationWithPrimaryTensor:scores
                secondaryTensor:v name:nil];
            attention = [graph_ transposeTensor:attention dimension:1
                                  withDimension:2 name:nil];
            attention = [graph_ reshapeTensor:attention
                withShape:shape({batch, 577, 1024}) name:nil];
            current = add(current, multiply(
                linear(attention, base + "attn.proj"),
                constant(base + "ls1.gamma")));
            MPSGraphTensor* hidden = gelu(linear(
                layer_norm(current, base + "norm2"), base + "mlp.fc1"));
            current = add(current, multiply(
                linear(hidden, base + "mlp.fc2"),
                constant(base + "ls2.gamma")));
            if (block == 5 || block == 11 || block == 17 || block == 23)
                output.captures[capture++] = current;
        }
        output.final = layer_norm(current, prefix + "norm");
        return output;
    }
    MPSGraphTensor* token_image(
        MPSGraphTensor* tokens, int batch, int channels = 1024) {
        MPSGraphTensor* patches = [graph_ sliceTensor:tokens dimension:1
            start:1 length:576 name:nil];
        patches = [graph_ transposeTensor:patches dimension:1
                            withDimension:2 name:nil];
        return [graph_ reshapeTensor:patches
            withShape:shape({batch, channels, 24, 24}) name:nil];
    }
    MPSGraphTensor* crop_tile(
        MPSGraphTensor* batch, int index, int top, int left,
        int bottom, int right) {
        MPSGraphTensor* tile = [graph_ sliceTensor:batch dimension:0
            start:index length:1 name:nil];
        tile = [graph_ sliceTensor:tile dimension:2 start:top
            length:24 - top - bottom name:nil];
        return [graph_ sliceTensor:tile dimension:3 start:left
            length:24 - left - right name:nil];
    }
    MPSGraphTensor* merge(
        MPSGraphTensor* batch, int steps, int padding, int offset) {
        NSMutableArray<MPSGraphTensor*>* rows = [NSMutableArray array];
        for (int y = 0; y < steps; ++y) {
            NSMutableArray<MPSGraphTensor*>* columns = [NSMutableArray array];
            for (int x = 0; x < steps; ++x)
                [columns addObject:crop_tile(batch, offset + y * steps + x,
                    y ? padding : 0, x ? padding : 0,
                    y + 1 < steps ? padding : 0,
                    x + 1 < steps ? padding : 0)];
            [rows addObject:[graph_ concatTensors:columns dimension:3 name:nil]];
        }
        return [graph_ concatTensors:rows dimension:2 name:nil];
    }
    MPSGraphTensor* project_upsample(
        MPSGraphTensor* value, const std::string& prefix, int layers) {
        value = conv(value, prefix + ".0", 1, 0, false);
        for (int i = 0; i < layers; ++i)
            value = deconv(value, prefix + "." + std::to_string(i + 1));
        return value;
    }
    MPSGraphTensor* residual(
        MPSGraphTensor* value, const std::string& prefix) {
        MPSGraphTensor* branch = [graph_ reLUWithTensor:value name:nil];
        branch = conv(branch, prefix + ".1", 1, 1);
        branch = [graph_ reLUWithTensor:branch name:nil];
        return add(value, conv(branch, prefix + ".3", 1, 1));
    }
    MPSGraphTensor* fusion(
        MPSGraphTensor* path, MPSGraphTensor* skip, int level) {
        const std::string base = "decoder.fusions." + std::to_string(level);
        if (skip != nil)
            path = add(path, residual(skip, base + ".resnet1.residual"));
        path = residual(path, base + ".resnet2.residual");
        if (level != 0) path = deconv(path, base + ".deconv");
        return conv(path, base + ".out_conv", 1, 0);
    }

    const ModelFile& model_;
    bool fp16_;
    bool forced_fov_;
    MPSGraph* graph_;
    MPSGraphTensor* patches_ = nil;
    MPSGraphTensor* image_ = nil;
    MPSGraphTensor* fov_input_ = nil;
    MPSGraphTensor* depth_ = nil;
    MPSGraphTensor* fov_ = nil;
};

struct PlanKey {
    bool forced;
    bool operator==(const PlanKey& other) const {
        return forced == other.forced;
    }
};
struct PlanHash {
    std::size_t operator()(const PlanKey& key) const {
        return static_cast<std::size_t>(key.forced);
    }
};
struct Plan {
    MPSGraph* graph = nil;
    MPSGraphExecutable* executable = nil;
    bool forced = false;
};

class MetalExternalJob final : public ExternalJob {
public:
    explicit MetalExternalJob(
        std::shared_ptr<inferbridge::native_harness::metal::Submission> value)
        : submission_(std::move(value)) {}
    ExternalJobState state() const override {
        if (submission_->cancelled()) return ExternalJobState::cancelled;
        return submission_->complete() ? ExternalJobState::complete :
            ExternalJobState::running;
    }
    void cancel() override { submission_->cancel(); }
private:
    std::shared_ptr<inferbridge::native_harness::metal::Submission> submission_;
};

}  // namespace

class MetalExecutor::Impl {
public:
    explicit Impl(const ModelFile& model) : model_(model) {
        const auto precision = inferbridge::native::requested_precision();
        if (precision == inferbridge::native::Precision::int8)
            throw std::invalid_argument("Depth Pro Metal does not support INT8 yet");
        fp16_ = precision == inferbridge::native::Precision::fp16 ||
            precision == inferbridge::native::Precision::automatic;
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) throw std::runtime_error("Metal is unavailable");
        queue_ = [device_ newCommandQueue];
        graph_device_ = [MPSGraphDevice deviceWithMTLDevice:device_];
        if (queue_ == nil || graph_device_ == nil)
            throw std::runtime_error("could not initialize Depth Pro Metal");
        create_texture_pipelines();
    }

    InferenceOutput infer(
        const float* rgb, std::uint32_t width, std::uint32_t height,
        float forced_fov_degrees) {
        const bool forced = forced_fov_degrees > 0.0f &&
            forced_fov_degrees < 180.0f;
        HostImage input{3, static_cast<int>(height), static_cast<int>(width),
            std::vector<float>(static_cast<std::size_t>(3) * width * height)};
        for (std::size_t i = 0; i < input.values.size(); ++i)
            input.values[i] = (rgb[i] - 0.5f) / 0.5f;
        HostImage x0 = resize_host(input, 1536, 1536);
        HostImage x1 = resize_host(x0, 768, 768);
        HostImage x2 = resize_host(x0, 384, 384);
        constexpr std::size_t patch_elements = 3u * 384u * 384u;
        std::vector<float> patches(35u * patch_elements);
        for (int y = 0; y < 5; ++y)
            for (int x = 0; x < 5; ++x)
                copy_crop(x0, y * 288, x * 288,
                          patches.data() + (y * 5 + x) * patch_elements);
        for (int y = 0; y < 3; ++y)
            for (int x = 0; x < 3; ++x)
                copy_crop(x1, y * 192, x * 192,
                          patches.data() + (25 + y * 3 + x) * patch_elements);
        std::copy(x2.values.begin(), x2.values.end(),
                  patches.begin() + 34 * patch_elements);

        std::lock_guard<std::mutex> guard(mutex_);
        @autoreleasepool {
            const Plan& plan = get_plan(forced);
            id<MTLBuffer> patch_buffer = [device_ newBufferWithBytes:patches.data()
                length:patches.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> image_buffer = [device_ newBufferWithBytes:x2.values.data()
                length:x2.values.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
            if (patch_buffer == nil || image_buffer == nil) throw std::bad_alloc();
            NSMutableArray<MPSGraphTensorData*>* inputs = [NSMutableArray array];
            [inputs addObject:[[MPSGraphTensorData alloc]
                initWithMTLBuffer:patch_buffer shape:shape({35, 3, 384, 384})
                dataType:MPSDataTypeFloat32]];
            [inputs addObject:[[MPSGraphTensorData alloc]
                initWithMTLBuffer:image_buffer shape:shape({1, 3, 384, 384})
                dataType:MPSDataTypeFloat32]];
            id<MTLBuffer> fov_buffer = nil;
            if (forced) {
                fov_buffer = [device_ newBufferWithBytes:&forced_fov_degrees
                    length:sizeof(float) options:MTLResourceStorageModeShared];
                [inputs addObject:[[MPSGraphTensorData alloc]
                    initWithMTLBuffer:fov_buffer shape:shape({1})
                    dataType:MPSDataTypeFloat32]];
            }
            MPSGraphExecutableExecutionDescriptor* execution =
                [MPSGraphExecutableExecutionDescriptor new];
            execution.waitUntilCompleted = YES;
            NSArray<MPSGraphTensorData*>* results = [plan.executable
                runWithMTLCommandQueue:queue_ inputsArray:inputs
                resultsArray:nil executionDescriptor:execution];
            if (results.count != 2)
                throw std::runtime_error("Depth Pro Metal returned incomplete outputs");
            std::vector<float> inverse(1536u * 1536u);
            float fov_degrees = 0.0f;
            [results[0].mpsndarray readBytes:inverse.data() strideBytes:nil];
            [results[1].mpsndarray readBytes:&fov_degrees strideBytes:nil];
            InferenceOutput output;
            output.focal_length_pixels = 0.5f * width /
                std::tan(0.5f * fov_degrees * static_cast<float>(M_PI / 180.0));
            output.depth.resize(static_cast<std::size_t>(width) * height);
            const float inverse_scale =
                static_cast<float>(width) / output.focal_length_pixels;
            for (std::uint32_t y = 0; y < height; ++y) {
                const float raw_y = (static_cast<float>(y) + 0.5f) *
                    1536.0f / height - 0.5f;
                const float sy = std::clamp(raw_y, 0.0f, 1535.0f);
                const std::uint32_t y0 = static_cast<std::uint32_t>(sy);
                const std::uint32_t y1 = std::min(y0 + 1, 1535u);
                const float fy = sy - y0;
                for (std::uint32_t x = 0; x < width; ++x) {
                    const float raw_x = (static_cast<float>(x) + 0.5f) *
                        1536.0f / width - 0.5f;
                    const float sx = std::clamp(raw_x, 0.0f, 1535.0f);
                    const std::uint32_t x0 = static_cast<std::uint32_t>(sx);
                    const std::uint32_t x1 = std::min(x0 + 1, 1535u);
                    const float fx = sx - x0;
                    const float top = inverse[y0 * 1536u + x0] * (1.0f - fx) +
                        inverse[y0 * 1536u + x1] * fx;
                    const float bottom = inverse[y1 * 1536u + x0] * (1.0f - fx) +
                        inverse[y1 * 1536u + x1] * fx;
                    const float value = std::clamp(
                        (top * (1.0f - fy) + bottom * fy) * inverse_scale,
                        1.0e-4f, 1.0e4f);
                    output.depth[static_cast<std::size_t>(y) * width + x] =
                        1.0f / value;
                }
            }
            return output;
        }
    }

    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request, float forced_fov_degrees) {
        if (!request.shared_texture_handle || !request.output_texture_handle ||
            !request.signal_fence_handle || !request.signal_fence_value ||
            !request.width || !request.height ||
            request.output_width != request.width ||
            request.output_height != request.height)
            throw std::invalid_argument("invalid Depth Pro Metal texture request");
        inferbridge::native_harness::metal::Prepared prepared;
        prepared.input_texture = (__bridge id<MTLTexture>)(
            reinterpret_cast<void*>(request.shared_texture_handle));
        prepared.output_texture = (__bridge id<MTLTexture>)(
            reinterpret_cast<void*>(request.output_texture_handle));
        prepared.wait_event = request.wait_fence_handle ?
            (__bridge id<MTLSharedEvent>)(reinterpret_cast<void*>(
                request.wait_fence_handle)) : nil;
        prepared.signal_event = (__bridge id<MTLSharedEvent>)(
            reinterpret_cast<void*>(request.signal_fence_handle));
        prepared.signal_value = request.signal_fence_value;
        if (prepared.input_texture.device.registryID != device_.registryID ||
            prepared.output_texture.device.registryID != device_.registryID ||
            prepared.input_texture.textureType != MTLTextureType2D ||
            prepared.input_texture.width != request.width ||
            prepared.input_texture.height != request.height ||
            prepared.input_texture.pixelFormat != MTLPixelFormatBGRA8Unorm ||
            prepared.output_texture.textureType != MTLTextureType2D ||
            prepared.output_texture.width != request.output_width ||
            prepared.output_texture.height != request.output_height ||
            prepared.output_texture.pixelFormat != MTLPixelFormatR32Float)
            throw std::invalid_argument("Depth Pro Metal texture descriptor mismatch");
        constexpr NSUInteger x0_count = 3u * 1536u * 1536u;
        constexpr NSUInteger x1_count = 3u * 768u * 768u;
        constexpr NSUInteger x2_count = 3u * 384u * 384u;
        constexpr NSUInteger patch_count = 35u * x2_count;
        std::lock_guard<std::mutex> guard(mutex_);
        @autoreleasepool {
            id<MTLBuffer> x0 = [device_ newBufferWithLength:x0_count*sizeof(float)
                options:MTLResourceStorageModePrivate];
            id<MTLBuffer> x1 = [device_ newBufferWithLength:x1_count*sizeof(float)
                options:MTLResourceStorageModePrivate];
            id<MTLBuffer> x2 = [device_ newBufferWithLength:x2_count*sizeof(float)
                options:MTLResourceStorageModePrivate];
            id<MTLBuffer> patches = [device_ newBufferWithLength:patch_count*sizeof(float)
                options:MTLResourceStorageModePrivate];
            id<MTLBuffer> inverse = [device_ newBufferWithLength:
                1536u*1536u*sizeof(float) options:MTLResourceStorageModePrivate];
            id<MTLBuffer> fov = [device_ newBufferWithLength:sizeof(float)
                options:MTLResourceStorageModePrivate];
            id<MTLBuffer> forced = [device_ newBufferWithBytes:&forced_fov_degrees
                length:sizeof(float) options:MTLResourceStorageModeShared];
            if (!x0 || !x1 || !x2 || !patches || !inverse || !fov || !forced)
                throw std::bad_alloc();
            id<MTLCommandBuffer> preprocess = [queue_ commandBuffer];
            if (prepared.wait_event)
                [preprocess encodeWaitForEvent:prepared.wait_event
                    value:request.wait_fence_value];
            id<MTLComputeCommandEncoder> encoder = [preprocess computeCommandEncoder];
            struct ResizeParameters { uint32_t sw, sh, dw, dh; };
            ResizeParameters p0{request.width, request.height,1536u,1536u};
            [encoder setComputePipelineState:texture_resize_pipeline_];
            [encoder setTexture:prepared.input_texture atIndex:0];
            [encoder setBuffer:x0 offset:0 atIndex:0];
            [encoder setBytes:&p0 length:sizeof(p0) atIndex:1];
            dispatch(encoder, texture_resize_pipeline_,1536,1536,3);
            [encoder setComputePipelineState:buffer_resize_pipeline_];
            ResizeParameters p1{1536,1536,768,768};
            [encoder setBuffer:x0 offset:0 atIndex:0]; [encoder setBuffer:x1 offset:0 atIndex:1];
            [encoder setBytes:&p1 length:sizeof(p1) atIndex:2];
            dispatch(encoder, buffer_resize_pipeline_,768,768,3);
            ResizeParameters p2{1536,1536,384,384};
            [encoder setBuffer:x0 offset:0 atIndex:0]; [encoder setBuffer:x2 offset:0 atIndex:1];
            [encoder setBytes:&p2 length:sizeof(p2) atIndex:2];
            dispatch(encoder, buffer_resize_pipeline_,384,384,3);
            [encoder setComputePipelineState:pack_pipeline_];
            [encoder setBuffer:x0 offset:0 atIndex:0]; [encoder setBuffer:x1 offset:0 atIndex:1];
            [encoder setBuffer:x2 offset:0 atIndex:2]; [encoder setBuffer:patches offset:0 atIndex:3];
            dispatch(encoder, pack_pipeline_,384,384,35*3);
            [encoder endEncoding]; [preprocess commit];
            const bool use_forced_fov = forced_fov_degrees > 0.0f;
            const Plan& plan = get_plan(use_forced_fov);
            NSMutableArray<MPSGraphTensorData*>* inputs =
                [NSMutableArray arrayWithObjects:
                    [[MPSGraphTensorData alloc] initWithMTLBuffer:patches
                        shape:shape({35,3,384,384})
                        dataType:MPSDataTypeFloat32],
                    [[MPSGraphTensorData alloc] initWithMTLBuffer:x2
                        shape:shape({1,3,384,384})
                        dataType:MPSDataTypeFloat32], nil];
            if (use_forced_fov) {
                [inputs addObject:[[MPSGraphTensorData alloc]
                    initWithMTLBuffer:forced shape:shape({1})
                    dataType:MPSDataTypeFloat32]];
            }
            NSArray<MPSGraphTensorData*>* outputs = @[
                [[MPSGraphTensorData alloc] initWithMTLBuffer:inverse
                    shape:shape({1,1,1536,1536}) dataType:MPSDataTypeFloat32],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:fov
                    shape:shape({1}) dataType:MPSDataTypeFloat32]];
            MPSGraphExecutableExecutionDescriptor* execution =
                [MPSGraphExecutableExecutionDescriptor new];
            execution.waitUntilCompleted = NO;
            NSArray* results = [plan.executable runAsyncWithMTLCommandQueue:queue_
                inputsArray:inputs resultsArray:outputs executionDescriptor:execution];
            if (results.count != 2u)
                throw std::runtime_error("Depth Pro Metal output binding failed");
            id<MTLCommandBuffer> completion = [queue_ commandBuffer];
            encoder = [completion computeCommandEncoder];
            struct FinalParameters { uint32_t width,height; } final{request.width,request.height};
            [encoder setComputePipelineState:final_pipeline_];
            [encoder setBuffer:inverse offset:0 atIndex:0];
            [encoder setBuffer:fov offset:0 atIndex:1];
            [encoder setTexture:prepared.output_texture atIndex:0];
            [encoder setBytes:&final length:sizeof(final) atIndex:2];
            dispatch(encoder, final_pipeline_,request.width,request.height,1);
            [encoder endEncoding];
            [completion encodeSignalEvent:prepared.signal_event value:prepared.signal_value];
            [completion commit];
            return std::make_shared<MetalExternalJob>(
                std::make_shared<inferbridge::native_harness::metal::Submission>(
                    prepared, completion));
        }
    }

private:
    static void dispatch(id<MTLComputeCommandEncoder> encoder,
        id<MTLComputePipelineState> pipeline, NSUInteger width,
        NSUInteger height, NSUInteger depth) {
        const NSUInteger x = pipeline.threadExecutionWidth;
        const NSUInteger y = std::max<NSUInteger>(1,
            pipeline.maxTotalThreadsPerThreadgroup / x);
        [encoder dispatchThreads:MTLSizeMake(width,height,depth)
            threadsPerThreadgroup:MTLSizeMake(x,y,1)];
    }

    void create_texture_pipelines() {
        static constexpr char source_text[] = R"METAL(
#include <metal_stdlib>
using namespace metal;
struct ResizeParameters { uint sw,sh,dw,dh; };
float bilinear(device const float* src,uint c,float sx,float sy,uint w,uint h){
 sx=clamp(sx,0.0f,float(w-1)); sy=clamp(sy,0.0f,float(h-1));
 uint x0=uint(sx),y0=uint(sy),x1=min(x0+1,w-1),y1=min(y0+1,h-1);
 float fx=sx-float(x0),fy=sy-float(y0); uint plane=w*h,base=c*plane;
 return mix(mix(src[base+y0*w+x0],src[base+y0*w+x1],fx),
            mix(src[base+y1*w+x0],src[base+y1*w+x1],fx),fy);
}
kernel void texture_resize(texture2d<float,access::read> src [[texture(0)]],
 device float* dst [[buffer(0)]],constant ResizeParameters&p [[buffer(1)]],
 uint3 q [[thread_position_in_grid]]){
 if(q.x>=p.dw||q.y>=p.dh||q.z>=3)return;
 float sx=(float(q.x)+0.5f)*float(p.sw)/float(p.dw)-0.5f;
 float sy=(float(q.y)+0.5f)*float(p.sh)/float(p.dh)-0.5f;
 sx=clamp(sx,0.0f,float(p.sw-1));sy=clamp(sy,0.0f,float(p.sh-1));
 uint x0=uint(sx),y0=uint(sy),x1=min(x0+1,p.sw-1),y1=min(y0+1,p.sh-1);
 float fx=sx-float(x0),fy=sy-float(y0);
 float a=src.read(uint2(x0,y0))[q.z],b=src.read(uint2(x1,y0))[q.z];
 float c=src.read(uint2(x0,y1))[q.z],d=src.read(uint2(x1,y1))[q.z];
 dst[q.z*p.dw*p.dh+q.y*p.dw+q.x]=mix(mix(a,b,fx),mix(c,d,fx),fy)*2.0f-1.0f;
}
kernel void buffer_resize(device const float*src [[buffer(0)]],device float*dst [[buffer(1)]],
 constant ResizeParameters&p [[buffer(2)]],uint3 q [[thread_position_in_grid]]){
 if(q.x>=p.dw||q.y>=p.dh||q.z>=3)return;
 float sx=(float(q.x)+0.5f)*float(p.sw)/float(p.dw)-0.5f;
 float sy=(float(q.y)+0.5f)*float(p.sh)/float(p.dh)-0.5f;
 dst[q.z*p.dw*p.dh+q.y*p.dw+q.x]=bilinear(src,q.z,sx,sy,p.sw,p.sh);
}
kernel void pack(device const float*x0 [[buffer(0)]],device const float*x1 [[buffer(1)]],
 device const float*x2 [[buffer(2)]],device float*dst [[buffer(3)]],
 uint3 q [[thread_position_in_grid]]){
 if(q.x>=384||q.y>=384||q.z>=105)return; uint image=q.z/3,channel=q.z%3;
 device const float*src=x2;uint w=384,ox=0,oy=0;
 if(image<25){src=x0;w=1536;ox=(image%5)*288;oy=(image/5)*288;}
 else if(image<34){src=x1;w=768;uint n=image-25;ox=(n%3)*192;oy=(n/3)*192;}
 dst[((image*3+channel)*384+q.y)*384+q.x]=src[(channel*w+oy+q.y)*w+ox+q.x];
}
struct FinalParameters{uint width,height;};
kernel void final_depth(device const float*inv [[buffer(0)]],device const float*fov [[buffer(1)]],
 texture2d<float,access::write>out [[texture(0)]],constant FinalParameters&p [[buffer(2)]],
 uint2 q [[thread_position_in_grid]]){
 if(q.x>=p.width||q.y>=p.height)return;
 float sx=(float(q.x)+0.5f)*1536.0f/float(p.width)-0.5f;
 float sy=(float(q.y)+0.5f)*1536.0f/float(p.height)-0.5f;
 float inverse=bilinear(inv,0,sx,sy,1536,1536);
 float focal=0.5f*float(p.width)/tan(0.5f*fov[0]*0.017453292519943295f);
 float value=clamp(inverse*float(p.width)/focal,1.0e-4f,1.0e4f);
 out.write(float4(1.0f/value),q);
}
)METAL";
        NSError* error=nil;
        id<MTLLibrary> library=[device_ newLibraryWithSource:
            [NSString stringWithUTF8String:source_text] options:nil error:&error];
        if(!library)throw std::runtime_error(error.localizedDescription.UTF8String?:
            "could not compile Depth Pro Metal texture kernels");
        auto make=[&](NSString* name){
            id<MTLComputePipelineState> value=[device_ newComputePipelineStateWithFunction:
                [library newFunctionWithName:name] error:&error];
            if(!value)throw std::runtime_error(error.localizedDescription.UTF8String?:
                "could not create Depth Pro Metal texture pipeline");
            return value;
        };
        texture_resize_pipeline_=make(@"texture_resize");
        buffer_resize_pipeline_=make(@"buffer_resize");
        pack_pipeline_=make(@"pack");
        final_pipeline_=make(@"final_depth");
    }

    const Plan& get_plan(bool forced) {
        const PlanKey key{forced};
        auto found = plans_.find(key);
        if (found != plans_.end()) return found->second;
        MPSGraphCompilationDescriptor* descriptor =
            [MPSGraphCompilationDescriptor new];
        // Depth Pro's 35-image encoder batch is deliberately kept as a Metal
        // graph. Level 1 may partition this graph to ANE, whose compiler takes
        // several minutes and duplicates the large ViT constants.
        descriptor.optimizationLevel = MPSGraphOptimizationLevel0;
        descriptor.waitForCompilationCompletion = YES;
        NSURL* package = cache_url(forced);
        MPSGraphExecutable* executable = nil;
        if (@available(macOS 14.0, *)) {
            if (package != nil && [[NSFileManager defaultManager]
                    fileExistsAtPath:package.path]) {
                @try {
                    executable = [[MPSGraphExecutable alloc]
                        initWithMPSGraphPackageAtURL:package
                        compilationDescriptor:descriptor];
                } @catch (NSException*) {
                    [[NSFileManager defaultManager]
                        removeItemAtURL:package error:nil];
                    executable = nil;
                }
            }
        }
        MPSGraph* graph = nil;
        if (executable == nil) {
            GraphBuilder builder(model_, fp16_, forced);
            builder.build();
            graph = builder.graph();
            NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds =
                [NSMutableDictionary dictionary];
            feeds[builder.patches()] = [[MPSGraphShapedType alloc]
                initWithShape:shape({35, 3, 384, 384})
                dataType:MPSDataTypeFloat32];
            feeds[builder.image()] = [[MPSGraphShapedType alloc]
                initWithShape:shape({1, 3, 384, 384})
                dataType:MPSDataTypeFloat32];
            if (forced)
                feeds[builder.fov_input()] = [[MPSGraphShapedType alloc]
                    initWithShape:shape({1}) dataType:MPSDataTypeFloat32];
            executable = [builder.graph()
                compileWithDevice:graph_device_ feeds:feeds
                targetTensors:@[builder.depth(), builder.fov()]
                targetOperations:nil compilationDescriptor:descriptor];
            if (@available(macOS 14.0, *)) {
                if (executable != nil && package != nil) {
                    @try {
                        [executable serializeToMPSGraphPackageAtURL:package
                                                         descriptor:nil];
                    } @catch (NSException*) {
                        [[NSFileManager defaultManager]
                            removeItemAtURL:package error:nil];
                    }
                }
            }
        }
        if (executable == nil)
            throw std::runtime_error("failed to compile Depth Pro Metal graph");
        executable.options = MPSGraphOptionsSynchronizeResults;
        Plan plan{graph, executable, forced};
        return plans_.emplace(key, std::move(plan)).first->second;
    }

    NSURL* cache_url(bool forced) const {
        if (@available(macOS 14.0, *)) {
            NSArray<NSString*>* directories =
                NSSearchPathForDirectoriesInDomains(
                    NSCachesDirectory, NSUserDomainMask, YES);
            if (directories.count == 0) return nil;
            NSString* directory = [directories.firstObject
                stringByAppendingPathComponent:
                    @"DepthExtractor/DepthProMetalGraphCache-v5"];
            if (![[NSFileManager defaultManager]
                    createDirectoryAtPath:directory
                    withIntermediateDirectories:YES
                    attributes:nil error:nil]) return nil;
            const NSOperatingSystemVersion os =
                NSProcessInfo.processInfo.operatingSystemVersion;
            const std::string key = hexadecimal(
                model_.derivation().canonical_sha256) + "-" +
                (forced ? "forced" : "estimated") + "-" +
                (fp16_ ? "fp16" : "fp32") + "-" +
                std::to_string(device_.registryID) + "-macos" +
                std::to_string(os.majorVersion) + "." +
                std::to_string(os.minorVersion) + ".mpsgraphpackage";
            return [NSURL fileURLWithPath:[directory
                stringByAppendingPathComponent:ns(key)]];
        }
        return nil;
    }

    const ModelFile& model_;
    bool fp16_ = false;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    MPSGraphDevice* graph_device_ = nil;
    std::unordered_map<PlanKey, Plan, PlanHash> plans_;
    std::mutex mutex_;
    id<MTLComputePipelineState> texture_resize_pipeline_=nil;
    id<MTLComputePipelineState> buffer_resize_pipeline_=nil;
    id<MTLComputePipelineState> pack_pipeline_=nil;
    id<MTLComputePipelineState> final_pipeline_=nil;
};

MetalExecutor::MetalExecutor(const ModelFile& model)
    : impl_(std::make_unique<Impl>(model)) {}
MetalExecutor::~MetalExecutor() = default;
InferenceOutput MetalExecutor::infer(
    const float* rgb, std::uint32_t width, std::uint32_t height,
    float forced_fov_degrees) {
    return impl_->infer(rgb, width, height, forced_fov_degrees);
}

std::shared_ptr<ExternalJob> MetalExecutor::submit_texture(
    const ExternalTextureRequest& request, float forced_fov_degrees) {
    return impl_->submit_texture(request, forced_fov_degrees);
}

}  // namespace depth_pro_native
