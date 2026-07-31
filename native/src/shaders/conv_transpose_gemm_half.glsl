#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input {
    float data[];
} input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight {
    uint data[];
} weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias {
    float data[];
} bias_buffer;
layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint input_channels;
    uint output_channels;
    uint kernel;
    uint batches;
    uint spatial_offset;
    uint spatial_count;
} parameters;

float read_weight(uint index) {
    const vec2 values =
        unpackHalf2x16(weight_buffer.data[index >> 1]);
    return (index & 1) == 0 ? values.x : values.y;
}

const uint inner_tile = 32;
const uint inner_stride = inner_tile + 1;
shared float input_tile[32 * inner_stride];
shared float weight_tile[32 * inner_stride];

void main() {
    const uint kernel_area = parameters.kernel * parameters.kernel;
    const uint batch = gl_GlobalInvocationID.z / kernel_area;
    const uint kernel_index = gl_GlobalInvocationID.z % kernel_area;
    const uint kernel_x = kernel_index % parameters.kernel;
    const uint kernel_y = kernel_index / parameters.kernel;
    const uint local_spatial_base =
        gl_WorkGroupID.x * 32 + gl_LocalInvocationID.x * 4;
    const uint output_channel_base =
        gl_WorkGroupID.y * 32 + gl_LocalInvocationID.y * 4;
    const uint input_spatial_count =
        parameters.input_width * parameters.input_height;
    const uint output_width =
        parameters.input_width * parameters.kernel;
    const uint output_height =
        parameters.input_height * parameters.kernel;
    const uint output_spatial_count = output_width * output_height;
    const uint lane =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    float sums[4][4];
    for (uint output_offset = 0; output_offset < 4; ++output_offset) {
        for (uint spatial_offset = 0;
             spatial_offset < 4;
             ++spatial_offset) {
            sums[output_offset][spatial_offset] = 0.0;
        }
    }
    for (uint inner_base = 0;
         inner_base < parameters.input_channels;
         inner_base += inner_tile) {
        for (uint index = lane;
             index < 32 * inner_tile;
             index += 64) {
            const uint spatial_offset = index / inner_tile;
            const uint inner = inner_base + index % inner_tile;
            const uint local_spatial =
                gl_WorkGroupID.x * 32 + spatial_offset;
            const uint spatial =
                parameters.spatial_offset + local_spatial;
            input_tile[
                spatial_offset * inner_stride + index % inner_tile] =
                batch < parameters.batches &&
                local_spatial < parameters.spatial_count &&
                spatial < input_spatial_count &&
                inner < parameters.input_channels
                ? input_buffer.data[
                    (batch * parameters.input_channels + inner) *
                        input_spatial_count +
                    spatial]
                : 0.0;
        }
        for (uint index = lane;
             index < 32 * inner_tile;
             index += 64) {
            const uint output_offset = index / inner_tile;
            const uint inner = inner_base + index % inner_tile;
            const uint output_channel =
                gl_WorkGroupID.y * 32 + output_offset;
            weight_tile[
                output_offset * inner_stride + index % inner_tile] =
                output_channel < parameters.output_channels &&
                inner < parameters.input_channels
                ? read_weight(
                    ((inner * parameters.output_channels +
                        output_channel) * parameters.kernel +
                        kernel_y) * parameters.kernel +
                    kernel_x)
                : 0.0;
        }
        barrier();
        const uint inner_count =
            min(inner_tile, parameters.input_channels - inner_base);
        for (uint inner = 0; inner < inner_count; ++inner) {
            float inputs[4];
            float weights[4];
            for (uint spatial_offset = 0;
                 spatial_offset < 4;
                 ++spatial_offset) {
                inputs[spatial_offset] = input_tile[
                    (gl_LocalInvocationID.x * 4 + spatial_offset) *
                        inner_stride +
                    inner];
            }
            for (uint output_offset = 0;
                 output_offset < 4;
                 ++output_offset) {
                weights[output_offset] = weight_tile[
                    (gl_LocalInvocationID.y * 4 + output_offset) *
                        inner_stride +
                    inner];
            }
            for (uint output_offset = 0;
                 output_offset < 4;
                 ++output_offset) {
                for (uint spatial_offset = 0;
                     spatial_offset < 4;
                     ++spatial_offset) {
                    sums[output_offset][spatial_offset] +=
                        weights[output_offset] *
                        inputs[spatial_offset];
                }
            }
        }
        barrier();
    }
    if (batch >= parameters.batches) return;
    for (uint output_offset = 0;
         output_offset < 4;
         ++output_offset) {
        const uint output_channel =
            output_channel_base + output_offset;
        if (output_channel >= parameters.output_channels) continue;
        const float bias = bias_buffer.data[output_channel];
        for (uint spatial_offset = 0;
             spatial_offset < 4;
             ++spatial_offset) {
            const uint local_spatial =
                local_spatial_base + spatial_offset;
            if (local_spatial >= parameters.spatial_count) continue;
            const uint input_spatial =
                parameters.spatial_offset + local_spatial;
            if (input_spatial >= input_spatial_count) continue;
            const uint input_x =
                input_spatial % parameters.input_width;
            const uint input_y =
                input_spatial / parameters.input_width;
            const uint output_x =
                input_x * parameters.kernel + kernel_x;
            const uint output_y =
                input_y * parameters.kernel + kernel_y;
            output_buffer.data[
                (batch * parameters.output_channels +
                    output_channel) * output_spatial_count +
                output_y * output_width + output_x] =
                sums[output_offset][spatial_offset] + bias;
        }
    }
}
