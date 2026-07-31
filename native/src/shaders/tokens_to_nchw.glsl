#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float v[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input { float v[]; } input_buffer;
layout(push_constant) uniform Parameters {
    uint channels;
    uint batch_index;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    const uint count = parameters.channels * 576;
    if (index >= count) {
        return;
    }
    const uint channel = index / 576;
    const uint patch_id = index % 576;
    output_buffer.v[index] =
        input_buffer.v[
            (parameters.batch_index * 577 + patch_id + 1) *
                parameters.channels +
            channel];
}
