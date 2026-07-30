#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) buffer Output { float v[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input { float v[]; } input_buffer;
layout(push_constant) uniform Parameters {
    uint channels;
    uint steps;
    uint padding;
    uint patch_index;
} parameters;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint channel = gl_GlobalInvocationID.z;
    if (x >= 24 || y >= 24 || channel >= parameters.channels) {
        return;
    }
    const uint row = parameters.patch_index / parameters.steps;
    const uint column = parameters.patch_index % parameters.steps;
    const uint top = row == 0 ? 0 : parameters.padding;
    const uint left = column == 0 ? 0 : parameters.padding;
    const uint bottom =
        row + 1 == parameters.steps ? 0 : parameters.padding;
    const uint right =
        column + 1 == parameters.steps ? 0 : parameters.padding;
    if (y < top || y >= 24 - bottom ||
        x < left || x >= 24 - right) {
        return;
    }
    const uint stride = 24 - 2 * parameters.padding;
    const uint side = 24 + (parameters.steps - 1) * stride;
    const uint ox = column * stride + x - left;
    const uint oy = row * stride + y - top;
    output_buffer.v[(channel * side + oy) * side + ox] =
        input_buffer.v[(channel * 24 + y) * 24 + x];
}
