#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float v[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input { float v[]; } input_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Fov { float v[]; } fov_buffer;
layout(push_constant) uniform Parameters {
    uint count;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index >= parameters.count) {
        return;
    }
    const float radians =
        fov_buffer.v[0] * 0.00872664625997164788462;
    const float scale = 2.0 * tan(radians);
    output_buffer.v[index] = input_buffer.v[index] * scale;
}
