#version 450 core

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float v[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer First { float v[]; } first_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Second { float v[]; } second_buffer;
layout(push_constant) uniform Parameters {
    uint first_count;
    uint second_count;
} parameters;

void main() {
    const uint index = gl_GlobalInvocationID.x;
    if (index < parameters.first_count) {
        output_buffer.v[index] = first_buffer.v[index];
    } else if (index < parameters.first_count + parameters.second_count) {
        output_buffer.v[index] =
            second_buffer.v[index - parameters.first_count];
    }
}
