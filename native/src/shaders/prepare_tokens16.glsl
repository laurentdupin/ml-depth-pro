#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float v[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Image { float v[]; } image_buffer;
layout(set = 0, binding = 2, std430) readonly buffer Weight { float v[]; } weight_buffer;
layout(set = 0, binding = 3, std430) readonly buffer Bias { float v[]; } bias_buffer;
layout(set = 0, binding = 4, std430) readonly buffer ClassToken { float v[]; } class_buffer;
layout(set = 0, binding = 5, std430) readonly buffer Position { float v[]; } position_buffer;

void main() {
    const uint feature = gl_GlobalInvocationID.x;
    const uint token = gl_GlobalInvocationID.y;
    if (feature >= 1024 || token >= 577) {
        return;
    }
    float value;
    if (token == 0) {
        value = class_buffer.v[feature];
    } else {
        const uint patch_id = token - 1;
        const uint px = patch_id % 24;
        const uint py = patch_id / 24;
        value = bias_buffer.v[feature];
        for (uint channel = 0; channel < 3; ++channel) {
            for (uint ky = 0; ky < 16; ++ky) {
                for (uint kx = 0; kx < 16; ++kx) {
                    const uint image_index =
                        (channel * 384 + py * 16 + ky) * 384 +
                        px * 16 + kx;
                    const uint weight_index =
                        ((feature * 3 + channel) * 16 + ky) * 16 + kx;
                    value += image_buffer.v[image_index] *
                        weight_buffer.v[weight_index];
                }
            }
        }
    }
    const uint index = token * 1024 + feature;
    output_buffer.v[index] = value + position_buffer.v[index];
}
