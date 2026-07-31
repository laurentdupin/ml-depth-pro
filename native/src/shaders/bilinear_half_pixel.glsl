#version 450 core

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(set = 0, binding = 0, std430) writeonly buffer Output { float v[]; } output_buffer;
layout(set = 0, binding = 1, std430) readonly buffer Input { float v[]; } input_buffer;
layout(push_constant) uniform Parameters {
    uint input_width;
    uint input_height;
    uint output_width;
    uint output_height;
    uint channels;
} p;

float at(uint c, uint y, uint x) {
    return input_buffer.v[(c * p.input_height + y) * p.input_width + x];
}

void main() {
    const uint ox = gl_GlobalInvocationID.x;
    const uint oy = gl_GlobalInvocationID.y;
    const uint c = gl_GlobalInvocationID.z;
    if (ox >= p.output_width || oy >= p.output_height || c >= p.channels) {
        return;
    }
    precise float raw_x = (float(ox) + 0.5) *
        float(p.input_width) / float(p.output_width) - 0.5;
    precise float raw_y = (float(oy) + 0.5) *
        float(p.input_height) / float(p.output_height) - 0.5;
    precise float sx = clamp(raw_x, 0.0, float(p.input_width - 1));
    precise float sy = clamp(raw_y, 0.0, float(p.input_height - 1));
    const uint x0 = uint(floor(sx));
    const uint y0 = uint(floor(sy));
    const uint x1 = min(x0 + 1, p.input_width - 1);
    const uint y1 = min(y0 + 1, p.input_height - 1);
    const float wx = sx - float(x0);
    const float wy = sy - float(y0);
    precise float top =
        at(c, y0, x0) * (1.0 - wx) + at(c, y0, x1) * wx;
    precise float bottom =
        at(c, y1, x0) * (1.0 - wx) + at(c, y1, x1) * wx;
    precise float value = top * (1.0 - wy) + bottom * wy;
    output_buffer.v[(c * p.output_height + oy) * p.output_width + ox] = value;
}
