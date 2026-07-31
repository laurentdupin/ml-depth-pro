#version 450

layout(local_size_x = 8, local_size_y = 8) in;
layout(binding = 0) uniform sampler2D source_texture;
layout(std430, binding = 1) writeonly buffer Destination {
    float values[];
} output_data;
layout(push_constant) uniform Parameters {
    uint source_width, source_height;
} p;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    if (x >= 1536u || y >= 1536u) return;
    precise float raw_x =
        (float(x) + 0.5) * float(p.source_width) / 1536.0 - 0.5;
    precise float raw_y =
        (float(y) + 0.5) * float(p.source_height) / 1536.0 - 0.5;
    precise float sx = clamp(raw_x, 0.0, float(p.source_width - 1u));
    precise float sy = clamp(raw_y, 0.0, float(p.source_height - 1u));
    const int x0 = int(sx);
    const int y0 = int(sy);
    const int x1 = min(x0 + 1, int(p.source_width) - 1);
    const int y1 = min(y0 + 1, int(p.source_height) - 1);
    // The host contract consumes the first three BGRA bytes as B,G,R planes.
    // Normalize before interpolation to preserve the host preprocessing order.
    precise vec3 p00 =
        texelFetch(source_texture, ivec2(x0, y0), 0).bgr * 2.0 - 1.0;
    precise vec3 p01 =
        texelFetch(source_texture, ivec2(x1, y0), 0).bgr * 2.0 - 1.0;
    precise vec3 p10 =
        texelFetch(source_texture, ivec2(x0, y1), 0).bgr * 2.0 - 1.0;
    precise vec3 p11 =
        texelFetch(source_texture, ivec2(x1, y1), 0).bgr * 2.0 - 1.0;
    precise vec3 top = p00 * (1.0 - fract(sx)) + p01 * fract(sx);
    precise vec3 bottom = p10 * (1.0 - fract(sx)) + p11 * fract(sx);
    precise vec3 value =
        top * (1.0 - fract(sy)) + bottom * fract(sy);
    const uint plane = 1536u * 1536u;
    const uint index = y * 1536u + x;
    output_data.values[index] = value.r;
    output_data.values[plane + index] = value.g;
    output_data.values[2u * plane + index] = value.b;
}
