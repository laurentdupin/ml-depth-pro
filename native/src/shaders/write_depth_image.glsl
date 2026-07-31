#version 450

layout(local_size_x = 8, local_size_y = 8) in;
layout(binding = 0, r32f) uniform writeonly image2D output_image;
layout(std430, binding = 1) readonly buffer Depth { float values[]; } depth_data;
layout(push_constant) uniform Parameters { uint width, height; } p;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    if (x < p.width && y < p.height)
        imageStore(output_image, ivec2(x, y),
                   vec4(depth_data.values[y * p.width + x]));
}
