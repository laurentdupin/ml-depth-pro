#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(std430, binding = 0) writeonly buffer Output { float values[]; } output_data;
layout(std430, binding = 1) readonly buffer X0 { float values[]; } x0_data;
layout(std430, binding = 2) readonly buffer X1 { float values[]; } x1_data;
layout(std430, binding = 3) readonly buffer X2 { float values[]; } x2_data;

void main() {
    const uint x = gl_GlobalInvocationID.x;
    const uint y = gl_GlobalInvocationID.y;
    const uint patch_index = gl_GlobalInvocationID.z;
    if (x >= 384u || y >= 384u || patch_index >= 35u) return;
    uint source_width;
    uint source_height;
    uint left;
    uint top;
    if (patch_index < 25u) {
        source_width = 1536u;
        source_height = 1536u;
        left = (patch_index % 5u) * 288u;
        top = (patch_index / 5u) * 288u;
    } else if (patch_index < 34u) {
        const uint local_patch = patch_index - 25u;
        source_width = 768u;
        source_height = 768u;
        left = (local_patch % 3u) * 192u;
        top = (local_patch / 3u) * 192u;
    } else {
        source_width = 384u;
        source_height = 384u;
        left = 0u;
        top = 0u;
    }
    const uint source_plane = source_width * source_height;
    const uint destination_plane = 384u * 384u;
    const uint source_pixel = (top + y) * source_width + left + x;
    const uint destination_pixel = y * 384u + x;
    for (uint channel = 0u; channel < 3u; ++channel) {
        const uint destination =
            patch_index * 3u * destination_plane + channel * destination_plane +
            destination_pixel;
        if (patch_index < 25u)
            output_data.values[destination] =
                x0_data.values[channel * source_plane + source_pixel];
        else if (patch_index < 34u)
            output_data.values[destination] =
                x1_data.values[channel * source_plane + source_pixel];
        else
            output_data.values[destination] =
                x2_data.values[channel * source_plane + source_pixel];
    }
}
