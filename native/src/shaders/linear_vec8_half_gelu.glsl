#version 450 core

#define K_VECTORS 8
#define HALF_WEIGHT
#define FC1_GELU_EPILOGUE
#include "linear_vec4_common.glsl"
