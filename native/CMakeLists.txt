cmake_minimum_required(VERSION 3.21)
project(depth_pro_native LANGUAGES C CXX)
include("${CMAKE_CURRENT_LIST_DIR}/../../native_support/InferBridgeNativeHarnessSupport.cmake")
link_libraries(InferBridge::NativeHarnessSupport)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(DEPTH_PRO_WITH_VULKAN
    "Build the dependency-free Vulkan executor" ON)
option(DEPTH_PRO_WITH_METAL
    "Build the native Metal executor" ${APPLE})

if(DEPTH_PRO_WITH_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "DEPTH_PRO_WITH_METAL requires Apple platforms")
    endif()
    enable_language(OBJCXX)
endif()

if(DEPTH_PRO_WITH_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)
    set(DEPTH_PRO_GENERATED_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${DEPTH_PRO_GENERATED_DIR}")

    function(depth_pro_compile_shader NAME)
        set(SOURCE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${NAME}.glsl")
        set(SPIRV "${DEPTH_PRO_GENERATED_DIR}/${NAME}.spv")
        set(HEADER "${DEPTH_PRO_GENERATED_DIR}/${NAME}_spv.h")
        add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND Vulkan::glslc -fshader-stage=compute
                --target-env=vulkan1.3 --target-spv=spv1.6 -Werror
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
                -o "${SPIRV}" "${SOURCE}"
            DEPENDS
                "${SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/indexing.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_pointwise_gemm_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_tiled_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_vec4_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/project_tokens_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv_transpose_nonoverlap_common.glsl"
            VERBATIM)
        add_custom_command(
            OUTPUT "${HEADER}"
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${SPIRV}"
                "-DOUTPUT=${HEADER}"
                "-DVARIABLE=dpro_${NAME}_spv"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            DEPENDS "${SPIRV}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            VERBATIM)
        set(DEPTH_PRO_SHADER_HEADERS
            ${DEPTH_PRO_SHADER_HEADERS} "${HEADER}" PARENT_SCOPE)
    endfunction()

    foreach(SHADER
            add_scaled add add_position bilinear_align_true
            bilinear_align_true_image bmm bmm_score_half
            bmm_value_half conv2d conv2d8 conv2d_half conv2d8_half
            conv2d_pointwise_gemm conv2d_pointwise_gemm_half
            conv2d8_tiled conv2d8_tiled_half conv2d8_tiled16x8_half
            conv_transpose_nonoverlap conv_transpose_nonoverlap_half
            conv_transpose_gemm_half
            gelu layer_norm linear linear16 linear_half linear16_half
            linear_vec4 linear_vec4_half linear_vec8 linear_vec8_half
            linear_vec16 linear_vec16_half
            prepare_tokens position_bicubic project_tokens
            project_tokens_half relu softmax_lastdim
            softmax_lastdim_half prepare_tokens16 tokens_to_nchw
            merge_patch concatenate bilinear_half_pixel
            scale_inverse reciprocal_depth preprocess_texture
            pyramid_patches write_depth_image)
        depth_pro_compile_shader(${SHADER})
    endforeach()
    foreach(SHARED_SHADER quantize_rows_int8 linear_int8_tiled
            layer_norm_parallel)
        inferbridge_compile_native_support_shader(
            NAME "${SHARED_SHADER}" PREFIX dpro
            GENERATED_DIR "${DEPTH_PRO_GENERATED_DIR}"
            EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            OUT_VAR DEPTH_PRO_SHADER_HEADERS)
    endforeach()
endif()

add_library(depth_pro_core STATIC
    src/model.cpp
    src/encoder_cpu.cpp
    src/graph_cpu.cpp)
set_target_properties(depth_pro_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(depth_pro_core PUBLIC src)

add_library(depth_pro_model SHARED
    src/depth_pro.cpp
    src/inferbridge_harness.cpp)
if(DEPTH_PRO_WITH_METAL)
    target_sources(depth_pro_model PRIVATE src/metal_executor.mm)
    target_link_libraries(depth_pro_model PRIVATE
        "-framework Foundation"
        "-framework Metal"
        "-framework MetalPerformanceShadersGraph")
    target_compile_definitions(
        depth_pro_model PRIVATE DEPTH_PRO_WITH_METAL=1)
    set_source_files_properties(src/metal_executor.mm PROPERTIES
        COMPILE_OPTIONS "-fobjc-arc")
endif()
if(DEPTH_PRO_WITH_VULKAN)
    target_sources(depth_pro_model PRIVATE
        src/vulkan.cpp
        src/operators.cpp
        src/gpu_model.cpp
        src/encoder_gpu.cpp
        src/graph_gpu.cpp
        src/gpu_io.cpp
        src/external_gpu.cpp
        ${DEPTH_PRO_SHADER_HEADERS})
    target_include_directories(
        depth_pro_model PRIVATE "${DEPTH_PRO_GENERATED_DIR}")
    target_link_libraries(depth_pro_model PRIVATE Vulkan::Vulkan)
    if(WIN32)
        target_link_libraries(depth_pro_model PRIVATE d3d12 dxgi)
    endif()
    target_compile_definitions(
        depth_pro_model PRIVATE DEPTH_PRO_WITH_VULKAN=1)
endif()
target_include_directories(
    depth_pro_model PUBLIC include PRIVATE src)
target_link_libraries(depth_pro_model PRIVATE depth_pro_core)
target_compile_definitions(
    depth_pro_model PRIVATE
        DEPTH_PRO_BUILD_DLL
        INFERBRIDGE_HARNESS_BUILDING_LIBRARY)
set_target_properties(
    depth_pro_model PROPERTIES OUTPUT_NAME depth_pro_native)

add_executable(depth_pro_model_probe tools/model_probe.cpp)
target_link_libraries(
    depth_pro_model_probe PRIVATE depth_pro_core)

add_executable(depth_pro_encoder_probe tools/encoder_probe.cpp)
target_link_libraries(
    depth_pro_encoder_probe PRIVATE depth_pro_core)

add_executable(
    depth_pro_full_graph_probe tools/full_graph_probe.cpp)
target_link_libraries(
    depth_pro_full_graph_probe PRIVATE depth_pro_core)

add_executable(depth_pro_inference_dump tools/inference_dump.cpp)
target_link_libraries(depth_pro_inference_dump PRIVATE depth_pro_model)
target_include_directories(depth_pro_inference_dump PRIVATE include)

include(CTest)
if(BUILD_TESTING)
    add_executable(depth_pro_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(
        depth_pro_c_abi_smoke PRIVATE depth_pro_model)
        add_test(
        NAME depth_pro_c_abi_smoke
        COMMAND depth_pro_c_abi_smoke)
    add_executable(
        depth_pro_harness_abi_smoke tests/harness_abi_smoke.c)
    target_link_libraries(
        depth_pro_harness_abi_smoke PRIVATE depth_pro_model)
    target_include_directories(
        depth_pro_harness_abi_smoke PRIVATE include)
    add_test(
        NAME depth_pro_harness_abi_smoke
        COMMAND depth_pro_harness_abi_smoke)
    add_executable(
        depth_pro_harness_full_graph tests/harness_full_graph.cpp)
    target_link_libraries(
        depth_pro_harness_full_graph PRIVATE depth_pro_model)
    target_include_directories(
        depth_pro_harness_full_graph PRIVATE include)
    add_test(
        NAME depth_pro_harness_full_graph
        COMMAND depth_pro_harness_full_graph)
    set_tests_properties(
        depth_pro_harness_full_graph PROPERTIES SKIP_RETURN_CODE 77)
    if(WIN32 AND DEPTH_PRO_WITH_VULKAN)
        add_executable(
            depth_pro_d3d12_full_graph
            tests/d3d12_full_graph.cpp
            src/vulkan.cpp
            src/operators.cpp
            src/gpu_model.cpp
            src/encoder_gpu.cpp
            src/graph_gpu.cpp
            src/gpu_io.cpp
            ${DEPTH_PRO_SHADER_HEADERS})
        target_link_libraries(
            depth_pro_d3d12_full_graph PRIVATE
            depth_pro_model depth_pro_core Vulkan::Vulkan d3d12 dxgi)
        target_include_directories(
            depth_pro_d3d12_full_graph PRIVATE
            include src "${DEPTH_PRO_GENERATED_DIR}")
        target_compile_options(
            depth_pro_d3d12_full_graph PRIVATE /W4 /permissive-)
        add_test(
            NAME depth_pro_d3d12_full_graph
            COMMAND depth_pro_d3d12_full_graph)
        set_tests_properties(
            depth_pro_d3d12_full_graph PROPERTIES SKIP_RETURN_CODE 77)
    endif()
    if(DEPTH_PRO_WITH_VULKAN)
        add_executable(
            depth_pro_gpu_full_graph_probe
            tools/gpu_full_graph_probe.cpp)
        target_link_libraries(
            depth_pro_gpu_full_graph_probe PRIVATE depth_pro_model)
        target_include_directories(
            depth_pro_gpu_full_graph_probe PRIVATE include)
        add_executable(
            depth_pro_gpu_encoder_probe
            tools/gpu_encoder_probe.cpp
            src/model.cpp src/vulkan.cpp src/operators.cpp
            src/gpu_model.cpp src/encoder_gpu.cpp
            ${DEPTH_PRO_SHADER_HEADERS})
        target_include_directories(
            depth_pro_gpu_encoder_probe PRIVATE
            include src "${DEPTH_PRO_GENERATED_DIR}")
        target_link_libraries(
            depth_pro_gpu_encoder_probe PRIVATE Vulkan::Vulkan)
    endif()
endif()

if(MSVC)
    target_compile_options(
        depth_pro_core PRIVATE /W4 /permissive-)
    target_compile_options(
        depth_pro_model PRIVATE /W4 /permissive-)
    target_compile_options(
        depth_pro_model_probe PRIVATE /W4 /permissive-)
    target_compile_options(
        depth_pro_encoder_probe PRIVATE /W4 /permissive-)
    target_compile_options(
        depth_pro_full_graph_probe PRIVATE /W4 /permissive-)
else()
    target_compile_options(
        depth_pro_core PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(
        depth_pro_model PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(
        depth_pro_model_probe PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(
        depth_pro_encoder_probe PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(
        depth_pro_full_graph_probe PRIVATE
        -Wall -Wextra -Wpedantic)
endif()
