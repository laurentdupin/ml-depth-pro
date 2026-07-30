cmake_minimum_required(VERSION 3.21)
project(depth_pro_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

option(DEPTH_PRO_WITH_VULKAN
    "Build the dependency-free Vulkan executor" ON)

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
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_common.glsl"
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
            conv_transpose_nonoverlap conv_transpose_nonoverlap_half
            gelu layer_norm linear linear16 linear_half linear16_half
            prepare_tokens position_bicubic project_tokens
            project_tokens_half relu softmax_lastdim
            softmax_lastdim_half)
        depth_pro_compile_shader(${SHADER})
    endforeach()
endif()

add_library(depth_pro_core STATIC
    src/model.cpp
    src/encoder_cpu.cpp
    src/graph_cpu.cpp)
target_include_directories(depth_pro_core PUBLIC src)

add_library(depth_pro_model SHARED src/depth_pro.cpp)
if(DEPTH_PRO_WITH_VULKAN)
    target_sources(depth_pro_model PRIVATE
        src/vulkan.cpp
        src/operators.cpp
        src/gpu_model.cpp
        ${DEPTH_PRO_SHADER_HEADERS})
    target_include_directories(
        depth_pro_model PRIVATE "${DEPTH_PRO_GENERATED_DIR}")
    target_link_libraries(depth_pro_model PRIVATE Vulkan::Vulkan)
    target_compile_definitions(
        depth_pro_model PRIVATE DEPTH_PRO_WITH_VULKAN=1)
endif()
target_include_directories(
    depth_pro_model PUBLIC include PRIVATE src)
target_link_libraries(depth_pro_model PRIVATE depth_pro_core)
target_compile_definitions(
    depth_pro_model PRIVATE DEPTH_PRO_BUILD_DLL)
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

include(CTest)
if(BUILD_TESTING)
    add_executable(depth_pro_c_abi_smoke tests/c_abi_smoke.c)
    target_link_libraries(
        depth_pro_c_abi_smoke PRIVATE depth_pro_model)
    add_test(
        NAME depth_pro_c_abi_smoke
        COMMAND depth_pro_c_abi_smoke)
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
