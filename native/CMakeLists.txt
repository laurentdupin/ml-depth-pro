cmake_minimum_required(VERSION 3.21)
project(depth_pro_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_library(depth_pro_core STATIC
    src/model.cpp
    src/encoder_cpu.cpp
    src/graph_cpu.cpp)
target_include_directories(depth_pro_core PUBLIC src)

add_library(depth_pro_model SHARED src/depth_pro.cpp)
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
