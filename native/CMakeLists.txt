cmake_minimum_required(VERSION 3.21)
project(depth_pro_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY
        "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

add_library(depth_pro_model STATIC
    src/model.cpp
    src/encoder_cpu.cpp)
target_include_directories(depth_pro_model PUBLIC src)

add_executable(depth_pro_model_probe tools/model_probe.cpp)
target_link_libraries(
    depth_pro_model_probe PRIVATE depth_pro_model)

add_executable(depth_pro_encoder_probe tools/encoder_probe.cpp)
target_link_libraries(
    depth_pro_encoder_probe PRIVATE depth_pro_model)

if(MSVC)
    target_compile_options(
        depth_pro_model PRIVATE /W4 /permissive-)
    target_compile_options(
        depth_pro_model_probe PRIVATE /W4 /permissive-)
    target_compile_options(
        depth_pro_encoder_probe PRIVATE /W4 /permissive-)
else()
    target_compile_options(
        depth_pro_model PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(
        depth_pro_model_probe PRIVATE -Wall -Wextra -Wpedantic)
    target_compile_options(
        depth_pro_encoder_probe PRIVATE -Wall -Wextra -Wpedantic)
endif()
