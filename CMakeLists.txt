cmake_minimum_required(VERSION 3.14)
project(diffuse-cpp VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(DIFFUSE_BUILD_TESTS   "Build tests"      ON)
option(DIFFUSE_BUILD_TOOLS   "Build CLI tools"   ON)

# ── GGML submodule ──────────────────────────────────────────────
set(GGML_BUILD_TESTS  OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP ON CACHE BOOL "" FORCE)
add_subdirectory(ggml)

# ── Core library (diffusion + autoregressive) ─────────────────
add_library(diffuse
    src/diffuse-model.cpp
    src/diffuse-graph.cpp
    src/diffuse-sampler.cpp
    src/ar-graph.cpp
    src/ar-sampler.cpp
)
target_include_directories(diffuse PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(diffuse PUBLIC ggml ggml-cpu)

# ── CLI tools ───────────────────────────────────────────────────
if(DIFFUSE_BUILD_TOOLS)
    # Diffusion CLI (LLaDA, Dream)
    add_executable(diffuse-cli tools/main-cli.cpp)
    target_link_libraries(diffuse-cli PRIVATE diffuse)

    # Autoregressive CLI (Qwen2.5, Llama, etc.)
    add_executable(diffuse-ar tools/ar-cli.cpp)
    target_link_libraries(diffuse-ar PRIVATE diffuse)

    add_executable(diffuse-bench tools/main-bench.cpp)
    target_link_libraries(diffuse-bench PRIVATE diffuse)

    add_executable(dump-logits tools/dump-logits.cpp)
    target_link_libraries(dump-logits PRIVATE diffuse)

    add_executable(diffuse-quantize tools/quantize.cpp)
    target_link_libraries(diffuse-quantize PRIVATE ggml)
endif()

# ── Tests ───────────────────────────────────────────────────────
if(DIFFUSE_BUILD_TESTS)
    enable_testing()
    add_executable(test-forward tests/test-forward.cpp)
    target_link_libraries(test-forward PRIVATE diffuse)
    add_test(NAME test-forward COMMAND test-forward)
endif()
