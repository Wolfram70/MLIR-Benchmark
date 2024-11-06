cmake_minimum_required(VERSION 3.16)

# Project Information
project(MLIRBenchmark LANGUAGES CXX)

# Set C++ standard and architecture flags
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include custom architecture detection
include(cmake/DetectArchitecture.cmake)

# Set profiling options
option(ENABLE_PROFILING "Enable detailed performance profiling" ON)

# Enable verbose output for a flashy effect
set(CMAKE_VERBOSE_MAKEFILE ON)

# Find MLIR and LLVM libraries
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Found LLVM: ${LLVM_DIR}")
message(STATUS "Found MLIR: ${MLIR_DIR}")

# Include directories
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS} ${MLIR_LIBRARY_DIRS})

# Compiler flags for different architectures
if(${ARCH} STREQUAL "x86_64")
    message(STATUS "Building for x86_64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
elseif(${ARCH} STREQUAL "arm64")
    message(STATUS "Building for ARM64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a")
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    # Cross-compile options
    set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
    set(CMAKE_SYSROOT /usr/aarch64-linux-gnu)
elseif(${ARCH} STREQUAL "cuda")
    message(STATUS "Building for NVIDIA GPU")
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    link_directories(${CUDA_LIBRARY_DIRS})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcudart -L${CUDA_LIBRARIES}")
else()
    message(FATAL_ERROR "Unknown architecture detected")
endif()

# Define source files
add_executable(mlir_benchmark
    src/benchmark_main.cpp
)

# Link LLVM and MLIR libraries
llvm_map_components_to_libnames(llvm_libs core support)
target_link_libraries(mlir_benchmark
    ${llvm_libs}
    MLIRAffineDialect
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRSCFDialect
    MLIRMemRefToLLVM
    MLIRArithToLLVM
    MLIRAffineToStandard
    MLIRFuncToLLVM
    MLIRSCFToControlFlow
    MLIRControlFlowToLLVM
    MLIRBuiltinToLLVMIRTranslation
    MLIRExecutionEngine
)

# Enable profiling flags
if(ENABLE_PROFILING)
    message(STATUS "Profiling enabled")
    target_compile_definitions(mlir_benchmark PRIVATE PROFILING_ENABLED)
    target_link_libraries(mlir_benchmark ${LLVM_LIBRARIES})
endif()
