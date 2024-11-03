# Detect the architecture we're building for
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(ARCH "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    set(ARCH "arm64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le|powerpc64le")
    set(ARCH "ppc64le")
else()
    message(STATUS "Unknown architecture, defaulting to x86_64")
    set(ARCH "x86_64")
endif()

# Optionally, handle CUDA separately if detected
find_package(CUDA QUIET)
if(CUDA_FOUND)
    set(ARCH "cuda")
    message(STATUS "CUDA detected, compiling for NVIDIA GPU")
endif()
