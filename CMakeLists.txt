cmake_minimum_required(VERSION 3.18)
project(flash_mla_cuda LANGUAGES CXX CUDA)

enable_language(C)
enable_language(CXX)
enable_language(CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
   set(CMAKE_CUDA_ARCHITECTURES "80")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0") 
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -w -g") 
endif()

if (CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -w")
endif()

find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
set(PYTHON_VERSION ${Python_VERSION})

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
   OUTPUT_VARIABLE PYTORCH_CMAKE_PREFIX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
message("PYTORCH_CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH}")
list(APPEND CMAKE_PREFIX_PATH ${PYTORCH_CMAKE_PREFIX_PATH})
message("CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
	OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

file(GLOB_RECURSE CSRC_SOURCES
    "csrc/*.cu"
)

file(GLOB_RECURSE CSRC_HEADERS
    "csrc/*.h"
)

add_library(csrc SHARED ${CSRC_SOURCES})
set_property(TARGET csrc PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(csrc
    PUBLIC "csrc/cutlass/include"
    ${CSRC_HEADERS}
)

pybind11_add_module(${PROJECT_NAME} csrc/flash_api.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE 
    csrc
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    "-Wl,-Bsymbolic -Wl,-Bsymbolic-functions")

