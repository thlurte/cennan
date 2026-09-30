cmake_minimum_required(VERSION 3.20)
project(cennan VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler optimization flags
if(MSVC)
  add_compile_options(/O2 /arch:AVX2 /W4)
else()
  add_compile_options(-O3 -march=native -mavx2 -mfma -Wall -Wextra -Wpedantic)
endif()

# Core static library
add_library(cennan_lib
  src/core/tensor.cpp
  src/core/mmap_loader.cpp
)

target_include_directories(cennan_lib PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Executable CLI
add_executable(cennan src/main.cpp)
target_link_libraries(cennan PRIVATE cennan_lib)

# Testing
enable_testing()
add_subdirectory(tests)
