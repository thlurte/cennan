# cennan command runner

default:
    @just --list

# Configure CMake build directory
config build_type="Release":
    cmake -B build -DCMAKE_BUILD_TYPE={{build_type}}

# Build the library, CLI, and tests
build j="$(nproc)":
    cmake --build build -j{{j}}

# Run unit tests via ctest
test:
    ctest --test-dir build --output-on-failure

# Build and run the entire test suite in one command
check: build test

# Run the CLI
run *args:
    ./build/cennan {{args}}

# Clean build artifacts
clean:
    rm -rf build
