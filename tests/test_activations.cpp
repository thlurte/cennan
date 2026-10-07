#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "cennan/core/activations.h"

int main() {
    std::cout << "--- Running Cennan Activations Test Suite ---\n";

    // 1. GELU In-place test
    std::vector<float> vec = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> expected_gelu = {-0.0454f, -0.1588f, 0.0f, 0.8412f, 1.9545f};

    cennan::gelu_inplace(vec);
    for (size_t i = 0; i < vec.size(); ++i) {
        assert(std::abs(vec[i] - expected_gelu[i]) < 1e-3f);
    }
    std::cout << "✓ gelu_inplace test passed!\n";

    // 2. SwiGLU forward test
    std::vector<float> x = {1.0f, 2.0f};
    std::vector<float> gate = {0.0f, 1.0f};
    std::vector<float> out(2);

    cennan::swiglu_forward(x, gate, out);
    assert(out[0] == 0.0f);
    assert(std::abs(out[1] - 1.462117f) < 1e-4f);
    std::cout << "✓ swiglu_forward test passed!\n";

    std::cout << "All Cennan Activation tests passed successfully!\n";
    return 0;
}
