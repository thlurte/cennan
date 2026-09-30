#include <iostream>
#include "cennan/core/tensor.h"

int main() {
    std::cout << "cennan: C++ High-Dimensional Embedding Engine v0.1.0\n";
    cennan::Tensor t({2, 4});
    std::cout << "Allocated test tensor with shape [2, 4], numel=" << t.numel() << "\n";
    return 0;
}
