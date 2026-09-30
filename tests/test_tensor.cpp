#include <iostream>
#include <cassert>
#include "cennan/core/tensor.h"

int main() {
    cennan::Tensor t({4, 16});
    assert(t.dim(0) == 4);
    assert(t.dim(1) == 16);
    assert(t.numel() == 64);
    assert(t.nbytes() == 64 * sizeof(float));
    std::cout << "All cennan tensor tests passed.\n";
    return 0;
}
