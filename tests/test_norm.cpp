#include "cennan/core/norm.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void test_rms_norm() {
  std::cout << "--- Testing RMSNorm SIMD Kernel ---" << std::endl;

  const size_t dim = 8;
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> weight = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> output(dim, 0.0f);

  cennan::rms_norm_f32(input.data(), weight.data(), output.data(), dim, 1e-5f);

  // Verification: compute expected variance
  float sum_sq = 0.0f;
  for (float v : input) sum_sq += v * v;
  float expected_inv_rms = 1.0f / std::sqrt((sum_sq / static_cast<float>(dim)) + 1e-5f);

  for (size_t i = 0; i < dim; ++i) {
    float expected = input[i] * expected_inv_rms;
    assert(std::abs(output[i] - expected) < 1e-5f);
  }

  std::cout << "✓ RMSNorm SIMD Kernel Passed" << std::endl;
}

void test_layer_norm() {
  std::cout << "--- Testing LayerNorm Kernel ---" << std::endl;

  const size_t dim = 4;
  std::vector<float> input = {2.0f, 4.0f, 4.0f, 2.0f};
  std::vector<float> weight = {1.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> bias = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> output(dim, 0.0f);

  cennan::layer_norm_f32(input.data(), weight.data(), bias.data(), output.data(), dim, 1e-5f);

  // Mean = 3.0, Var = 1.0 -> norm = [-1, 1, 1, -1]
  assert(std::abs(output[0] - (-1.0f)) < 1e-4f);
  assert(std::abs(output[1] - (1.0f)) < 1e-4f);

  std::cout << "✓ LayerNorm Kernel Passed" << std::endl;
}

int main() {
  test_rms_norm();
  test_layer_norm();
  std::cout << "All Norm tests passed successfully!" << std::endl;
  return 0;
}
