#include "cennan/core/gemm.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define REQUIRE(cond) \
  do { \
    if (!(cond)) { \
      std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__ << " (" #cond ")" << std::endl; \
      std::abort(); \
    } \
  } while (0)

void test_dot_product() {
  std::cout << "--- Testing AVX2 Dot Product Kernel ---" << std::endl;

  // Basic hand-calculated test
  float a[] = {1.0f, 2.0f, 3.0f};
  float b[] = {4.0f, 5.0f, 6.0f};
  float res = cennan::dot_f32_avx2(a, b, 3);
  // 1*4 + 2*5 + 3*6 = 32
  REQUIRE(std::abs(res - 32.0f) < 1e-5f);

  // Varied sizes from small tails to large vectors
  for (size_t K : {1, 3, 7, 8, 15, 16, 19, 31, 32, 45, 64, 128, 512, 768, 1536, 4096}) {
    std::vector<float> u(K), v(K);
    float scalar_expected = 0.0f;
    for (size_t i = 0; i < K; ++i) {
      u[i] = std::sin(static_cast<float>(i + 1) * 0.3f);
      v[i] = std::cos(static_cast<float>(i + 2) * 0.7f);
      scalar_expected += u[i] * v[i];
    }
    float avx2_res = cennan::dot_f32_avx2(u.data(), v.data(), K);
    float diff = std::abs(avx2_res - scalar_expected);
    float tol = std::max(1e-4f, std::abs(scalar_expected) * 1e-4f);
    REQUIRE(diff < tol);
  }

  std::cout << "✓ AVX2 Dot Product Kernel Passed" << std::endl;
}

void test_gemv_identity() {
  std::cout << "--- Testing GEMV Identity Matrix ---" << std::endl;

  const size_t N = 64;
  std::vector<float> I(N * N, 0.0f);
  for (size_t i = 0; i < N; ++i) {
    I[i * N + i] = 1.0f;
  }

  std::vector<float> x(N);
  for (size_t i = 0; i < N; ++i) {
    x[i] = static_cast<float>(i + 1);
  }

  std::vector<float> y(N, 0.0f);
  cennan::gemv_f32(I.data(), x.data(), y.data(), N, N);

  for (size_t i = 0; i < N; ++i) {
    REQUIRE(std::abs(y[i] - x[i]) < 1e-5f);
  }

  std::cout << "✓ GEMV Identity Matrix Passed" << std::endl;
}

void test_gemv_avx2_shapes() {
  std::cout << "--- Testing GEMV across diverse matrix shapes ---" << std::endl;

  struct Shape {
    size_t M;
    size_t K;
  };

  std::vector<Shape> shapes = {
    {1, 128},
    {128, 1},
    {17, 33},    // Odd tails
    {64, 64},
    {128, 768},  // Transformer hidden projection
    {768, 768},
    {256, 1024},
    {512, 512}
  };

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  for (const auto &[M, K] : shapes) {
    std::vector<float> A(M * K);
    std::vector<float> x(K);
    std::vector<float> y_scalar(M, 0.0f);
    std::vector<float> y_avx2(M, 0.0f);

    for (float &val : A) val = dist(rng);
    for (float &val : x) val = dist(rng);

    cennan::gemv_f32_scalar(A.data(), x.data(), y_scalar.data(), M, K);
    cennan::gemv_f32_avx2(A.data(), x.data(), y_avx2.data(), M, K);

    for (size_t m = 0; m < M; ++m) {
      float diff = std::abs(y_avx2[m] - y_scalar[m]);
      float tol = std::max(1e-4f, std::abs(y_scalar[m]) * 1e-4f);
      REQUIRE(diff < tol);
    }
  }

  std::cout << "✓ GEMV Shape Parity Passed" << std::endl;
}

void test_gemm_basic_and_identity() {
  std::cout << "--- Testing GEMM Hand Calculation & Identity ---" << std::endl;

  // Hand-calculated 2x2
  // A = [1 2; 3 4], B = [5 6; 7 8]
  // C = [19 22; 43 50]
  float A[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  float B[4] = {5.0f, 6.0f, 7.0f, 8.0f};
  float C[4] = {0.0f};

  cennan::gemm_f32(A, B, C, 2, 2, 2);
  REQUIRE(std::abs(C[0] - 19.0f) < 1e-5f);
  REQUIRE(std::abs(C[1] - 22.0f) < 1e-5f);
  REQUIRE(std::abs(C[2] - 43.0f) < 1e-5f);
  REQUIRE(std::abs(C[3] - 50.0f) < 1e-5f);

  // Identity test on 32x32
  const size_t N = 32;
  std::vector<float> Ident(N * N, 0.0f);
  for (size_t i = 0; i < N; ++i) Ident[i * N + i] = 1.0f;

  std::vector<float> Mat(N * N);
  for (size_t i = 0; i < N * N; ++i) Mat[i] = static_cast<float>(i + 1);

  std::vector<float> Out(N * N, 0.0f);
  cennan::gemm_f32(Mat.data(), Ident.data(), Out.data(), N, N, N);

  for (size_t i = 0; i < N * N; ++i) {
    REQUIRE(std::abs(Out[i] - Mat[i]) < 1e-4f);
  }

  std::cout << "✓ GEMM Hand Calculation & Identity Passed" << std::endl;
}

void test_gemm_2d_blocked_shapes() {
  std::cout << "--- Testing 2D Register-Blocked GEMM (4x16 unrolled) Shapes ---" << std::endl;

  struct GemmShape {
    size_t M;
    size_t N;
    size_t K;
  };

  std::vector<GemmShape> shapes = {
    {4, 16, 32},    // Exact 1 register tile
    {8, 32, 64},    // 2x2 register tiles
    {7, 23, 19},    // Uneven odd dimensions exercising all residual paths
    {1, 64, 128},   // Single row (vector-matrix projection)
    {32, 1, 128},   // Single column
    {16, 768, 768}, // Transformer sequence token projection
    {64, 3072, 768} // MLP Feed-Forward up-projection (64 tokens, d_model=768 -> d_ff=3072)
  };

  std::mt19937 rng(1337);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  for (const auto &[M, N, K] : shapes) {
    std::vector<float> A(M * K);
    std::vector<float> B(K * N);
    std::vector<float> C_scalar(M * N, 0.0f);
    std::vector<float> C_avx2(M * N, 0.0f);

    for (float &val : A) val = dist(rng);
    for (float &val : B) val = dist(rng);

    cennan::gemm_f32_scalar(A.data(), B.data(), C_scalar.data(), M, N, K);
    cennan::gemm_f32_avx2(A.data(), B.data(), C_avx2.data(), M, N, K);

    for (size_t i = 0; i < M * N; ++i) {
      float diff = std::abs(C_avx2[i] - C_scalar[i]);
      float tol = std::max(1e-3f, std::abs(C_scalar[i]) * 1e-3f);
      REQUIRE(diff < tol);
    }
  }

  std::cout << "✓ 2D Register-Blocked GEMM Parity Passed" << std::endl;
}

int main() {
  test_dot_product();
  test_gemv_identity();
  test_gemv_avx2_shapes();
  test_gemm_basic_and_identity();
  test_gemm_2d_blocked_shapes();
  std::cout << "All GEMM/GEMV tests passed successfully!" << std::endl;
  return 0;
}
