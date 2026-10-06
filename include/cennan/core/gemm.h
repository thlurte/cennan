#ifndef CENNAN_CORE_GEMM_H
#define CENNAN_CORE_GEMM_H

#include <cstddef>
#include <span>

namespace cennan {

/// Scalar reference implementation for Matrix-Vector Multiplication: y = A * x
/// A: [M, K] row-major matrix
/// x: [K] input vector
/// y: [M] output vector
void gemv_f32_scalar(const float *A, const float *x, float *y, size_t M, size_t K);

/// Vectorized AVX2+FMA Matrix-Vector Multiplication: y = A * x
/// 4-way unrolled reduction across dimension K using AVX2 FMA instructions.
/// A: [M, K] row-major matrix
/// x: [K] input vector
/// y: [M] output vector
void gemv_f32_avx2(const float *A, const float *x, float *y, size_t M, size_t K);

/// Default GEMV entry point
inline void gemv_f32(const float *A, const float *x, float *y, size_t M, size_t K) {
  gemv_f32_avx2(A, x, y, M, K);
}

/// Vectorized AVX2+FMA Dot Product between two vectors of length K
float dot_f32_avx2(const float *a, const float *b, size_t K);

/// Scalar reference Matrix-Matrix Multiplication: C = A * B
/// A: [M, K] row-major matrix
/// B: [K, N] row-major matrix
/// C: [M, N] row-major output matrix
void gemm_f32_scalar(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);

/// 2D Register-Blocked AVX2 Matrix-Matrix Multiplication: C = A * B
/// 4x16 register blocking using 8 YMM accumulators.
/// A: [M, K] row-major matrix
/// B: [K, N] row-major matrix
/// C: [M, N] row-major output matrix
void gemm_f32_avx2(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);

/// Default GEMM entry point
inline void gemm_f32(const float *A, const float *B, float *C, size_t M, size_t N, size_t K) {
  gemm_f32_avx2(A, B, C, M, N, K);
}

} // namespace cennan

#endif // CENNAN_CORE_GEMM_H
