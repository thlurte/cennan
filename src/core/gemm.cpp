#include "cennan/core/gemm.h"
#include <immintrin.h>
#include <cstring>

namespace cennan {

void gemv_f32_scalar(const float *A, const float *x, float *y, size_t M, size_t K) {
  for (size_t m = 0; m < M; ++m) {
    const float *a_row = A + m * K;
    float sum = 0.0f;
    for (size_t k = 0; k < K; ++k) {
      sum += a_row[k] * x[k];
    }
    y[m] = sum;
  }
}

float dot_f32_avx2(const float *a, const float *b, size_t K) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  size_t k = 0;
  // 32-float unrolled loop (4 x 8-float AVX2 registers)
  for (; k + 32 <= K; k += 32) {
    __m256 a0 = _mm256_loadu_ps(a + k);
    __m256 b0 = _mm256_loadu_ps(b + k);
    acc0 = _mm256_fmadd_ps(a0, b0, acc0);

    __m256 a1 = _mm256_loadu_ps(a + k + 8);
    __m256 b1 = _mm256_loadu_ps(b + k + 8);
    acc1 = _mm256_fmadd_ps(a1, b1, acc1);

    __m256 a2 = _mm256_loadu_ps(a + k + 16);
    __m256 b2 = _mm256_loadu_ps(b + k + 16);
    acc2 = _mm256_fmadd_ps(a2, b2, acc2);

    __m256 a3 = _mm256_loadu_ps(a + k + 24);
    __m256 b3 = _mm256_loadu_ps(b + k + 24);
    acc3 = _mm256_fmadd_ps(a3, b3, acc3);
  }

  // Combine accumulators
  acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));

  // 8-float chunks
  for (; k + 8 <= K; k += 8) {
    __m256 a0 = _mm256_loadu_ps(a + k);
    __m256 b0 = _mm256_loadu_ps(b + k);
    acc0 = _mm256_fmadd_ps(a0, b0, acc0);
  }

  // Horizontal reduction of acc0
  __m128 lo = _mm256_castps256_ps128(acc0);
  __m128 hi = _mm256_extractf128_ps(acc0, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  float sum = _mm_cvtss_f32(s);

  // Scalar tail
  for (; k < K; ++k) {
    sum += a[k] * b[k];
  }

  return sum;
}

void gemv_f32_avx2(const float *A, const float *x, float *y, size_t M, size_t K) {
  for (size_t m = 0; m < M; ++m) {
    const float *a_row = A + m * K;
    y[m] = dot_f32_avx2(a_row, x, K);
  }
}

void gemm_f32_scalar(const float *A, const float *B, float *C, size_t M, size_t N, size_t K) {
  for (size_t m = 0; m < M; ++m) {
    for (size_t n = 0; n < N; ++n) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum;
    }
  }
}

void gemm_f32_avx2(const float *A, const float *B, float *C, size_t M, size_t N, size_t K) {
  if (A == nullptr || B == nullptr || C == nullptr || M == 0 || N == 0 || K == 0) {
    return;
  }

  // Zero-initialize output C
  std::memset(C, 0, M * N * sizeof(float));

  size_t m = 0;
  // 4-row register blocking
  for (; m + 4 <= M; m += 4) {
    size_t n = 0;
    // 16-column register blocking (2 x 8-float YMM registers per row)
    for (; n + 16 <= N; n += 16) {
      __m256 c00 = _mm256_setzero_ps();
      __m256 c01 = _mm256_setzero_ps();
      __m256 c10 = _mm256_setzero_ps();
      __m256 c11 = _mm256_setzero_ps();
      __m256 c20 = _mm256_setzero_ps();
      __m256 c21 = _mm256_setzero_ps();
      __m256 c30 = _mm256_setzero_ps();
      __m256 c31 = _mm256_setzero_ps();

      for (size_t k = 0; k < K; ++k) {
        const float *b_ptr = B + k * N + n;
        __m256 b0 = _mm256_loadu_ps(b_ptr);
        __m256 b1 = _mm256_loadu_ps(b_ptr + 8);

        __m256 a0 = _mm256_set1_ps(A[(m + 0) * K + k]);
        __m256 a1 = _mm256_set1_ps(A[(m + 1) * K + k]);
        __m256 a2 = _mm256_set1_ps(A[(m + 2) * K + k]);
        __m256 a3 = _mm256_set1_ps(A[(m + 3) * K + k]);

        c00 = _mm256_fmadd_ps(a0, b0, c00);
        c01 = _mm256_fmadd_ps(a0, b1, c01);
        c10 = _mm256_fmadd_ps(a1, b0, c10);
        c11 = _mm256_fmadd_ps(a1, b1, c11);
        c20 = _mm256_fmadd_ps(a2, b0, c20);
        c21 = _mm256_fmadd_ps(a2, b1, c21);
        c30 = _mm256_fmadd_ps(a3, b0, c30);
        c31 = _mm256_fmadd_ps(a3, b1, c31);
      }

      _mm256_storeu_ps(C + (m + 0) * N + n, c00);
      _mm256_storeu_ps(C + (m + 0) * N + n + 8, c01);
      _mm256_storeu_ps(C + (m + 1) * N + n, c10);
      _mm256_storeu_ps(C + (m + 1) * N + n + 8, c11);
      _mm256_storeu_ps(C + (m + 2) * N + n, c20);
      _mm256_storeu_ps(C + (m + 2) * N + n + 8, c21);
      _mm256_storeu_ps(C + (m + 3) * N + n, c30);
      _mm256_storeu_ps(C + (m + 3) * N + n + 8, c31);
    }

    // Residual 8 columns
    for (; n + 8 <= N; n += 8) {
      __m256 c0 = _mm256_setzero_ps();
      __m256 c1 = _mm256_setzero_ps();
      __m256 c2 = _mm256_setzero_ps();
      __m256 c3 = _mm256_setzero_ps();

      for (size_t k = 0; k < K; ++k) {
        __m256 b0 = _mm256_loadu_ps(B + k * N + n);
        __m256 a0 = _mm256_set1_ps(A[(m + 0) * K + k]);
        __m256 a1 = _mm256_set1_ps(A[(m + 1) * K + k]);
        __m256 a2 = _mm256_set1_ps(A[(m + 2) * K + k]);
        __m256 a3 = _mm256_set1_ps(A[(m + 3) * K + k]);

        c0 = _mm256_fmadd_ps(a0, b0, c0);
        c1 = _mm256_fmadd_ps(a1, b0, c1);
        c2 = _mm256_fmadd_ps(a2, b0, c2);
        c3 = _mm256_fmadd_ps(a3, b0, c3);
      }

      _mm256_storeu_ps(C + (m + 0) * N + n, c0);
      _mm256_storeu_ps(C + (m + 1) * N + n, c1);
      _mm256_storeu_ps(C + (m + 2) * N + n, c2);
      _mm256_storeu_ps(C + (m + 3) * N + n, c3);
    }

    // Scalar column tail for 4 rows
    for (; n < N; ++n) {
      for (size_t row = 0; row < 4; ++row) {
        float sum = 0.0f;
        for (size_t k = 0; k < K; ++k) {
          sum += A[(m + row) * K + k] * B[k * N + n];
        }
        C[(m + row) * N + n] = sum;
      }
    }
  }

  // Residual rows (M % 4)
  for (; m < M; ++m) {
    size_t n = 0;
    for (; n + 8 <= N; n += 8) {
      __m256 c = _mm256_setzero_ps();
      for (size_t k = 0; k < K; ++k) {
        __m256 a = _mm256_set1_ps(A[m * K + k]);
        __m256 b = _mm256_loadu_ps(B + k * N + n);
        c = _mm256_fmadd_ps(a, b, c);
      }
      _mm256_storeu_ps(C + m * N + n, c);
    }
    for (; n < N; ++n) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; ++k) {
        sum += A[m * K + k] * B[k * N + n];
      }
      C[m * N + n] = sum;
    }
  }
}

} // namespace cennan
