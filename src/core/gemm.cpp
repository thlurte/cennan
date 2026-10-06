#include "cennan/core/gemm.h"
#include <immintrin.h>

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

} // namespace cennan
