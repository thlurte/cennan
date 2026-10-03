#include "cennan/core/norm.h"
#include <cmath>
#include <immintrin.h>

namespace cennan {

void rms_norm_f32(const float *input, const float *weight, float *output, size_t dim, float eps) {
  // 1. Calculate sum of squares with AVX2
  __m256 sum_sq_vec = _mm256_setzero_ps();
  size_t i = 0;

  for (; i + 8 <= dim; i += 8) {
    __m256 x = _mm256_loadu_ps(input + i);
    sum_sq_vec = _mm256_fmadd_ps(x, x, sum_sq_vec);
  }

  // Horizontal sum of AVX2 register
  __m128 lo = _mm256_castps256_ps128(sum_sq_vec);
  __m128 hi = _mm256_extractf128_ps(sum_sq_vec, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_hadd_ps(s, s);
  s = _mm_hadd_ps(s, s);
  float sum_sq = _mm_cvtss_f32(s);

  // Scalar tail
  for (; i < dim; ++i) {
    sum_sq += input[i] * input[i];
  }

  float variance = sum_sq / static_cast<float>(dim);
  float inv_rms = 1.0f / std::sqrt(variance + eps);

  // 2. Normalize and scale
  __m256 vinv = _mm256_set1_ps(inv_rms);
  i = 0;
  for (; i + 8 <= dim; i += 8) {
    __m256 x = _mm256_loadu_ps(input + i);
    __m256 norm_x = _mm256_mul_ps(x, vinv);
    if (weight != nullptr) {
      __m256 w = _mm256_loadu_ps(weight + i);
      norm_x = _mm256_mul_ps(norm_x, w);
    }
    _mm256_storeu_ps(output + i, norm_x);
  }

  // Scalar tail
  for (; i < dim; ++i) {
    float val = input[i] * inv_rms;
    if (weight != nullptr) {
      val *= weight[i];
    }
    output[i] = val;
  }
}

void layer_norm_f32(const float *input, const float *weight, const float *bias, float *output, size_t dim, float eps) {
  // 1. Mean calculation
  float sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    sum += input[i];
  }
  float mean = sum / static_cast<float>(dim);

  // 2. Variance calculation
  float var_sum = 0.0f;
  for (size_t i = 0; i < dim; ++i) {
    float diff = input[i] - mean;
    var_sum += diff * diff;
  }
  float variance = var_sum / static_cast<float>(dim);
  float inv_std = 1.0f / std::sqrt(variance + eps);

  // 3. Normalize, scale & bias
  for (size_t i = 0; i < dim; ++i) {
    float val = (input[i] - mean) * inv_std;
    if (weight != nullptr) {
      val *= weight[i];
    }
    if (bias != nullptr) {
      val += bias[i];
    }
    output[i] = val;
  }
}

} // namespace cennan
