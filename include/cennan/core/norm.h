#ifndef CENNAN_CORE_NORM_H
#define CENNAN_CORE_NORM_H

#include <cstddef>
#include <cstdint>
#include <span>

namespace cennan {

/// Vectorized Root Mean Square Normalization (RMSNorm) used in LLaMA, Mistral, Qwen, ColBERT.
/// y_i = (x_i / sqrt(mean(x^2) + eps)) * weight_i
void rms_norm_f32(const float *input, const float *weight, float *output, size_t dim, float eps = 1e-5f);

/// Vectorized Layer Normalization (LayerNorm)
/// y_i = ((x_i - mean(x)) / sqrt(var(x) + eps)) * weight_i + bias_i
void layer_norm_f32(const float *input, const float *weight, const float *bias, float *output, size_t dim, float eps = 1e-5f);

} // namespace cennan

#endif // CENNAN_CORE_NORM_H
