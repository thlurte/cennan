#pragma once

#include <span>
#include <cmath>
#include <cstddef>

namespace cennan {

constexpr float SQRT_2_OVER_PI = 0.7978845608f;
constexpr float GELU_COEFF = 0.044715f;

inline float sigmoid(float x) noexcept {
    return 1.0f / (1.0f + std::exp(-x));
}

inline float silu(float x) noexcept {
    return x * sigmoid(x);
}

// In-place GELU activation using the tanh approximation (Hendrycks & Gimpel, 2016)
inline void gelu_inplace(std::span<float> data) noexcept {
    for (float &x : data) {
        float x_cubed = x * x * x;
        float inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
        float tanh_val = std::tanh(inner);
        x = 0.5f * x * (1.0f + tanh_val);
    }
}

// In-place SiLU / Swish activation
inline void silu_inplace(std::span<float> data) noexcept {
    for (float &x : data) {
        x = silu(x);
    }
}

// SwiGLU forward pass: out[i] = x[i] * SiLU(gate[i])
inline void swiglu_forward(
    std::span<const float> x,
    std::span<const float> gate,
    std::span<float> out
) noexcept {
    const size_t n = x.size();
    for (size_t i = 0; i < n; ++i) {
        out[i] = x[i] * silu(gate[i]);
    }
}

} // namespace cennan
