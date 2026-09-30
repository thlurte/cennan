#ifndef CENNAN_CORE_TENSOR_H
#define CENNAN_CORE_TENSOR_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace cennan {

enum class DataType {
  Float32,
  Float16,
  BFloat16,
  Int8,
  Int32
};

/// High-performance multi-dimensional tensor view / buffer.
/// Supports both owned memory buffers and zero-copy non-owning memory mapped views.
class Tensor {
public:
  Tensor() = default;
  Tensor(std::vector<size_t> shape, DataType dtype = DataType::Float32);
  Tensor(std::vector<size_t> shape, float *data_ptr, bool owns_memory = false);

  // Metadata
  [[nodiscard]] const std::vector<size_t> &shape() const noexcept { return shape_; }
  [[nodiscard]] size_t dim(size_t index) const;
  [[nodiscard]] size_t numel() const noexcept { return numel_; }
  [[nodiscard]] size_t nbytes() const noexcept;
  [[nodiscard]] DataType dtype() const noexcept { return dtype_; }
  [[nodiscard]] bool is_empty() const noexcept { return numel_ == 0; }
  [[nodiscard]] bool owns_memory() const noexcept { return owns_memory_; }

  // Data pointers
  [[nodiscard]] float *data() noexcept { return data_; }
  [[nodiscard]] const float *data() const noexcept { return data_; }

  template <typename T>
  [[nodiscard]] T *data_as() noexcept {
    return reinterpret_cast<T *>(data_);
  }

  template <typename T>
  [[nodiscard]] const T *data_as() const noexcept {
    return reinterpret_cast<const T *>(data_);
  }

  // Row slice access for 2D/3D tensors
  [[nodiscard]] const float *row(size_t index) const;
  [[nodiscard]] float *row(size_t index);

  void reshape(std::vector<size_t> new_shape);

private:
  std::vector<size_t> shape_;
  size_t numel_{0};
  DataType dtype_{DataType::Float32};
  float *data_{nullptr};
  bool owns_memory_{false};
  std::vector<float> storage_; // Used when tensor owns its memory
};

} // namespace cennan

#endif // CENNAN_CORE_TENSOR_H
