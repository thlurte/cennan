#include "cennan/core/tensor.h"
#include <numeric>
#include <stdexcept>

namespace cennan {

namespace {

inline size_t compute_numel(const std::vector<size_t> &shape) {
  if (shape.empty()) return 0;
  return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
}

} // anonymous namespace

Tensor::Tensor(std::vector<size_t> shape, DataType dtype)
    : shape_(std::move(shape)), numel_(compute_numel(shape_)), dtype_(dtype), owns_memory_(true) {
  if (dtype == DataType::Float32) {
    storage_.resize(numel_, 0.0f);
    data_ = storage_.data();
  }
}

Tensor::Tensor(std::vector<size_t> shape, float *data_ptr, bool owns_memory)
    : shape_(std::move(shape)), numel_(compute_numel(shape_)), dtype_(DataType::Float32),
      data_(data_ptr), owns_memory_(owns_memory) {
  if (owns_memory && data_ptr != nullptr && numel_ > 0) {
    storage_.assign(data_ptr, data_ptr + numel_);
    data_ = storage_.data();
  }
}

size_t Tensor::dim(size_t index) const {
  if (index >= shape_.size()) {
    throw std::out_of_range("Tensor dimension index out of range");
  }
  return shape_[index];
}

size_t Tensor::nbytes() const noexcept {
  switch (dtype_) {
    case DataType::Float32:
    case DataType::Int32:
      return numel_ * 4;
    case DataType::Float16:
    case DataType::BFloat16:
      return numel_ * 2;
    case DataType::Int8:
      return numel_ * 1;
    default:
      return numel_ * 4;
  }
}

const float *Tensor::row(size_t index) const {
  if (shape_.empty()) {
    throw std::runtime_error("Cannot get row from 0-dimensional tensor");
  }
  size_t stride = numel_ / shape_[0];
  if (index >= shape_[0]) {
    throw std::out_of_range("Row index out of range");
  }
  return data_ + index * stride;
}

float *Tensor::row(size_t index) {
  if (shape_.empty()) {
    throw std::runtime_error("Cannot get row from 0-dimensional tensor");
  }
  size_t stride = numel_ / shape_[0];
  if (index >= shape_[0]) {
    throw std::out_of_range("Row index out of range");
  }
  return data_ + index * stride;
}

void Tensor::reshape(std::vector<size_t> new_shape) {
  size_t new_numel = compute_numel(new_shape);
  if (new_numel != numel_) {
    throw std::invalid_argument("Cannot reshape tensor: total element count mismatch");
  }
  shape_ = std::move(new_shape);
}

} // namespace cennan
