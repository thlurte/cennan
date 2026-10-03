#ifndef CENNAN_CORE_MMAP_LOADER_H
#define CENNAN_CORE_MMAP_LOADER_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <span>
#include <string_view>
#include <system_error>

namespace cennan {

/// Zero-copy POSIX memory-mapped file reader for model weights and safetensors.
class MMapLoader {
public:
  MMapLoader() = default;
  explicit MMapLoader(const std::string &filepath);
  ~MMapLoader();

  // Non-copyable, movable (RAII ownership of OS file descriptor and mmap region)
  MMapLoader(const MMapLoader &) = delete;
  MMapLoader &operator=(const MMapLoader &) = delete;
  MMapLoader(MMapLoader &&other) noexcept;
  MMapLoader &operator=(MMapLoader &&other) noexcept;

  void open(const std::string &filepath);
  void close() noexcept;

  [[nodiscard]] bool is_open() const noexcept { return addr_ != nullptr; }
  [[nodiscard]] size_t size() const noexcept { return size_; }
  [[nodiscard]] const uint8_t *data() const noexcept { return addr_; }
  [[nodiscard]] const char *data_as_char() const noexcept {
    return reinterpret_cast<const char *>(addr_);
  }

  /// Read a subslice of memory without copying
  [[nodiscard]] std::span<const uint8_t> slice(size_t offset, size_t length) const;

  /// Advise kernel on memory access patterns
  void advise_sequential() const noexcept;
  void advise_random() const noexcept;

private:
  int fd_{-1};
  uint8_t *addr_{nullptr};
  size_t size_{0};
};

} // namespace cennan

#endif // CENNAN_CORE_MMAP_LOADER_H
