#include "cennan/core/mmap_loader.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace cennan {

MMapLoader::MMapLoader(const std::string &filepath) {
  open(filepath);
}

MMapLoader::~MMapLoader() {
  close();
}

MMapLoader::MMapLoader(MMapLoader &&other) noexcept
    : fd_(other.fd_), addr_(other.addr_), size_(other.size_) {
  other.fd_ = -1;
  other.addr_ = nullptr;
  other.size_ = 0;
}

MMapLoader &MMapLoader::operator=(MMapLoader &&other) noexcept {
  if (this != &other) {
    close();
    fd_ = other.fd_;
    addr_ = other.addr_;
    size_ = other.size_;
    other.fd_ = -1;
    other.addr_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

void MMapLoader::open(const std::string &filepath) {
  close();

  fd_ = ::open(filepath.c_str(), O_RDONLY);
  if (fd_ == -1) {
    throw std::runtime_error("MMapLoader: Failed to open file: " + filepath + " (" + std::strerror(errno) + ")");
  }

  struct stat sb{};
  if (::fstat(fd_, &sb) == -1) {
    ::close(fd_);
    fd_ = -1;
    throw std::runtime_error("MMapLoader: Failed to stat file: " + filepath);
  }

  size_ = static_cast<size_t>(sb.st_size);
  if (size_ == 0) {
    // Empty file, don't mmap 0 bytes
    return;
  }

  void *mapped = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
  if (mapped == MAP_FAILED) {
    ::close(fd_);
    fd_ = -1;
    size_ = 0;
    throw std::runtime_error("MMapLoader: mmap failed for: " + filepath + " (" + std::strerror(errno) + ")");
  }

  addr_ = static_cast<uint8_t *>(mapped);
}

void MMapLoader::close() noexcept {
  if (addr_ != nullptr && size_ > 0) {
    ::munmap(addr_, size_);
    addr_ = nullptr;
  }
  if (fd_ != -1) {
    ::close(fd_);
    fd_ = -1;
  }
  size_ = 0;
}

std::span<const uint8_t> MMapLoader::slice(size_t offset, size_t length) const {
  if (offset + length > size_) {
    throw std::out_of_range("MMapLoader::slice: slice range [" + std::to_string(offset) + ", " +
                            std::to_string(offset + length) + "] exceeds mapped size " + std::to_string(size_));
  }
  return {addr_ + offset, length};
}

void MMapLoader::advise_sequential() const noexcept {
  if (addr_ != nullptr && size_ > 0) {
    ::madvise(addr_, size_, MADV_SEQUENTIAL | MADV_WILLNEED);
  }
}

void MMapLoader::advise_random() const noexcept {
  if (addr_ != nullptr && size_ > 0) {
    ::madvise(addr_, size_, MADV_RANDOM);
  }
}

} // namespace cennan
