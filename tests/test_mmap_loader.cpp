#include "cennan/core/mmap_loader.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

void test_mmap_loader_basic() {
  std::cout << "--- Testing MMapLoader Basic Lifecycle ---" << std::endl;

  // Create temporary test binary file
  const std::string test_file = "/tmp/cennan_test_mmap.bin";
  std::vector<uint8_t> test_data = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};

  {
    std::ofstream ofs(test_file, std::ios::binary);
    ofs.write(reinterpret_cast<const char *>(test_data.data()), test_data.size());
  }

  // 1. Open & inspect
  cennan::MMapLoader loader(test_file);
  assert(loader.is_open());
  assert(loader.size() == 8);
  assert(loader.data() != nullptr);

  // 2. Verify bytes
  for (size_t i = 0; i < test_data.size(); ++i) {
    assert(loader.data()[i] == test_data[i]);
  }

  // 3. Zero-copy slicing
  auto slice = loader.slice(2, 4);
  assert(slice.size() == 4);
  assert(slice[0] == 0x03);
  assert(slice[3] == 0x06);

  // 4. Move semantics
  cennan::MMapLoader moved_loader = std::move(loader);
  assert(!loader.is_open());
  assert(moved_loader.is_open());
  assert(moved_loader.size() == 8);

  moved_loader.advise_sequential();
  moved_loader.close();
  assert(!moved_loader.is_open());

  std::cout << "✓ MMapLoader Basic Tests Passed" << std::endl;
}

int main() {
  test_mmap_loader_basic();
  std::cout << "All MMapLoader tests passed successfully!" << std::endl;
  return 0;
}
