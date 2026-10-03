# cennan

C++ embedding and latent representation library for dense, vision, and late-interaction models.

## Architecture

`cennan` is designed to provide zero-copy feature extraction directly into vector search engines like [`secan`](https://github.com/thlurte/secan).

- **Dense Models**: BERT, MiniLM, BGE, ModernBERT
- **Multi-Vector / Late Interaction**: ColBERT v2
- **Vision Models**: SigLIP, ViT, ColPali
- **Zero-Copy Ingestion**: POSIX `mmap` Safetensors reader

## Core Components

### 1. `cennan::Tensor`
Multi-dimensional tensor abstraction supporting owned contiguous heap buffers and zero-copy non-owning memory views.

### 2. `cennan::MMapLoader`
RAII-managed zero-copy binary and weight loader with `madvise` page hints (`MADV_SEQUENTIAL`, `MADV_WILLNEED`).

### 3. SIMD Normalization Kernels
Vectorized AVX2 + FMA `rms_norm_f32` (Root Mean Square Normalization for LLaMA / Mistral / ColBERT) and `layer_norm_f32`.

## Build & Test

```bash
# Configure Release build
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build library, executable, and tests
cmake --build build -j$(nproc)

# Run full test suite
ctest --test-dir build --output-on-failure
```

## Roadmap

- [x] Modern C++20 CMake Build Harness (`-O3 -march=native -mavx2 -mfma`)
- [x] Multi-Dimensional Tensor Abstraction (`cennan::Tensor`)
- [x] Zero-Copy POSIX `MMapLoader` with `slice()` view semantics
- [x] Vectorized AVX2 + FMA `RMSNorm` and `LayerNorm` Kernels
- [ ] Safetensors 8-Byte JSON Header Parser
- [ ] Batch Matrix Multiplication (GEMM) for Linear Projection Layers
- [ ] Standalone C++ WordPiece / BPE Tokenizer

## License

MIT License. Copyright (c) 2026 Adheeb Ahmed.
