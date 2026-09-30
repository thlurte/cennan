# cennan

C++ embedding and latent representation library for dense, vision, and late-interaction models.

## Architecture

`cennan` is designed to provide zero-copy feature extraction directly into vector search engines like [`secan`](https://github.com/thlurte/secan).

- **Dense Models**: BERT, MiniLM, BGE
- **Multi-Vector / Late Interaction**: ColBERT v2
- **Vision Models**: SigLIP, ViT, ColPali

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

## License

MIT License. Copyright (c) 2026 Adheeb Ahmed.
