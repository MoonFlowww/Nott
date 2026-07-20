# Building Nott

Nott is a header-only framework layered over LibTorch: almost everything under `src/` is templates and inline code pulled in by whichever translation unit includes it. The `Nott` CMake target mainly exists to (1) resolve LibTorch, Boost and OpenCV once and expose them to every consumer, and (2) give IDEs a single place to index the full header tree. There is currently no test suite or benchmark harness in this repository — the buildable units are the library and the files under [`examples/`](../../../examples).

## Prerequisites

| Dependency | Notes |
| --- | --- |
| CMake ≥ 3.18 | |
| A C++20 compiler | GCC or Clang with full C++20 support |
| [LibTorch](https://pytorch.org/get-started/locally/) | Defaults to `/opt/libtorch`; override with `-DTorch_DIR=...` or `-D_libtorch_root=...` if installed elsewhere |
| Boost (`filesystem`) | Resolved via `find_package(Boost)` |
| OpenCV (`core`, `imgcodecs`, `imgproc`, `highgui`) | Resolved via `find_package(OpenCV)` |
| CUDA toolkit (optional) | Only needed for GPU builds; picked up automatically through LibTorch's CUDA-enabled build |

If LibTorch isn't at `/opt/libtorch`, point CMake at it explicitly:
```bash
cmake -S . -B build -DTorch_DIR=/path/to/libtorch/share/cmake/Torch
```

## Configuring and building the library

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

A plain build only compiles the `Nott` shared library (`libNott.so`). None of the examples are built by default — see below.

## Building examples

Every file under `examples/` gets its own executable target ([`examples/CMakeLists.txt`](../../../examples/CMakeLists.txt)), but every one of those targets is registered `EXCLUDE_FROM_ALL`. Each example links the full Torch/OpenCV stack, so compiling all of them unconditionally can take the better part of an hour — you must name the target(s) you want:

```bash
cmake --build build --target example_cifar10
cmake --build build --target example_etth example_ptbxl   # multiple at once
```

| Target | Source | What it demonstrates |
| --- | --- | --- |
| `example_ntk` | `examples/analysis/ntk.cpp` | Neural tangent kernel analysis on a synthetic spiral dataset |
| `example_overhead` | `examples/analysis/overhead.cpp` | Minimal CNN on MNIST, used to measure Nott's overhead vs. raw LibTorch |
| `example_cifar10` | `examples/classification/images/cifar10.cpp` | CIFAR-10 image classification with a Vision Transformer backbone |
| `example_dubai_segment` | `examples/classification/masks/dubai_segment.cpp` | Semantic segmentation on the Dubai aerial imagery dataset (uses OpenCV directly) |
| `example_ptbxl` | `examples/classification/timeseries/ptbxl.cpp` | PTB-XL ECG classification with a Conv1d-Residual backbone |
| `example_etth` | `examples/regression/timeseries/etth.cpp` | ETTh1 electricity transformer temperature forecasting with a TCN backbone |
| `example_speedtest` | `examples/speedtest.cpp` | Nott vs. raw LibTorch throughput/latency comparison |

Run a built example directly, e.g.:
```bash
./build/examples/example_cifar10
```

Most examples load a dataset from a hardcoded local path near the top of their `main()` (e.g. `/home/.../DATASETS/...`) — edit that path to point at your own copy of the dataset before running.

## Using Nott from another CMake project

Since everything lives in headers, the simplest integration is `add_subdirectory`:
```cmake
add_subdirectory(path/to/Nott)
target_link_libraries(your_target PRIVATE Nott)
```
This pulls in the include directories and the Torch/Boost/OpenCV link requirements transitively.
