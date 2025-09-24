
# CUDA Multi-Stream Image Pipeline (Gaussian Blur + Sobel)

## Overview
A CUDA-based image pipeline that applies a **separable Gaussian blur** followed by **Sobel edge detection** over **batches of large images**. The project demonstrates:
- Running at scale (tens of images at 1080p–1440p and 4K).
- Using **multiple CUDA streams** to overlap H2D/D2H transfers with kernel execution.
- Reproducible evidence (logs, timings CSV, and sample images) with helper scripts.

Outputs are written as `.pgm` images (blur + edges) under `out/results/`. Optional conversion to `.png` is provided for easier viewing.

---

## Features
- Separable Gaussian blur (two 1D passes) + Sobel magnitude.
- Multi-stream host orchestration to overlap copies and kernels.
- Command-line interface: `--n --w --h --streams --sigma [--tpb]`.
- **Windows** build script (`build_windows.ps1`) using NVCC + MSVC.
- **Linux/macOS** portable `Makefile`.
- Helper script `run_samples.ps1` to run large presets, parse timings → CSV, and package a submission ZIP.

---

## Repository Layout
```

bin/                 # built executable(s)
out/
results/           # PGM outputs (\*\_blur.pgm, \*\_edges.pgm) and optional PNGs
logs/              # timings.csv (aggregated timings)
submission/        # environment + run logs for grading (ZIP source)
src/
main.cu            # host orchestration / CLI
kernels.cuh        # CUDA kernels (Gaussian + Sobel)
utils.hpp          # helpers (alloc, I/O)
timers.hpp         # simple timing macros/utilities
Makefile             # Linux/macOS build
build\_windows.ps1    # Windows build (CUDA + MSVC)
run\_samples.ps1      # batch runner (logs + CSV + ZIP)

````

---

## Build & Run

### Windows (MSVC + NVCC)
**Requirements**
- NVIDIA CUDA Toolkit (includes `nvcc`)
- Visual Studio **Build Tools 2022** (Desktop development with C++)

**Build**
```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
````

**Run (example)**

```powershell
.\bin\image_pipeline.exe --n 20 --w 2560 --h 1440 --streams 8 --sigma 1.6
```

### Linux/macOS

**Requirements**: CUDA Toolkit (`nvcc`) and a supported host compiler.

**Build**

```bash
make
```

**Run (example)**

```bash
./bin/image_pipeline.exe --n 20 --w 2560 --h 1440 --streams 8 --sigma 1.6
```

---

## CLI

```
image_pipeline.exe
  --n <int>        # number of images (batch size)
  --w <int>        # image width
  --h <int>        # image height
  --streams <int>  # CUDA streams (e.g., 1, 4, 8, 12)
  --sigma <float>  # Gaussian sigma (kernel derived internally)
  --tpb <int>      # optional threads per block (default 256)
```

---

## Quick Evidence (Windows)

The helper script runs three large cases, saves logs to `out/submission/`, appends a timings CSV at `out/logs/timings.csv`, and produces a ZIP file for submission.

```powershell
# Build
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1

# Run presets + logs + CSV + ZIP
powershell -ExecutionPolicy Bypass -File .\run_samples.ps1

# The ZIP is created as: .\submission_evidence.zip
```

If you prefer a single custom run:

```powershell
.\bin\image_pipeline.exe --n 30 --w 2560 --h 1440 --streams 8 --sigma 1.6
```

---

## Example Timings (Windows)

```
# 2560x1440, N=20, streams=8, sigma=1.6
Config: 2560x1440, images=20, streams=8, tpb=256, sigma=1.60
Timing(ms): H2D=6.971  BLUR=33.267  SOBEL=1.132  D2H=104.127  TOTAL=627.521

# 3840x2160, N=12, streams=8, sigma=2.0
Config: 3840x2160, images=12, streams=8, tpb=256, sigma=2.00
Timing(ms): H2D=16.605  BLUR=10.540  SOBEL=2.688  D2H=204.780  TOTAL=1592.423
```

---

## How it works

* **Separable Gaussian**: two 1D passes (X then Y) reduce arithmetic and improve cache reuse vs a full 2D convolution.
* **Sobel**: gradient magnitude from horizontal/vertical filters.
* **Multi-stream**: the batch is partitioned across CUDA streams to **overlap** H2D/D2H with kernels and improve throughput.
* **Timing**: the app prints `Timing(ms): ...`; scripts parse and append to `out/logs/timings.csv`.

---

## Style & Code Quality

The code is organized in small, single-purpose functions and headers (`kernels.cuh`, `utils.hpp`, `timers.hpp`), uses descriptive names and comments, and exposes a clear CLI. A future step would add a `.clang-format` and `cpplint` config to align even more closely with the Google C++ Style Guide.

---

## Troubleshooting

* **`nvcc fatal: cannot find 'cl.exe'`** → Install VS 2022 Build Tools (Desktop C++), then run `build_windows.ps1`.
* **Cannot view `.pgm`** → Install ImageMagick (`winget install -e --id ImageMagick.ImageMagick`) and convert to PNG:

  ```powershell
  magick mogrify -format png .\out\results\*.pgm
  ```
* **Long D2H timing** → Increase `--streams`; consider pinned (page-locked) memory and CUDA events in future work.

---

## License

GPL-3.0 (see LICENSE).

---

## Citation / Course Context

Developed as part of a GPU programming capstone; the pipeline and scripts are designed to make peer review easy: build, run with CLI, inspect outputs, and verify logs/ZIP evidence quickly.


