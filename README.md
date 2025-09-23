# CUDA Image Pipeline (Gaussian Blur + Sobel) with Multi-Stream

## Overview
This project implements a GPU image-processing pipeline in CUDA that applies a **separable Gaussian blur** followed by **Sobel edge detection** over **batches of large images**. It demonstrates:
- Running on tens of large frames (e.g., 20–40 images at 1080p–1440p, and 4K batches).
- Using **multiple CUDA streams** to overlap H2D/D2H transfers with device computation.

Outputs are written as `.pgm` images (blur + edges) into `out/results/`.

---

## Repository Layout
bin/ # Built executable(s)
data/ # (optional) input assets if needed
lib/ # (optional) third-party libs
out/
results/ # PGM outputs (*_blur.pgm, *_edges.pgm)
logs/ # timings.csv (optional)
submission/ # env + run logs for Coursera
src/
main.cu # entry point / host orchestration
kernels.cuh # CUDA kernels (Gaussian + Sobel)
utils.hpp # helpers (alloc, I/O)
timers.hpp # simple timing macros/utilities
Makefile # Linux/macOS build
build_windows.ps1 # Windows build (CUDA + MSVC)
run_samples.ps1 # Runs big cases + writes logs + ZIP evidence

yaml
Copy code

---

## Quickstart (Windows)

**Prerequisites**
- NVIDIA CUDA Toolkit (includes `nvcc`)
- Visual Studio **Build Tools 2022** with **Desktop development with C++**

**Build**
```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
Run (example)

powershell
Copy code
.\bin\image_pipeline.exe --n 20 --w 2560 --h 1440 --streams 8 --sigma 1.6
Expected console format:

makefile
Copy code
Config: 2560x1440, images=20, streams=8, tpb=256, sigma=1.60
Timing(ms): H2D=6.971  BLUR=33.267  SOBEL=1.132  D2H=104.127  TOTAL=627.521
Outputs:

out/results/img_XXXXX_blur.pgm

out/results/img_XXXXX_edges.pgm

Quickstart (Linux/macOS)
Prerequisites

CUDA Toolkit with nvcc

gcc/clang toolchain

Build

bash
Copy code
make
Run (example)

bash
Copy code
./bin/image_pipeline.exe --n 20 --w 2560 --h 1440 --streams 8 --sigma 1.6
Command-Line Interface
php
Copy code
image_pipeline.exe
  --n <int>        # number of images (batch size)
  --w <int>        # image width
  --h <int>        # image height
  --streams <int>  # number of CUDA streams (e.g., 1,4,8,12)
  --sigma <float>  # Gaussian sigma (kernel derived internally)
  --tpb <int>      # optional: threads per block (default 256)
Examples:

powershell
Copy code
# Many large frames w/ overlap
.\bin\image_pipeline.exe --n 30 --w 2560 --h 1440 --streams 8 --sigma 1.6

# Full-HD batch
.\bin\image_pipeline.exe --n 40 --w 1920 --h 1080 --streams 8 --sigma 1.4

# 4K batch
.\bin\image_pipeline.exe --n 12 --w 3840 --h 2160 --streams 8 --sigma 2.0
Reproduce Evidence (logs + ZIP)
This script runs three large cases, writes raw logs under out/submission/, appends out/logs/timings.csv, and creates submission_evidence.zip in the repo root.

powershell
Copy code
powershell -ExecutionPolicy Bypass -File .\run_samples.ps1
Upload submission_evidence.zip to Coursera as “Proof of execution artifacts”.

Example Results (captured on Windows)
makefile
Copy code
# 20 images @ 2560x1440, streams=8, sigma=1.6
Config: 2560x1440, images=20, streams=8, tpb=256, sigma=1.60
Timing(ms): H2D=6.971  BLUR=33.267  SOBEL=1.132  D2H=104.127  TOTAL=627.521
PGM files are written into out/results/ as *_blur.pgm and *_edges.pgm.

Implementation Notes
Separable Gaussian: two 1D passes (X then Y) reduce arithmetic vs full 2D convolution.

Sobel: gradient magnitude from horizontal/vertical filters.

Multi-stream: chunked batches mapped to CUDA streams to overlap transfers and kernels.

Timing: simple host-side timers; optional CSV aggregation.

Troubleshooting
nvcc fatal: Cannot find compiler 'cl.exe' (Windows)
Install VS 2022 Build Tools and include Desktop development with C++. Re-run build_windows.ps1.

Can’t view .pgm
Use IrfanView, XnView, or convert to PNG:

sql
Copy code
magick convert img_00000_edges.pgm img_00000_edges.png
Long D2H time
Try increasing streams or using pinned host memory (future work).

License
GPL-3.0 (see LICENSE).

