# BetterMotionBlur

A GPU-accelerated OpenFX 1.4 plugin (`BetterMotionBlur.ofx.bundle`) for **DaVinci Resolve** on Windows. It synthesizes natural shutter motion blur on high-shutter-speed footage and gapless long-exposure motion trails on standard video, operating natively in **DaVinci Wide Gamut / DaVinci Intermediate (DW/I)**.

## Techniques Used

- **Bidirectional Optical Flow & Caching**
  - Coarse-to-fine pyramidal **PatchMatch + Lucas-Kanade** (`eFlowCudaPyramidal`) and **Neural Proxy Flow** (`eFlowNeural`) with 4K joint-bilateral guided upsampling.
  - Canonical bidirectional FP16 flow caching (`FlowCache`) and stream-keyed GPU scratch buffer pooling to avoid redundant flow computation during sequential playback.
- **Natural Shutter Blur (`0°–720°`)**
  - Quadratic curved sub-frame trajectory integration using `16×16` Tile-Max and `3×3` Neighbor-Max velocity dilation, with tile-adaptive sample scaling (`Auto Shutter Samples`) so fast foreground edges blur smoothly over static backgrounds.
  - Optional matte-based foreground isolation to prevent foreground/background color bleeding.
- **Gapless Long-Exposure Motion Trails (`2–24` frames)**
  - Flow-bridged sub-frame splatting across consecutive frame pairs via 2-pass tile-aware shared-memory CUDA kernels.
  - Supports **Physical Exposure (Linear)**, **Light-Painting (Max/Screen)** with motion/luma gating, **Comet / Rear-Curtain Decay**, and an optional **Sequential Accumulation Cache**.
- **Native DW/I Linear-Light Color Pipeline**
  - Decodes DaVinci Intermediate to scene-linear DaVinci Wide Gamut for physically accurate energy-conserving blur, highlight streak boosting, and trail compositing before re-encoding to DW/I.
- **Runtime NVRTC & Built-in Diagnostics**
  - Compiles embedded CUDA C kernels at runtime via NVRTC (no offline `nvcc` required) with a multithreaded CPU fallback when CUDA is unavailable.
  - Includes diagnostic views (`Motion Vectors HSV`, `Occlusion Confidence`, `Trail Motion/Luma Mask`) for tuning edge preservation, thresholds, and mattes.

## Requirements

- **Operating System:** Windows 10 / 11 (64-bit)
- **Host Software:** DaVinci Resolve (OpenFX 1.4 compatible host)
- **GPU (Recommended):** NVIDIA GPU (Compute Capability 7.5+ / Turing or newer) with up-to-date NVIDIA drivers (`nvcuda.dll`) and NVRTC (`nvrtc64_120_0.dll`, included with DaVinci Resolve or CUDA Toolkit 12.x). A CPU fallback is used automatically if CUDA is unavailable.
- **Optional External Neural Model:** `onnxruntime.dll` (provided by DaVinci Resolve) when using a custom external `neuflow_v2.onnx` model; otherwise, the built-in GPU-native residual refinement kernel is used.
- **Building from Source (Optional):** CMake `>= 3.20` and MSVC with C++17 (x64) — OpenFX 1.4 headers and support library are vendored in `third_party/openfx`.

## Download

1. Go to the [Releases](https://github.com/dont-panic-guide/better-motion-blur/releases/latest) page.
2. Under **Assets**, download `BetterMotionBlur-Windows-x64.zip`.

## Installation

1. Extract `BetterMotionBlur-Windows-x64.zip`.
2. Copy the `BetterMotionBlur.ofx.bundle` folder to:
   ```text
   C:\Program Files\Common Files\OFX\Plugins\
   ```
3. Restart DaVinci Resolve. The plugin will appear under **OpenFX** as **Better Motion Blur**.

## Building from Source

```powershell
cmake -B build -A x64
cmake --build build --config Release
```

If you want to build against an external OpenFX 1.4 SDK instead of `third_party/openfx`, pass `-DOFX_SDK_ROOT=<path-to-openfx-sdk>` during the configure step.

## License

Released under the **GNU General Public License v3.0 or later** (`GPL-3.0-or-later`). See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.
