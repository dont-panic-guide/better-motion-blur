<!-- doc-type: product-features -->
# BetterMotionBlur (OpenFX 1.4)

GPU-accelerated OpenFX plugin (`BetterMotionBlur.ofx.bundle`) for **DaVinci Resolve** on Windows, built for:
1. **Natural Shutter Blur** on high-shutter-speed footage (curved quadratic sub-frame integration, Tile-Max / Neighbor-Max dilation, matte foreground isolation).
2. **Gapless Long-Exposure Motion Trails** on standard-shutter clips (flow-bridged sub-frame splatting, Physical Exposure, Light-Painting, and Comet/Rear-Curtain modes).
3. **Native DaVinci Wide Gamut / DaVinci Intermediate (DW/I)** linear-light color pipeline.

## Project Documentation (`docs/`)
- [00 — Project Intent & Overview](docs/00-Project-Intent-and-Overview.md) — Core goals, design decisions, architecture, build & install.
- [01 — OFX Plugin & Parameters](docs/01-OFX-Plugin-and-Parameters.md) — OFX 1.4 contexts, temporal frame requests, parameter table, CPU/GPU dispatch.
- [02 — Optical Flow & Caching](docs/02-Optical-Flow-and-Caching.md) — Pyramidal PatchMatch+LK, Neural proxy refinement (embedded GPU & external ONNX), canonical bidirectional LRU cache, and stream-keyed scratch pool.
- [03 — Blur, Trails, Color & Diagnostics](docs/03-Blur-Trail-Color-and-Diagnostics.md) — Shutter blur synthesis, flow-bridged trail splatting, DW/I transfer math, and diagnostic views.
- [04 — Lifecycle, Performance & Testing](docs/04-Lifecycle-Performance-and-Testing.md) — Windows DLL/CUDA/ORT unload safety, NVRTC compilation, and automated verification (`BetterMotionBlurTests.exe`).
- [CHANGELOG.md](CHANGELOG.md) — Unreleased and pushed release history.

## Quick Build & Test (Windows)
```powershell
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" -B build -G "Visual Studio 18 2026" -A x64
& "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" --build build --config Release
.\build\Release\BetterMotionBlurTests.exe
```

## Install in DaVinci Resolve
Copy `build\Release\BetterMotionBlur.ofx.bundle` to:
`C:\Program Files\Common Files\OFX\Plugins\BetterMotionBlur.ofx.bundle`
