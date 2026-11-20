<!-- doc-type: changelog -->
# Changelog

## [Unreleased]
### Added
- Native OpenFX 1.4 plugin (`BetterMotionBlur.ofx.bundle`) for DaVinci Resolve on Windows with CUDA (runtime-compiled via NVRTC) and multithreaded CPU fallback.
- Dual optical flow engines: Neural Optical Flow (`neuflow_v2.onnx` via ORT 1.13.1 or GPU-native `embeddedResidualRefineKernel`) and Fast CUDA Pyramidal (`PatchMatch + Lucas-Kanade`), with 4K Joint-Bilateral Guided Upsampling.
- Natural Shutter Blur (`0°–720°` curved quadratic integration, coalesced `16×16` Tile-Max / Neighbor-Max dilation, foreground matte isolation).
- Long-Exposure Motion Trail (`2–24` frames, adaptive flow-bridged sub-frame splatting, Physical Exposure, Light-Painting Max/Screen, Comet/Rear-Curtain Decay, optional Sequential Accumulation Cache).
- Native **DaVinci Wide Gamut / DaVinci Intermediate (DW/I)** linear-light integration and DWG CIE $Y$ luminance pipeline.
- Built-in Diagnostic Views (`Motion Vectors HSV`, `Occlusion Confidence`, `Trail Motion/Luma Mask`).
- Canonical bidirectional `FlowCache` (`1.5 GB` LRU), stream-keyed GPU scratch pool (`512 MB` cap), and graceful OFX `unload()` / Windows Loader-Lock safe teardown.
- Automated 8-test verification and 1080p/4K benchmark suite (`BetterMotionBlurTests.exe`).
- Recorded GNU GPLv3 (`GPL-3.0-or-later`) release licensing intent in `docs/00-Project-Intent-and-Overview.md`.

<!-- FROZEN_PUSHED_HISTORY -->
