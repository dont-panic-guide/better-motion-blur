# 00 — Project Intent & Architecture Overview

## User Intent & Product Goals
**BetterMotionBlur** is an OpenFX 1.4 plugin engineered specifically for **DaVinci Resolve** on Windows (NVIDIA CUDA primary, multithreaded CPU fallback) to solve two motion-synthesis problems cleanly in a single tool:
1. **Natural Motion Blur for High-Shutter-Speed Footage**:
   - Outperform basic frame-blending or artifact-prone optical-flow plugins (such as RSMB) on fast movement.
   - Uses bidirectional optical flow (`t -> t+1` and `t -> t-1`) to fit **quadratic curved trajectories**, McGuire/Guertin **Tile-Max / Neighbor-Max** velocity dilation so fast foreground edges blur smoothly onto static backgrounds, and optional **Matte foreground isolation** to prevent background/foreground color bleeding.
2. **Smooth Long-Exposure Motion Trails for Regular Video**:
   - Emulate long-exposure photography without stroboscopic stepping/dots on fast-moving subjects or lights.
   - Integrates across a multi-frame lookback window (`2–24` frames) using **flow-bridged sub-frame splatting** between consecutive frames (`frame[k]` -> `frame[k-1]`).
   - Supports three artistic/physical modes: **Physical Exposure (Linear)**, **Light-Painting (Max/Screen)** with motion/luma gating, and **Comet / Rear-Curtain Decay** with crisp current-frame subject blending and optional **Sequential Accumulation Cache** for ultra-long trails.
3. **Native DaVinci Wide Gamut / DaVinci Intermediate (DW/I) Pipeline**:
   - Always assumes incoming and outgoing images are encoded in **DaVinci Wide Gamut / DaVinci Intermediate (DW/I)** (no sRGB toggle).
   - Decodes to scene-linear DWG light for physically accurate energy-conserving blur and highlight streaks, then re-encodes to DW/I.
4. **Edge-Artifact Diagnostics & Zero-Degradation Speed**:
   - Provides built-in **Diagnostic Views** (`Motion Vectors HSV`, `Occlusion Confidence`, `Trail Motion/Luma Mask`) so users can inspect flow quality and tune `Edge Preservation`, `Motion Threshold`, or `Matte` isolation directly.
   - Optimized for real-time 1080p/4K GPU playback with shared-memory CUDA kernels, stream-keyed scratch buffer pooling, canonical bidirectional flow caching, and GPU-native embedded neural refinement.
5. **Graceful Unload & Process Exit**:
   - Cleanly frees GPU buffers, CUDA modules/contexts, and ONNX Runtime sessions via OFX `unload()` (`kOfxActionUnload`) and guards against Windows OS Loader-Lock deadlocks/crashes on `ExitProcess`.

---

## High-Level Architecture

```
DaVinci Resolve (OpenFX 1.4 Host)
  │
  ▼
BetterMotionBlurPlugin (.ofx)  [src/BetterMotionBlurPlugin.h/.cpp]
  ├─► CUDA Stream Available ──► CudaEngine [src/CudaEngine.h/.cpp]
  │                               ├─ NVRTC Runtime Compiler (nvcuda.dll + nvrtc64_120_0.dll)
  │                               ├─ Stream-Keyed Scratch Buffer Pool (512 MB cap)
  │                               ├─ Canonical FlowCache [src/FlowCache.h/.cpp]
  │                               └─ Optional External ONNX (neuflow_v2.onnx) via
  │                                  NeuralFlowEngine [src/NeuralFlowEngine.h/.cpp]
  │                                  (or GPU-native embeddedResidualRefineKernel)
  │
  └─► Non-CUDA Fallback ───────► CpuFallbackProcessor [src/CpuFallbackProcessor.h/.cpp]
```

## Source File Map
| File | Responsibility |
| :--- | :--- |
| `CMakeLists.txt` | Builds `BetterMotionBlur.ofx` bundle (`Contents/Win64/BetterMotionBlur.ofx`) and `BetterMotionBlurTests.exe` against Resolve's OpenFX 1.4 Support SDK. |
| `src/CommonTypes.h` | Shared enums (`EffectMode`, `FlowEngineType`, `TrailBlendMode`, `MatteMode`, `DiagnosticView`), `RenderParams`, and `FrameInput`. |
| `src/BetterMotionBlurPlugin.h/.cpp` | OFX factory, parameter descriptors, temporal frame negotiation (`getFramesNeeded`), `isIdentity`, CUDA/CPU processor dispatch, and `unload()`. |
| `src/FlowCache.h/.cpp` | Thread-safe LRU cache for bidirectional flow entries (`FlowEntry`) and per-instance `SequentialTrailState`. |
| `src/NeuralFlowEngine.h/.cpp` | Dynamic loader for Resolve's `onnxruntime.dll` (ORT 1.13.1), external/embedded ONNX session management, and `refineFlow`. |
| `src/OnnxModelGenerator.h` | Embedded ONNX IR v7 residual flow-refinement graph generator (also serves as fallback when no external `neuflow_v2.onnx` is present). |
| `src/CudaEngine.h/.cpp` | Embedded CUDA C kernels compiled via NVRTC, stream-keyed scratch pool, shared pyramids, canonical flow estimation, shutter blur, and long-exposure trails. |
| `src/CpuFallbackProcessor.h/.cpp` | Multithreaded-ready CPU mirror of optical flow, DW/I linear-light shutter blur, and flow-bridged trail synthesis. |
| `tests/TestRunner.cpp` | 8 automated unit, visual-accuracy, cache, OFX export, lifecycle, and 1080p/4K GPU benchmark tests. |
