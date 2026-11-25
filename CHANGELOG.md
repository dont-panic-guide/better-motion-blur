<!-- doc-type: changelog -->
# Changelog

## [Unreleased]

<!-- FROZEN_PUSHED_HISTORY -->

## [v1.2.0] - 2026-11-25
### Added
- **Shutter Timebase (`shutterTimebase`), Trail Timebase (`trailTimebase`) & Sub-Frame / Pulldown Handling**: Added `Timeline FPS` vs `Source Clip FPS` selectors for both Shutter Blur and Long-Exposure Trail, automatic duplicate-frame skipping for lower-FPS clips on higher-FPS timelines (`areFramesDistinct`, including `SequentialTrailState` repeat guard), and fractional sub-frame probing (`clipFps > timelineFps`) with piecewise Voronoi linear-DWG shutter integration and milli-frame-keyed sub-frame trail splatting.
- **Test 11 & Test 12**: Added physical 180° 64-subframe camera ground-truth comparison, 1080p `140 px/frame` optical flow accuracy, curved exterior arc, single-sided occlusion, multi-subframe/`eTimebaseClipFps` verification (`Test 11`), plus Long-Exposure curved circular arc, `eMatteForegroundIsolation` background non-smearing, short-window wake backfill, and `pairTimeStep` Comet scaling (`Test 12`).

### Fixed & Changed
- **Long-Exposure Trail Upgrades (`CudaEngine` & `CpuFallbackProcessor`)**:
  - Added time-invariant confidence-gated quadratic curved splatting in `trailPairSplatMovingKernel` and `CpuFallbackProcessor` using `pairFlows[k + 1]` ($t_{k-1} \to t_{k-2}$), eliminating polygonal kinks on arcs while preserving `TrailSplatCache` invariance.
  - Added `eMatteForegroundIsolation` (`matteMode == 1`) emission gating on `matteA` and crisp background compositing (`bgW = max(0, totalWindowW - acc.w)`), plus short-window wake backfill (`acc.w < 0.25`) from `lastRGBA`.
- **Physical 180° Shutter Reconstruction (`CudaEngine` & `CpuFallbackProcessor`)**:
  - Replaced background-discarding `wSample` renormalization with physically normalized foreground/background integration (`fgFrac` + `bgFrac` with unoccluded `centerLin` fallback and Jacobian weighting) so silhouettes taper smoothly (`100% -> 50% -> 0%`) and thin fast subjects turn semi-transparent.
  - Added confidence-weighted bidirectional velocity/acceleration (`evalConfWeightedVelAcc` using `confFwd`/`confBwd`) to prevent 50% velocity loss at single-sided occlusions, corrected inverse gather kinematics (`-v * tau - 0.5 * a * tau^2`), carried acceleration `aDom` into separable dynamic-radius `neighborMaxVelH/V` (`tileRadius` up to `12` tiles), and sampled `prevRGBA`/`nextRGBA` for true temporal disocclusion behind moving edges.
  - Expanded optical flow pyramid to `7` levels (`~320 px/frame` search at 4K, `~160 px/frame` at 1080p), calibrated DW/I joint-bilateral `sigmaRange = 0.012 + (1 - flowDetail) * 0.055`, and added 1D parabolic sub-pixel refinement in `correlationFlowKernel`.

## [v1.1.0] - 2026-11-21
### Added
- **Tile-Adaptive Shutter Samples (`autoShutterSamples`)**: Enabled by default; dynamically scales `synthesizeShutterBlurKernel` and `CpuFallbackProcessor` sample counts per `16×16` tile using `neighborMax` velocity (`clamp(ceil(domSpeed * shutterSpan * 1.25), min(samples, 8), samples)`), using `shutterSamples` (default `48`) as the upper cap with zero CUDA intra-warp divergence.
- **Dynamic OFX Parameter Visibility (`changedParam`)**: Automatically hides/disables inactive parameter groups and mode-specific controls (`groupShutter`, `groupTrail`, `trailDecay`, `rearCurtainCrispness`, `lumaThreshold`, `highlightBoost`, `matteMode`) in Resolve's Inspector based on the active Render Mode, Trail Blend Mode, Sequential Cache toggle, and OFX context.
- **GitHub Actions Release Workflow**: Added `.github/workflows/release.yml` to build and package `BetterMotionBlur-Windows-x64.zip` on `v*` tags and `workflow_dispatch`.
- **Test 10**: Added mixed-velocity GPU/CPU parity and `eModeCombined` validation for tile-adaptive shutter samples in `TestRunner.cpp`.

### Changed
- **Long-Exposure & Flow Pipeline Performance**:
  - Replaced per-frame full-window global-atomic re-splatting with 2-pass tile-aware SoA shared-memory splatting (`trailPairInitStaticKernel` + `trailPairSplatMovingKernel`), register cell-coalescing, single-instruction hardware `atomicMax`, multiplicative Comet decay, inline matte reads, and fused `trailCompositeAndFinalizeKernel` (linear `sequentialCache`, exact Comet scalar factorization).
  - Added per-pair `TrailSplatCache` and compressed GPU `FlowEntry` storage to FP16 `uint2` + `uchar2` (`10 B/px`, `82.9 MB` at 4K) inside unified `FlowCache` with `cuMemGetInfo_v2` dynamic budget (`1.5–5.0 GB`) and `768 MB` stream-pool recycling for evicted cache buffers, eliminating 4K LRU thrashing up to `trailFrames = 24`.
  - Enabled shared `PyramidSet` reuse across bidirectional shutter pairs and consecutive Long-Exposure misses, byte-identity guard in `NeuralFlowEngine` so embedded `neuflow_v2.onnx` stays on GPU-native `embeddedResidualRefineKernel` (`1080p` Neural shutter blur `4.7 ms / 212 fps`, `4K` `10.1 ms / 99.2 fps`), and trimmed OFX host lookback to `max(1, trailFrames - 1)` with conditional matte requests.
  - Added `CudaEngine::downloadFlowToHost()` and `Test 9` covering Long-Exposure mathematical parity (`0.00000` diff vs scalar reference), `> 40 px` global fallback, `sequentialCache`, `Combined` cache non-pollution, and 1080p/4K benchmarks (`4K trailFrames=16` at `18.3 ms / 54.7 fps`, `93%` cache hit rate).

## [v1.0.0] - 2026-11-20
### Added
- Native OpenFX 1.4 plugin (`BetterMotionBlur.ofx.bundle`) for DaVinci Resolve on Windows with CUDA (runtime-compiled via NVRTC) and multithreaded CPU fallback.
- Dual optical flow engines: Neural Optical Flow (`neuflow_v2.onnx` via ORT 1.13.1 or GPU-native `embeddedResidualRefineKernel`) and Fast CUDA Pyramidal (`PatchMatch + Lucas-Kanade`), with 4K Joint-Bilateral Guided Upsampling.
- Natural Shutter Blur (`0°–720°` curved quadratic integration, coalesced `16×16` Tile-Max / Neighbor-Max dilation, foreground matte isolation).
- Long-Exposure Motion Trail (`2–24` frames, adaptive flow-bridged sub-frame splatting, Physical Exposure, Light-Painting Max/Screen, Comet/Rear-Curtain Decay, optional Sequential Accumulation Cache).
- Native **DaVinci Wide Gamut / DaVinci Intermediate (DW/I)** linear-light integration and DWG CIE $Y$ luminance pipeline.
- Built-in Diagnostic Views (`Motion Vectors HSV`, `Occlusion Confidence`, `Trail Motion/Luma Mask`).
- Canonical bidirectional `FlowCache` (`1.5 GB` LRU), stream-keyed GPU scratch pool (`512 MB` cap), and graceful OFX `unload()` / Windows Loader-Lock safe teardown.
- Automated 8-test verification and 1080p/4K benchmark suite (`BetterMotionBlurTests.exe`).
- GNU GPLv3 (`GPL-3.0-or-later`) `LICENSE`, `THIRD_PARTY_NOTICES.md`, and project-level documentation (`docs/`).
