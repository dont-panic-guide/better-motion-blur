<!-- doc-type: product-features -->
# 02 — Optical Flow Engine & Caching Architecture

Primary files: [CudaEngine.h](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/CudaEngine.h), [CudaEngine.cpp](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/CudaEngine.cpp), [NeuralFlowEngine.h](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/NeuralFlowEngine.h), [NeuralFlowEngine.cpp](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/NeuralFlowEngine.cpp), [OnnxModelGenerator.h](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/OnnxModelGenerator.h), [FlowCache.h](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/FlowCache.h), [FlowCache.cpp](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/FlowCache.cpp)

---

## 1. Shared Pyramids & Conditional Matte Processing
- `CudaEngine::buildPyramid()` extracts perceptual DW/I luma once at level 0 via `extractLumaAndMatteKernel` (`__ldg` vectorized `float4` reads) and builds up to 5 levels (`>= 16×16`) via `downsample2xKernel`.
- Matte pyramids are allocated and downsampled **only** when `matteMode == eMatteForegroundIsolation` and a valid matte buffer is attached (`needMatte == true`).
- `getOrComputeFlowPairCanonical()` builds `pyrLow` and `pyrHigh` **once** per frame pair and invokes `launchFlowFromPyramids()` for both `low -> high` and `high -> low`.

---

## 2. Coarse-to-Fine PatchMatch + Lucas-Kanade (`patchMatchLKFlowKernel`)
Executed per pyramid level from coarsest (`numLevels - 1`) down to `finestLevel` (`0` for `eFlowCudaPyramidal`, proxy `~960px` width for `eFlowNeural`):
- **Shared-Memory Tiling**: Each `16×16` block cooperatively loads a `22×22` `lumaA` tile (and `20×20` `matteA` tile when `hasMatte == 1`) into `__shared__` memory before `__syncthreads()`.
- **Candidate Evaluation (`evalPatchCostShared`)**:
  - Evaluates initial upsampled flow, `(0, 0)` static snap, 4 spatial neighbors `{(-2,0), (2,0), (0,-2), (0,2)}` (skipping duplicates), and local `searchRadius` (`5`/`3` at coarsest, `1` at `lvl >= 2`, `0` at fine levels).
  - Uses an integer-displacement fast path (`1` `__ldg` read instead of `4`-tap bilinear) and exact branch-and-bound early exit when `!hasMatte`.
  - Out-of-bounds target coordinates `(ax + vx, ay + vy)` incur a `0.25` penalty rather than clamping borders.
- **Neighbor Cleanup & Regularization**: At `lvl >= 2`, runs a second zero-radius neighbor-propagation cleanup pass (`lkIters = 2`) to eliminate isolated coarse outliers, followed by `jointBilateralFlowFilterKernel` (`20×20` shared-memory tile).
- **Sub-Pixel Iterative LK**: Computes `valA`, `gx`, `gy`, and weights once per pixel during structure-tensor assembly and reuses them across `lkIterations` (`4` coarse, `5` fine) with Charbonnier/Huber weighting and backtracking cost guard.

---

## 3. Neural Proxy Flow (`eFlowNeural`) & Upsampling
At `finestLevel` (`lvl = 2` for `>= 2560px`, `lvl = 1` for `>= 1024px`, `lvl = 0` for `< 1024px`):
1. **`correlationFlowKernel`**: Preloads a `3×3` `lumaA` patch into registers and performs a `5×5` half-pixel correlation search with exact branch-and-bound early exit.
2. **Residual Flow Refinement**:
   - **GPU-Native Fast Path (Default)**: When no external `neuflow_v2.onnx` file is loaded (`!NeuralFlowEngine::instance().isUsingExternalModel()`), `embeddedResidualRefineKernel` evaluates `refined = flow + 0.08 * Laplacian_3x3(flow)` directly in an `18×18` shared-memory tile — algebraically identical to `buildResidualFlowRefinementOnnx()` with zero PCIe transfer or stream sync.
   - **External ONNX Path**: If `Contents/Resources/neuflow_v2.onnx` or `Contents/Win64/models/neuflow_v2.onnx` exists, `NeuralFlowEngine` loads Resolve's `onnxruntime.dll` (ORT 1.13.1 C API vtable slots) and runs `refineFlow` using persistent host staging buffers (`m_ortFlowBuf`, `m_ortGradBuf`, `m_ortInNchw4`, `m_ortOutNchw2`).
3. **4K Guided Upsampling (`jointBilateralUpsampleKernel`)**: Upsamples proxy flow to full resolution guided by full-res `lumaA[0]` and `matteA[0]`.

---

## 4. Canonical Bidirectional `FlowCache` & Scratch Buffer Pool
- **Canonical Key Ordering**: `getOrComputeFlowPairCanonical()` orders `(frameLow = min(A, B), frameHigh = max(A, B))`.
  - `packBidirectionalKernel` stores `flow4 = (fwd.x, fwd.y, bwd.x, bwd.y)` and `conf2 = (confFwd, confBwd)`.
  - Consumers receive `reversed = (frameAIndex > frameBIndex)` and read `.xy`/`.x` when `0` or `.zw`/`.y` when `1`.
  - During sequential playback (`t -> t+1`), `t -> t-1` hits the `(t-1, t)` entry computed on the previous frame (~50% reduction in flow compute).
- **LRU Budget**: `FlowCache::global()` defaults to a `1.5 GB` LRU budget and stores per-instance `SequentialTrailState`.
- **Stream-Keyed Scratch Pool**: `CudaEngine::acquireScratch(bytes, stream)` / `releaseScratch(ptr, bytes, stream)` caches temporary GPU allocations per CUDA stream (`512 MB` cap) to avoid synchronous WDDM `cuMemAlloc`/`cuMemFree` stalls without cross-stream hazards.
