<!-- doc-type: product-implementation -->
# 03 — Shutter Blur, Long-Exposure Trails, Color & Diagnostics

Primary files: [CudaEngine.cpp](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/CudaEngine.cpp), [CpuFallbackProcessor.cpp](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/CpuFallbackProcessor.cpp)

---

## 1. Native DaVinci Wide Gamut / Intermediate (DW/I) Color Math
All pixel processing assumes incoming and outgoing images are in **DaVinci Wide Gamut / DaVinci Intermediate (DW/I)**:
- **DaVinci Intermediate Constants**:
  - `DI_A = 0.0075f`, `DI_B = 7.0f`, `DI_C = 0.07329248f`, `DI_M = 10.44426855f`
  - `DI_LIN_CUT = 0.00262409f`, `DI_LOG_CUT = 0.02740668f`
- **Transfer Functions**:
  - `dwiToLinearChannel(v)`: `(v <= DI_LOG_CUT) ? (v / DI_M) : (exp2f(v / DI_C - DI_B) - DI_A)`
  - `linearToDwiChannel(l)`: `(l <= DI_LIN_CUT) ? (l * DI_M) : ((log2f(l + DI_A) + DI_B) * DI_C)`
- **Luminance (`dwgLuminance`)**:
  - Uses official DaVinci Wide Gamut CIE $Y$ row: `fmaxf(0.0f, 0.27411851f * r + 0.87363190f * g - 0.14775041f * b)`.
  - Optical flow luma, joint-bilateral edge stopping, `Luma Threshold` (`0.0–1.0`), and `Trail Motion/Luma Mask` operate in perceptual DW/I space.
  - Temporal accumulation (shutter blur, trail splatting, rear-curtain blend, sequential feedback) operates in scene-linear DWG light, with `Highlight Streak Boost` activating above diffuse white (`luma > 0.80f` linear DWG).

---

## 2. Natural Shutter Blur (`synthesizeShutterBlurKernel`)
1. **Confidence-Weighted Bidirectional Velocity & Separable Dilation (`tileMaxVelKernel` + `neighborMaxVelHKernel` / `neighborMaxVelVKernel`)**:
   - `evalConfWeightedVelAcc` weights `vf` and `-vb` by `conf^2 + 0.02` (preserving true speed under single-sided temporal occlusion instead of halving) and scales curvature `a0` by `min(cf, cb)^2`.
   - `tileMaxVelKernel` launches one `16×16` coalesced block per `16×16` tile and performs a shared-memory parallel max reduction on gated `(v0, a0)` (`float4`).
   - `neighborMaxVelHKernel` and `neighborMaxVelVKernel` perform separable 1D dilation across dynamic `tileRadius = clamp(ceil(0.14 * width * maxHalfSpan / 16), 1, 12)` (`up to 192 px` half-blur reach at 4K).
2. **Inverse Quadratic Trajectory & Multi-Frame Disocclusion**:
   - When `autoShutterSamples` is enabled (`true` by default), computes a tile-uniform sample count `effSamples = clamp(ceil(max(centerSpeed, domSpeed) * shutterSpan * 1.25), min(samples, 8), samples)` from `neighborMax` and local velocity (zero intra-warp divergence).
   - Integrates `effSamples` jittered taps across `[phaseCenter - 0.5 * shutterSpan, phaseCenter + 0.5 * shutterSpan]` using inverse gather displacement `-v * tau - 0.5 * a * tau^2`:
     - Exterior / static-background taps (`useDom`) trace along `(-vDom, -aDom)` and test curved arrival `tauStar` and perpendicular miss distance `perpDist`; foreground hits accumulate with Jacobian weight `clamp(domSpeed / max(sSpeed, 1), 1, 2.5)` capped to `fgFrac = clamp(fgTime / effSamples, 0, 1)`, while misses accumulate unoccluded static `centerLin` (`bgFrac = 1 - fgFrac`) for exact linear `50% -> 0%` silhouette ramps.
     - Interior moving pixels stepping off the silhouette check zero-cost reverse flow (`vAtPrev`, `vAtNext`) / neighbor mattes at `(x, y)` to sample true disoccluded background from `prevRGBA` (`t-1`) or `nextRGBA` (`t+1`) when uncovered, and suppress static foreground occluders (`0.0` with matte, `0.05` without matte).
   - When `shutterSubFrames` (`>= 3` distinct fractional sub-frames) are provided, `renderFrame()` runs piecewise Voronoi-weighted `renderSingleShutterPass()` segments and blends them in linear DWG via `accumLinearFromDwiKernel` + `encodeLinearToDwiKernel`.

---

## 3. Long-Exposure Motion Trails (`trailPairInitStaticKernel`, `trailPairSplatMovingKernel` & `trailCompositeAndFinalizeKernel`)
Processes frame pairs `(idxA = centerIdx - k, idxB = idxA - 1)` for `k = 0 .. maxLookback - 1` (`maxLookback = min(centerIdx, max(1, trailFrames - 1))`) plus the oldest endpoint frame `centerIdx - maxLookback`:
- **2-Pass Tile-Aware Shared-Memory Splatting**:
  - **`trailPairInitStaticKernel`**: Non-atomically writes static/masked pixels with coalesced `float4` stores (skipping global zeroing), reads 1- or 4-channel mattes inline (`readMatteVal`), suppresses non-foreground background emission when `matteMode == 1` (`Foreground Isolation`), early-rejects gated Light-Painting pixels on `speed <= motionThreshold` before loading `frameA`, skips unused `log2f`/`powf`, scales Comet decay by `pairTimeStep`, and outputs per-`16×16`-tile `tileActive` flags via `__syncthreads_or`.
  - **`trailPairSplatMovingKernel`**: Skips static tiles immediately; gates moving emission by `matteA` when `matteMode == 1`; fits a time-invariant confidence-gated quadratic curve $\mathbf{p}(\alpha)$ from `pairFlows[k+1]` ($t_{k-1} \to t_{k-2}$, preserving `TrailSplatCache` invariance with zero extra flow passes); reduces the exact quadratic trajectory bounding box via warp shuffles; for `boxArea <= 1536` (`24 KB` bank-conflict-free SoA `sR/sG/sB/sW`), splats in `__shared__` memory using register cell-coalescing, multiplicative exponential decay (`timeW *= decayMul` scaled by `pairTimeStep`), and single-instruction hardware `atomicMax` (`atomicMaxPosFloat`) before flushing non-zero pixels in coalesced row order.
- **Per-Pair Factorization & Fused Finalize (`trailCompositeAndFinalizeKernel`)**:
  - Every pair `(f, f-1)` is cached at `pairAgeIndex = 0` (`TrailSplatCache`) and composited in one pass with base `currRGBA`, oldest endpoint frame `centerIdx - maxLookback`, optional low-VRAM `uncachedAccum`, and `eModeCombined` `k == 0` shutter-blurred pair (`devPair0`):
    - `0 — Physical Exposure (Linear)`: Linear integration weighted by `hdrW` (`scale = 1.0`). When `matteMode == 1` (`Foreground Isolation`), composites accumulated foreground exposure over unoccluded background (`bgW = max(0, totalWindowW - acc.w)`) so shaky/moving backgrounds stay crisp.
    - `1 — Light-Painting (Max/Screen)`: Peak-envelope `fmaxf` compositing with `motionGate * lumaGate` (`scale = 1.0`).
    - `2 — Comet / Rear-Curtain Decay`: Applies exact scalar factorization `exp(-decayLambda * k * pairTimeStep)` per pair, backfills vacated low-weight wakes (`acc.w < 0.25`) from `lastRGBA`, and blends sharp current frame via `rearCurtainCrispness`.
- **Fused Linear Sequential Accumulation Cache**: When `sequentialCache == true` and playback advances sequentially (`curr.frameIndex == lastFrame + 1`), `trailCompositeAndFinalizeKernel` blends the prior frame's linear DWG buffer (`seq->devAccumRGBA`) directly before DW/I re-encoding with zero extra kernel launches or device-to-device copies, and skips re-decaying when `isDuplicateFrame == true`.

---

## 4. Diagnostic Views (`renderDiagnosticViewKernel`)
- `1 — Motion Vectors (HSV)`: Maps flow angle to hue and `mag / 24.0` to brightness.
- `2 — Occlusion Confidence`: Green = high bidirectional consistency (`confFwd` or `confBwd` matching `flowRev`), Red = occluded/disoccluded.
- `3 — Trail Motion/Luma Mask`: Previews `motionGate * lumaGate * matte` pixels that emit light-painting trails.
