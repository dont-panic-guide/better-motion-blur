<!-- doc-type: product-features -->
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
1. **Velocity Dilation (`tileMaxVelKernel` + `neighborMaxVelKernel`)**:
   - `tileMaxVelKernel` launches one `16×16` coalesced block per `16×16` tile and performs a shared-memory parallel max reduction on gated center velocity `v0 = 0.5 * (vf - vb) * sensitivity`.
   - `neighborMaxVelKernel` takes the `3×3` tile maximum (`vDom`), allowing fast-moving foreground edges to blur cleanly onto static background pixels.
2. **Quadratic Sub-Frame Trajectory**:
   - Computes velocity `v0 = 0.5 * (vf - vb)` and curvature acceleration `a0 = vf + vb`.
   - Integrates `shutterSamples` jittered taps across `[phaseCenter - 0.5 * shutterSpan, phaseCenter + 0.5 * shutterSpan]`:
     - Static-background pixels adjacent to fast foreground trace along `vDom * tau`.
     - Moving pixels trace along `v0 * tau + 0.5 * a0 * tau^2`.
3. **Foreground Isolation & Depth Ordering**:
   - With `matteMode == eMatteForegroundIsolation`: foreground samples reject static background taps (zero background bleed); background pixels accept foreground taps whose blur reach covers `(x, y)`.
   - Without matte: attenuates static background bleed into fast foreground (`wSample = 0.05f` when `centerSpeed > 1.5 && sSpeed < 0.35 * centerSpeed`).
   - Uses vectorized `float4` `__ldg` loads/stores and defers `flowFwd[sIdx]` / `flowBwd[sIdx]` reads until required by the active branch.

---

## 3. Long-Exposure Motion Trails (`trailSplatSubFrameKernel` & `trailFinalizeKernel`)
Iterates over frame pairs `(idxA = centerIdx - k, idxB = idxA - 1)` for `k = 0 .. min(centerIdx, max(1, trailFrames - 1))`:
- **Gapless Sub-Frame Bridging**: For moving pixels (`speed >= 0.35f`, `subFrameSteps > 1`), computes `adaptiveSteps = max(subFrameSteps, min(72, ceil(speed * 1.25)))` and bilinearly splats crossfaded linear RGB (`rA -> pB` weighted by `alpha * conf`) onto `accumRGBW` via `atomicAdd` / `atomicMaxFloat`.
- **Blend Modes**:
  - `0 — Physical Exposure (Linear)`: Energy-conserving linear integration weighted by `hdrW`.
  - `1 — Light-Painting (Max/Screen)`: Early-rejects pixels failing `motionGate * lumaGate <= 0.01f` before sampling `frameB`, then splats peak envelopes via `atomicMaxFloat`.
  - `2 — Comet / Rear-Curtain Decay`: Applies exponential decay `exp(-decayLambda * (k + alpha))` and blends sharp current frame at output via `rearCurtainCrispness`.
- **Sequential Accumulation Cache**: When `sequentialCache == true` and playback advances sequentially (`curr.frameIndex == lastFrame + 1`), `applySeqFeedbackKernel` blends the prior frame's output in linear DWG light.

---

## 4. Diagnostic Views (`renderDiagnosticViewKernel`)
- `1 — Motion Vectors (HSV)`: Maps flow angle to hue and `mag / 24.0` to brightness.
- `2 — Occlusion Confidence`: Green = high bidirectional consistency (`confFwd` or `confBwd` matching `flowRev`), Red = occluded/disoccluded.
- `3 — Trail Motion/Luma Mask`: Previews `motionGate * lumaGate * matte` pixels that emit light-painting trails.
