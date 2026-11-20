# 01 — OFX Plugin Interface & Parameters

Primary files: [BetterMotionBlurPlugin.h](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/BetterMotionBlurPlugin.h), [BetterMotionBlurPlugin.cpp](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/BetterMotionBlurPlugin.cpp), [CommonTypes.h](file:///c:/Users/pan_y/Documents/Better%20Motion%20Blur/src/CommonTypes.h)

---

## OFX Contexts & Clips
- **Plugin Identifier**: `com.BetterMotionBlur.OFX.BetterMotionBlur` (v1.0, Grouping: `"Better Motion Blur"`).
- **Supported Contexts**: `eContextFilter` and `eContextGeneral`.
- **Supported Bit Depth**: `eBitDepthFloat` (`RGBA` 32-bit float).
- **CUDA Support**: `setSupportsCudaRender(true)` and `setSupportsCudaStream(true)` enabled on non-Apple builds.
- **Clips**:
  - `Source` (`kOfxImageEffectSimpleSourceClipName`): Required `RGBA` float clip with `setTemporalClipAccess(true)`.
  - `Matte` (`"Matte"`, `eContextGeneral` only): Optional `Alpha` or `RGBA` mask clip with `setTemporalClipAccess(true)` and `setIsMask(true)`.
  - `Output` (`kOfxImageEffectOutputClipName`): `RGBA` float output clip.

---

## Temporal Frame Negotiation & Early Exit
- **`getFramesNeeded()`**:
  - `lookback = 1` for `Natural Shutter Blur`; `lookback = max(1, trailFrames)` for `Long-Exposure Trail` and `Combined`.
  - Requests `[time - lookback, time + 1.0]` when future frame `t+1` is needed (`Natural Shutter Blur`, `Combined`, or any non-Final `Diagnostic View`), and `[time - lookback, time]` in `Long-Exposure Trail` (`Final` view) so Resolve does not render an unused future frame.
- **`isIdentity()`**:
  - Returns `true` (`Source` at `time`) when `viewMode == Final`, `mode == Natural Shutter Blur`, and either `shutterAngle <= 0.001` or `motionSensitivity <= 0.0001`.

---

## Parameter Reference

| Script Name | UI Label | Type / Range | Default | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| `mode` | Render Mode | Choice (`0..2`) | `0` (Natural Shutter Blur) | `0`: Shutter Blur, `1`: Long-Exposure Trail, `2`: Combined. |
| `viewMode` | Diagnostic View | Choice (`0..3`) | `0` (Final Output) | `0`: Final, `1`: Motion Vectors (HSV), `2`: Occlusion Confidence, `3`: Trail Motion/Luma Mask. |
| `flowEngine` | Flow Engine | Choice (`0..1`) | `0` (Neural Optical Flow) | `0`: Proxy correlation + neural refinement + JBU, `1`: Full-res CUDA PatchMatch + LK. |
| `flowDetail` | Edge Preservation | Double (`0.0–1.0`) | `0.75` | Joint-bilateral range stopping (`sigmaRange = 0.03 + (1 - flowDetail) * 0.12`). Higher values prevent edge bleed. |
| `motionSensitivity` | Motion Sensitivity | Double (`0.0–4.0`) | `1.0` | Global multiplier applied to estimated optical flow vectors. |
| `motionThreshold` | Motion Threshold (px) | Double (`0.0–10.0`) | `0.5` | Smoothstep velocity gate below which static/noisy pixels do not blur or trail. |
| `matteMode` | Matte Handling | Choice (`0..2`) | `1` (Foreground Isolation) | `0`: Ignore Matte, `1`: Foreground Isolation (blocks bg/fg bleed), `2`: Mask Effect. |
| `shutterAngle` | Shutter Angle (deg) | Double (`0.0–720.0`) | `180.0` | Virtual shutter angle (`180°` = 0.5 frame span). |
| `shutterPhase` | Shutter Phase (deg) | Double (`-180.0–180.0`) | `0.0` | Temporal center offset of the shutter window. |
| `shutterSamples` | Shutter Samples | Int (`8–64`) | `24` | Sub-frame trajectory integration samples per pixel. |
| `trailFrames` | Trail Length (Frames) | Int (`2–24`) | `8` | Lookback frame window for long-exposure accumulation. |
| `subFrameSteps` | Sub-Frame Steps | Int (`1–16`) | `6` | Minimum flow-bridged steps between frames (adaptively scales up to 72 for fast motion). |
| `trailBlendMode` | Trail Blend Mode | Choice (`0..2`) | `0` (Physical Exposure) | `0`: Physical Linear Exposure, `1`: Light-Painting Max/Screen, `2`: Comet / Rear-Curtain Decay. |
| `trailDecay` | Trail Decay | Double (`0.0–1.0`) | `0.35` | Exponential temporal falloff rate in Comet and Sequential modes. |
| `rearCurtainCrispness` | Rear-Curtain Crispness | Double (`0.0–1.0`) | `0.25` | Blends sharp current-frame subject over trailing wake in Comet mode. |
| `lumaThreshold` | Luma Threshold | Double (`0.0–1.0`) | `0.0` | Perceptual DW/I luma threshold restricting Light-Painting trails to highlights. |
| `sequentialCache` | Sequential Accumulation Cache | Bool | `false` | Carries feedback forward across sequential frames (`frame == lastFrame + 1`). |
| `highlightBoost` | Highlight Streak Boost | Double (`0.0–5.0`) | `1.5` | Linear-light DWG highlight weight (`luma > 0.80`) for realistic specular streaks. |

---

## Rendering Dispatch (`BetterMotionBlurProcessor`)
- **GPU Path (`processImagesCUDA`)**: Passes Resolve's CUDA device pointers (`getPixelData()`) and `_pCudaStream` directly into `CudaEngine::instance().renderFrame(...)` with zero host pixel copies.
- **CPU Path (`preProcess` + `multiThreadProcessImages`)**: Packs source/matte planes into contiguous float buffers and invokes `CpuFallbackProcessor::renderFrame(...)`, then copies scanlines to `_dstImg`.
