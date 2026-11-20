#pragma once

#include <cstdint>

namespace bmb {

/// Synthesis mode selected by the user.
enum EffectMode {
    eModeShutterBlur = 0,
    eModeLongExposureTrail = 1,
    eModeCombined = 2
};

enum FlowEngineType {
    eFlowNeural = 0,
    eFlowCudaPyramidal = 1
};

enum TrailBlendMode {
    eTrailPhysicalExposure = 0,
    eTrailLightPainting = 1,
    eTrailCometDecay = 2
};

enum MatteMode {
    eMatteIgnore = 0,
    eMatteForegroundIsolation = 1,
    eMatteMaskEffect = 2
};

enum DiagnosticView {
    eViewFinal = 0,
    eViewFlowHSV = 1,
    eViewConfidence = 2,
    eViewTrailMask = 3
};

struct RenderParams {
    int width = 0;
    int height = 0;

    EffectMode mode = eModeShutterBlur;
    FlowEngineType flowEngine = eFlowNeural;
    DiagnosticView viewMode = eViewFinal;
    MatteMode matteMode = eMatteForegroundIsolation;

    float flowDetail = 0.7f;         // 0.0 - 1.0
    float motionSensitivity = 1.0f;  // 0.0 - 4.0
    float motionThreshold = 0.5f;    // 0.0 - 10.0 px

    float shutterAngle = 180.0f;     // 0.0 - 720.0 deg
    float shutterPhase = 0.0f;       // -180.0 - +180.0 deg
    int shutterSamples = 24;         // 8 - 64

    int trailFrames = 8;             // 2 - 24
    int subFrameSteps = 6;           // 1 - 16
    TrailBlendMode trailBlendMode = eTrailPhysicalExposure;
    float trailDecay = 0.35f;        // 0.0 - 1.0
    float rearCurtainCrispness = 0.25f; // 0.0 - 1.0
    float lumaThreshold = 0.0f;      // 0.0 - 1.0
    bool sequentialCache = false;

    float highlightBoost = 1.5f;     // 0.0 - 5.0
};

struct Float4 {
    float x, y, z, w;
};

struct Float2 {
    float x, y;
};

struct FlowPixel {
    float fwdX;
    float fwdY;
    float bwdX;
    float bwdY;
    float confidence;
    float fgWeight;
};

struct FrameInput {
    int frameIndex = 0;
    const float* hostRGBA = nullptr;
    uint64_t deviceRGBA = 0; // CUdeviceptr when running on GPU
    const float* hostMatte = nullptr;
    uint64_t deviceMatte = 0; // CUdeviceptr when running on GPU
    int matteComponents = 0; // 1 = Alpha, 4 = RGBA
};

} // namespace bmb
