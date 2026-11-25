#pragma once

#include "CommonTypes.h"
#include "FlowCache.h"

#include <memory>
#include <vector>

namespace bmb {

/// Multithreaded CPU fallback mirror of the CUDA optical flow and synthesis pipeline.
class CpuFallbackProcessor {
public:
    static std::shared_ptr<FlowEntry> getOrComputeFlowPair(
        const void* instanceId,
        int frameAIndex, const float* rgbaA, const float* matteA, int matteAComp,
        int frameBIndex, const float* rgbaB, const float* matteB, int matteBComp,
        const RenderParams& params,
        FlowCache& cache);

    static bool renderFrame(
        const void* instanceId,
        const std::vector<FrameInput>& frames,
        int centerIdx,
        float* dstRGBA,
        const RenderParams& params,
        FlowCache& cache,
        const std::vector<FrameInput>* shutterSubFrames = nullptr);

    static bool areFramesDistinct(
        const float* rgbaA,
        const float* rgbaB,
        int width,
        int height);
};

} // namespace bmb
