#pragma once

#include "CommonTypes.h"
#include "FlowCache.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace bmb {

/// Runtime-NVRTC CUDA compute engine for optical flow estimation, shutter blur, and long-exposure trails.
class CudaEngine {
public:
    static CudaEngine& instance();

    bool initialize();
    bool isAvailable() const { return m_initialized; }
    std::string deviceName() const { return m_deviceName; }
    std::string lastError() const { return m_lastError; }

    // Memory helpers (operates in current CUDA context or engine's context)
    uint64_t allocDevice(size_t bytes);
    void freeDevice(uint64_t dptr);
    bool uploadToDevice(uint64_t dstDev, const void* srcHost, size_t bytes, void* stream = nullptr);
    bool downloadFromDevice(void* dstHost, uint64_t srcDev, size_t bytes, void* stream = nullptr);
    bool copyDeviceToDevice(uint64_t dstDev, uint64_t srcDev, size_t bytes, void* stream = nullptr);
    bool memsetDevice(uint64_t dstDev, unsigned char val, size_t bytes, void* stream = nullptr);
    bool syncStream(void* stream = nullptr);

    // Computes or retrieves cached bidirectional optical flow + confidence for frame pair (frameA, frameB).
    // Returns a GPU-resident FlowEntry (devFlow4: fwdX, fwdY, bwdX, bwdY; devConf2: confidence, fgWeight).
    std::shared_ptr<FlowEntry> getOrComputeFlowPair(
        const void* instanceId,
        int frameAIndex, uint64_t devFrameA, uint64_t devMatteA, int matteAComp,
        int frameBIndex, uint64_t devFrameB, uint64_t devMatteB, int matteBComp,
        const RenderParams& params,
        FlowCache& cache,
        void* stream = nullptr);

    // Executes the full GPU rendering pipeline given frames [t - lookback .. t + 1] already on GPU.
    // `frames` must be ordered or indexed; `centerIdx` points to frame `t`.
    bool renderFrame(
        const void* instanceId,
        const std::vector<FrameInput>& frames,
        int centerIdx,
        uint64_t devOutputRGBA,
        const RenderParams& params,
        FlowCache& cache,
        void* stream = nullptr,
        const std::vector<FrameInput>* shutterSubFrames = nullptr);

    // Checks whether two GPU RGBA frames differ (used for pulldown/sub-frame probing).
    bool areFramesDistinct(
        uint64_t devFrameA,
        uint64_t devFrameB,
        int width,
        int height,
        void* stream = nullptr);

    // Convenience helper for standalone tests using host RGBA float buffers.
    bool renderHostFrames(
        const void* instanceId,
        const std::vector<FrameInput>& hostFrames,
        int centerIdx,
        float* hostOutputRGBA,
        const RenderParams& params,
        FlowCache& cache);

    // Downloads GPU FlowEntry (decoding FP16 flow if packed) into host Float4 vectors.
    bool downloadFlowToHost(
        const FlowEntry& entry,
        std::vector<Float4>& outHostFlow,
        void* stream = nullptr);

    void shutdown();

private:
    CudaEngine() = default;
    ~CudaEngine();
    CudaEngine(const CudaEngine&) = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;

    struct FlowPairResult {
        std::shared_ptr<FlowEntry> entry;
        bool reversed = false;
    };

    struct PyramidSet {
        static constexpr int kMaxLevels = 7;
        int numLevels = 0;
        int lw[kMaxLevels] = {};
        int lh[kMaxLevels] = {};
        uint64_t luma[kMaxLevels] = {};
        uint64_t matte[kMaxLevels] = {};
        bool hasMatte = false;
    };

    bool ensureContext();
    bool compileKernels();

    uint64_t acquireScratch(size_t bytes, void* stream);
    void releaseScratch(uint64_t dptr, size_t bytes, void* stream);
    void clearScratchPool();

    bool buildPyramid(
        uint64_t devRGBA, uint64_t devMatte, int matteComp, bool needMatte,
        int width, int height, PyramidSet& outPyr, void* stream);
    void releasePyramid(PyramidSet& pyr, void* stream);

    bool launchFlowFromPyramids(
        const PyramidSet& pyrA, const PyramidSet& pyrB,
        int width, int height,
        FlowEngineType engineType, float flowDetail, MatteMode matteMode,
        uint64_t devOutFlow2, void* stream);

    bool renderSingleShutterPass(
        const void* instanceId,
        const FrameInput* prevF,
        const FrameInput& currF,
        const FrameInput* nextF,
        PyramidSet* pyrPrev,
        PyramidSet* pyrCurr,
        PyramidSet* pyrNext,
        uint64_t devCurrMatte,
        int hasMatte,
        uint64_t targetBuf,
        const RenderParams& params,
        FlowCache& cache,
        void* stream);

    FlowPairResult getOrComputeFlowPairCanonical(
        const void* instanceId,
        int frameAIndex, uint64_t devFrameA, uint64_t devMatteA, int matteAComp,
        int frameBIndex, uint64_t devFrameB, uint64_t devMatteB, int matteBComp,
        const RenderParams& params,
        FlowCache& cache,
        void* stream,
        PyramidSet* callerPyrA = nullptr,
        PyramidSet* callerPyrB = nullptr,
        bool skipCacheLookup = false);

    std::mutex m_mutex;
    std::atomic<bool> m_alive{true};
    bool m_attemptedInit = false;
    bool m_initialized = false;
    int m_ccMajor = 7;
    int m_ccMinor = 5;
    std::string m_deviceName;
    std::string m_lastError;

    void* m_cudaDll = nullptr;
    void* m_nvrtcDll = nullptr;
    void* m_ownedCtx = nullptr;
    void* m_module = nullptr;

    // Stream-keyed GPU scratch buffer pool (768 MB cap)
    static constexpr size_t kMaxScratchPoolBytes = 768ULL * 1024ULL * 1024ULL;
    std::mutex m_scratchMutex;
    std::unordered_map<uintptr_t, std::unordered_map<size_t, std::vector<uint64_t>>> m_scratchPool;
    size_t m_scratchPoolBytes = 0;

    // Reusable host staging buffers for external ONNX model fallback
    std::mutex m_ortBufMutex;
    std::vector<Float2> m_ortFlowBuf;
    std::vector<Float2> m_ortGradBuf;
    std::vector<float> m_ortInNchw4;
    std::vector<float> m_ortOutNchw2;

    // Kernel function handles
    void* m_fnExtractLumaMatte = nullptr;
    void* m_fnDownsample2x = nullptr;
    void* m_fnInitZeroFlow = nullptr;
    void* m_fnUpsampleFlow2x = nullptr;
    void* m_fnPatchMatchLK = nullptr;
    void* m_fnCorrelationFlow = nullptr;
    void* m_fnEmbeddedResidualRefine = nullptr;
    void* m_fnJointBilateralFilter = nullptr;
    void* m_fnJointBilateralUpsample = nullptr;
    void* m_fnPackBidirectional = nullptr;
    void* m_fnCompareFrames = nullptr;
    void* m_fnTileMaxVel = nullptr;
    void* m_fnNeighborMaxVelH = nullptr;
    void* m_fnNeighborMaxVelV = nullptr;
    void* m_fnShutterBlur = nullptr;
    void* m_fnAccumLinearFromDwi = nullptr;
    void* m_fnEncodeLinearToDwi = nullptr;
    void* m_fnTrailPairInitStatic = nullptr;
    void* m_fnTrailPairSplatMoving = nullptr;
    void* m_fnTrailCompositeAndFinalize = nullptr;
    void* m_fnRenderDiagnostic = nullptr;
};

} // namespace bmb
