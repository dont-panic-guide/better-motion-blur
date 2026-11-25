#include "BetterMotionBlurPlugin.h"
#include "CommonTypes.h"
#include "CpuFallbackProcessor.h"
#include "CudaEngine.h"
#include "FlowCache.h"
#include "NeuralFlowEngine.h"

#include "ofxsImageEffect.h"
#include "ofxsLog.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#define kPluginName "Better Motion Blur"
#define kPluginGrouping "Better Motion Blur"
#define kPluginDescription "Neural & CUDA optical-flow natural shutter motion blur and gapless long-exposure motion trails."
#define kPluginIdentifier "com.BetterMotionBlur.OFX.BetterMotionBlur"
#define kPluginVersionMajor 1
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

#define kClipMatteName "Matte"

namespace {

// Parameter script names
constexpr const char* kParamMode = "mode";
constexpr const char* kParamViewMode = "viewMode";
constexpr const char* kParamFlowEngine = "flowEngine";
constexpr const char* kParamFlowDetail = "flowDetail";
constexpr const char* kParamMotionSensitivity = "motionSensitivity";
constexpr const char* kParamMotionThreshold = "motionThreshold";
constexpr const char* kParamMatteMode = "matteMode";
constexpr const char* kParamShutterAngle = "shutterAngle";
constexpr const char* kParamShutterPhase = "shutterPhase";
constexpr const char* kParamShutterTimebase = "shutterTimebase";
constexpr const char* kParamAutoShutterSamples = "autoShutterSamples";
constexpr const char* kParamShutterSamples = "shutterSamples";
constexpr const char* kParamTrailFrames = "trailFrames";
constexpr const char* kParamTrailTimebase = "trailTimebase";
constexpr const char* kParamSubFrameSteps = "subFrameSteps";
constexpr const char* kParamTrailBlendMode = "trailBlendMode";
constexpr const char* kParamTrailDecay = "trailDecay";
constexpr const char* kParamRearCurtainCrispness = "rearCurtainCrispness";
constexpr const char* kParamLumaThreshold = "lumaThreshold";
constexpr const char* kParamSequentialCache = "sequentialCache";
constexpr const char* kParamHighlightBoost = "highlightBoost";

constexpr const char* kGroupShutter = "groupShutter";
constexpr const char* kGroupTrail = "groupTrail";
constexpr const char* kGroupPhoto = "groupPhoto";

} // namespace

class BetterMotionBlurProcessor : public OFX::ImageProcessor
{
public:
    explicit BetterMotionBlurProcessor(OFX::ImageEffect& p_Instance)
        : OFX::ImageProcessor(p_Instance)
    {}

    void setInputFrames(const void* instanceId,
                        std::vector<std::unique_ptr<OFX::Image>> srcImages,
                        std::vector<std::unique_ptr<OFX::Image>> matteImages,
                        std::vector<int> frameIndices,
                        int centerIdx,
                        const bmb::RenderParams& params,
                        bool probePulldown = false,
                        std::vector<std::unique_ptr<OFX::Image>> subImages = {},
                        std::vector<std::unique_ptr<OFX::Image>> subMatteImages = {},
                        std::vector<int> subFrameIndices = {},
                        std::vector<float> subRelTimes = {},
                        std::vector<std::unique_ptr<OFX::Image>> trailSubImages = {},
                        std::vector<std::unique_ptr<OFX::Image>> trailSubMatteImages = {},
                        std::vector<int> trailSubFrameIndices = {},
                        float trailPairTimeStep = 1.0f)
    {
        m_instanceId = instanceId;
        m_srcImages = std::move(srcImages);
        m_matteImages = std::move(matteImages);
        m_frameIndices = std::move(frameIndices);
        m_centerIdx = centerIdx;
        m_params = params;
        m_probePulldown = probePulldown;
        m_subImages = std::move(subImages);
        m_subMatteImages = std::move(subMatteImages);
        m_subFrameIndices = std::move(subFrameIndices);
        m_subRelTimes = std::move(subRelTimes);
        m_trailSubImages = std::move(trailSubImages);
        m_trailSubMatteImages = std::move(trailSubMatteImages);
        m_trailSubFrameIndices = std::move(trailSubFrameIndices);
        m_trailPairTimeStep = trailPairTimeStep;
    }

    virtual void processImagesCUDA() override
    {
        if (m_centerIdx < 0 || m_centerIdx >= static_cast<int>(m_srcImages.size()) || !m_srcImages[m_centerIdx] || !_dstImg) {
            return;
        }

        const OfxRectI& bounds = m_srcImages[m_centerIdx]->getBounds();
        m_params.width = bounds.x2 - bounds.x1;
        m_params.height = bounds.y2 - bounds.y1;
        if (m_params.width <= 0 || m_params.height <= 0) return;

        auto makeGpuFrame = [](const OFX::Image* img, const OFX::Image* mImg, int frameIdx, float relT) {
            bmb::FrameInput f;
            f.frameIndex = frameIdx;
            f.relTime = relT;
            f.deviceRGBA = reinterpret_cast<uint64_t>(img->getPixelData());
            if (mImg) {
                f.deviceMatte = reinterpret_cast<uint64_t>(mImg->getPixelData());
                f.matteComponents = (mImg->getPixelComponents() == OFX::ePixelComponentAlpha) ? 1 : 4;
            }
            return f;
        };

        std::vector<bmb::FrameInput> gpuFrames(m_srcImages.size());
        for (size_t i = 0; i < m_srcImages.size(); ++i) {
            const OFX::Image* mImg = (i < m_matteImages.size()) ? m_matteImages[i].get() : nullptr;
            gpuFrames[i] = makeGpuFrame(m_srcImages[i].get(), mImg, m_frameIndices[i], 0.0f);
        }

        std::vector<bmb::FrameInput> distinctSubFrames;
        const std::vector<bmb::FrameInput>* subPtr = nullptr;
        if (m_subImages.size() >= 3) {
            for (size_t i = 0; i < m_subImages.size(); ++i) {
                if (!m_subImages[i]) continue;
                const OFX::Image* mImg = (i < m_subMatteImages.size()) ? m_subMatteImages[i].get() : nullptr;
                bmb::FrameInput sf = makeGpuFrame(m_subImages[i].get(), mImg, m_subFrameIndices[i], m_subRelTimes[i]);
                if (distinctSubFrames.empty() ||
                    bmb::CudaEngine::instance().areFramesDistinct(
                        distinctSubFrames.back().deviceRGBA, sf.deviceRGBA, m_params.width, m_params.height, _pCudaStream)) {
                    distinctSubFrames.push_back(sf);
                }
            }
            if (distinctSubFrames.size() >= 3) {
                subPtr = &distinctSubFrames;
            }
        }

        std::vector<bmb::FrameInput> distinctTrailSub;
        const std::vector<bmb::FrameInput>* renderFramesPtr = &gpuFrames;
        int renderCenterIdx = m_centerIdx;

        if (m_trailSubImages.size() >= 2 && (m_params.mode == bmb::eModeLongExposureTrail || m_params.mode == bmb::eModeCombined)) {
            for (size_t i = 0; i < m_trailSubImages.size(); ++i) {
                if (!m_trailSubImages[i]) continue;
                const OFX::Image* mImg = (i < m_trailSubMatteImages.size()) ? m_trailSubMatteImages[i].get() : nullptr;
                bmb::FrameInput tf = makeGpuFrame(m_trailSubImages[i].get(), mImg, m_trailSubFrameIndices[i], 0.0f);
                if (distinctTrailSub.empty() ||
                    bmb::CudaEngine::instance().areFramesDistinct(
                        distinctTrailSub.back().deviceRGBA, tf.deviceRGBA, m_params.width, m_params.height, _pCudaStream)) {
                    distinctTrailSub.push_back(tf);
                }
            }
            if (distinctTrailSub.size() >= 2) {
                renderCenterIdx = static_cast<int>(distinctTrailSub.size()) - 1;
                if (m_params.mode == bmb::eModeCombined && m_centerIdx + 1 < static_cast<int>(gpuFrames.size())) {
                    distinctTrailSub.push_back(gpuFrames[m_centerIdx + 1]);
                }
                m_params.pairTimeStep = m_trailPairTimeStep;
                m_params.trailFrames = renderCenterIdx + 1;
                renderFramesPtr = &distinctTrailSub;
            }
        }

        std::vector<bmb::FrameInput> pulldownFrames;
        if (renderFramesPtr == &gpuFrames && m_probePulldown && gpuFrames.size() > 2) {
            if (m_centerIdx > 0 &&
                !bmb::CudaEngine::instance().areFramesDistinct(
                    gpuFrames[m_centerIdx].deviceRGBA, gpuFrames[m_centerIdx - 1].deviceRGBA, m_params.width, m_params.height, _pCudaStream)) {
                m_params.isDuplicateFrame = true;
            }
            if (m_params.mode == bmb::eModeShutterBlur) {
                if (!subPtr && gpuFrames.size() > 3) {
                    int prevIdx = -1;
                    for (int i = m_centerIdx - 1; i >= 0; --i) {
                        if (bmb::CudaEngine::instance().areFramesDistinct(
                                gpuFrames[m_centerIdx].deviceRGBA, gpuFrames[i].deviceRGBA, m_params.width, m_params.height, _pCudaStream)) {
                            prevIdx = i;
                            m_params.bwdFrameStep = static_cast<float>(m_centerIdx - i);
                            break;
                        }
                    }
                    int nextIdx = -1;
                    for (int i = m_centerIdx + 1; i < static_cast<int>(gpuFrames.size()); ++i) {
                        if (bmb::CudaEngine::instance().areFramesDistinct(
                                gpuFrames[m_centerIdx].deviceRGBA, gpuFrames[i].deviceRGBA, m_params.width, m_params.height, _pCudaStream)) {
                            nextIdx = i;
                            m_params.fwdFrameStep = static_cast<float>(i - m_centerIdx);
                            break;
                        }
                    }
                    if (prevIdx >= 0) pulldownFrames.push_back(gpuFrames[prevIdx]);
                    renderCenterIdx = static_cast<int>(pulldownFrames.size());
                    pulldownFrames.push_back(gpuFrames[m_centerIdx]);
                    if (nextIdx >= 0) pulldownFrames.push_back(gpuFrames[nextIdx]);
                    renderFramesPtr = &pulldownFrames;
                }
            } else {
                std::vector<int> keptRev;
                keptRev.push_back(m_centerIdx);
                int maxKeep = (m_params.trailTimebase == bmb::eTimebaseClipFps) ? m_params.trailFrames : 64;
                for (int i = m_centerIdx - 1; i >= 0 && static_cast<int>(keptRev.size()) < maxKeep; --i) {
                    if (bmb::CudaEngine::instance().areFramesDistinct(
                            gpuFrames[keptRev.back()].deviceRGBA, gpuFrames[i].deviceRGBA, m_params.width, m_params.height, _pCudaStream)) {
                        keptRev.push_back(i);
                    }
                }
                for (auto it = keptRev.rbegin(); it != keptRev.rend(); ++it) {
                    pulldownFrames.push_back(gpuFrames[*it]);
                }
                renderCenterIdx = static_cast<int>(pulldownFrames.size()) - 1;
                if (keptRev.size() >= 2) {
                    m_params.bwdFrameStep = static_cast<float>(m_centerIdx - keptRev[1]);
                }
                if (m_params.mode == bmb::eModeCombined) {
                    for (int i = m_centerIdx + 1; i < static_cast<int>(gpuFrames.size()); ++i) {
                        if (bmb::CudaEngine::instance().areFramesDistinct(
                                gpuFrames[m_centerIdx].deviceRGBA, gpuFrames[i].deviceRGBA, m_params.width, m_params.height, _pCudaStream)) {
                            m_params.fwdFrameStep = static_cast<float>(i - m_centerIdx);
                            pulldownFrames.push_back(gpuFrames[i]);
                            break;
                        }
                    }
                }
                if (pulldownFrames.size() >= 2) {
                    m_params.trailFrames = renderCenterIdx + 1;
                    renderFramesPtr = &pulldownFrames;
                }
            }
        }

        uint64_t devOut = reinterpret_cast<uint64_t>(_dstImg->getPixelData());
        bmb::CudaEngine::instance().renderFrame(
            m_instanceId, *renderFramesPtr, renderCenterIdx, devOut, m_params, bmb::FlowCache::global(), _pCudaStream, subPtr);
    }

    virtual void preProcess() override
    {
        if (_isEnabledCudaRender) return;
        if (m_centerIdx < 0 || m_centerIdx >= static_cast<int>(m_srcImages.size()) || !m_srcImages[m_centerIdx] || !_dstImg) {
            return;
        }

        const OfxRectI& bounds = m_srcImages[m_centerIdx]->getBounds();
        const int w = bounds.x2 - bounds.x1;
        const int h = bounds.y2 - bounds.y1;
        if (w <= 0 || h <= 0) return;

        m_params.width = w;
        m_params.height = h;
        const size_t nPix = static_cast<size_t>(w) * h;

        auto packFrames = [&](const std::vector<std::unique_ptr<OFX::Image>>& imgs,
                              const std::vector<std::unique_ptr<OFX::Image>>& mattes,
                              const std::vector<int>& indices,
                              const std::vector<float>* relTimes,
                              std::vector<std::vector<float>>& outPackedSrc,
                              std::vector<std::vector<float>>& outPackedMatte,
                              std::vector<bmb::FrameInput>& outCpuFrames) {
            outPackedSrc.resize(imgs.size());
            outPackedMatte.resize(imgs.size());
            outCpuFrames.resize(imgs.size());
            for (size_t i = 0; i < imgs.size(); ++i) {
                outPackedSrc[i].resize(nPix * 4);
                OFX::Image* img = imgs[i].get();
                for (int y = 0; y < h; ++y) {
                    const float* row = static_cast<const float*>(img->getPixelAddress(bounds.x1, bounds.y1 + y));
                    if (row) {
                        std::memcpy(outPackedSrc[i].data() + static_cast<size_t>(y) * w * 4, row, static_cast<size_t>(w) * 4 * sizeof(float));
                    }
                }
                outCpuFrames[i].frameIndex = indices[i];
                outCpuFrames[i].relTime = relTimes ? (*relTimes)[i] : 0.0f;
                outCpuFrames[i].hostRGBA = outPackedSrc[i].data();

                if (i < mattes.size() && mattes[i]) {
                    OFX::Image* mImg = mattes[i].get();
                    int comp = (mImg->getPixelComponents() == OFX::ePixelComponentAlpha) ? 1 : 4;
                    outPackedMatte[i].resize(nPix * comp);
                    for (int y = 0; y < h; ++y) {
                        const float* row = static_cast<const float*>(mImg->getPixelAddress(bounds.x1, bounds.y1 + y));
                        if (row) {
                            std::memcpy(outPackedMatte[i].data() + static_cast<size_t>(y) * w * comp, row, static_cast<size_t>(w) * comp * sizeof(float));
                        }
                    }
                    outCpuFrames[i].hostMatte = outPackedMatte[i].data();
                    outCpuFrames[i].matteComponents = comp;
                }
            }
        };

        m_cpuPackedSrc.clear();
        m_cpuPackedMatte.clear();
        std::vector<bmb::FrameInput> cpuFrames;
        packFrames(m_srcImages, m_matteImages, m_frameIndices, nullptr, m_cpuPackedSrc, m_cpuPackedMatte, cpuFrames);

        std::vector<bmb::FrameInput> cpuSubFrames, distinctSubFrames;
        const std::vector<bmb::FrameInput>* subPtr = nullptr;
        if (m_subImages.size() >= 3) {
            packFrames(m_subImages, m_subMatteImages, m_subFrameIndices, &m_subRelTimes, m_cpuPackedSubSrc, m_cpuPackedSubMatte, cpuSubFrames);
            for (const auto& sf : cpuSubFrames) {
                if (distinctSubFrames.empty() ||
                    bmb::CpuFallbackProcessor::areFramesDistinct(distinctSubFrames.back().hostRGBA, sf.hostRGBA, w, h)) {
                    distinctSubFrames.push_back(sf);
                }
            }
            if (distinctSubFrames.size() >= 3) {
                subPtr = &distinctSubFrames;
            }
        }

        std::vector<bmb::FrameInput> cpuTrailSub, distinctTrailSub;
        const std::vector<bmb::FrameInput>* renderFramesPtr = &cpuFrames;
        int renderCenterIdx = m_centerIdx;

        if (m_trailSubImages.size() >= 2 && (m_params.mode == bmb::eModeLongExposureTrail || m_params.mode == bmb::eModeCombined)) {
            packFrames(m_trailSubImages, m_trailSubMatteImages, m_trailSubFrameIndices, nullptr, m_cpuPackedTrailSubSrc, m_cpuPackedTrailSubMatte, cpuTrailSub);
            for (const auto& tf : cpuTrailSub) {
                if (distinctTrailSub.empty() ||
                    bmb::CpuFallbackProcessor::areFramesDistinct(distinctTrailSub.back().hostRGBA, tf.hostRGBA, w, h)) {
                    distinctTrailSub.push_back(tf);
                }
            }
            if (distinctTrailSub.size() >= 2) {
                renderCenterIdx = static_cast<int>(distinctTrailSub.size()) - 1;
                if (m_params.mode == bmb::eModeCombined && m_centerIdx + 1 < static_cast<int>(cpuFrames.size())) {
                    distinctTrailSub.push_back(cpuFrames[m_centerIdx + 1]);
                }
                m_params.pairTimeStep = m_trailPairTimeStep;
                m_params.trailFrames = renderCenterIdx + 1;
                renderFramesPtr = &distinctTrailSub;
            }
        }

        std::vector<bmb::FrameInput> pulldownFrames;
        if (renderFramesPtr == &cpuFrames && m_probePulldown && cpuFrames.size() > 2) {
            if (m_centerIdx > 0 &&
                !bmb::CpuFallbackProcessor::areFramesDistinct(cpuFrames[m_centerIdx].hostRGBA, cpuFrames[m_centerIdx - 1].hostRGBA, w, h)) {
                m_params.isDuplicateFrame = true;
            }
            if (m_params.mode == bmb::eModeShutterBlur) {
                if (!subPtr && cpuFrames.size() > 3) {
                    int prevIdx = -1;
                    for (int i = m_centerIdx - 1; i >= 0; --i) {
                        if (bmb::CpuFallbackProcessor::areFramesDistinct(cpuFrames[m_centerIdx].hostRGBA, cpuFrames[i].hostRGBA, w, h)) {
                            prevIdx = i;
                            m_params.bwdFrameStep = static_cast<float>(m_centerIdx - i);
                            break;
                        }
                    }
                    int nextIdx = -1;
                    for (int i = m_centerIdx + 1; i < static_cast<int>(cpuFrames.size()); ++i) {
                        if (bmb::CpuFallbackProcessor::areFramesDistinct(cpuFrames[m_centerIdx].hostRGBA, cpuFrames[i].hostRGBA, w, h)) {
                            nextIdx = i;
                            m_params.fwdFrameStep = static_cast<float>(i - m_centerIdx);
                            break;
                        }
                    }
                    if (prevIdx >= 0) pulldownFrames.push_back(cpuFrames[prevIdx]);
                    renderCenterIdx = static_cast<int>(pulldownFrames.size());
                    pulldownFrames.push_back(cpuFrames[m_centerIdx]);
                    if (nextIdx >= 0) pulldownFrames.push_back(cpuFrames[nextIdx]);
                    renderFramesPtr = &pulldownFrames;
                }
            } else {
                std::vector<int> keptRev;
                keptRev.push_back(m_centerIdx);
                int maxKeep = (m_params.trailTimebase == bmb::eTimebaseClipFps) ? m_params.trailFrames : 64;
                for (int i = m_centerIdx - 1; i >= 0 && static_cast<int>(keptRev.size()) < maxKeep; --i) {
                    if (bmb::CpuFallbackProcessor::areFramesDistinct(cpuFrames[keptRev.back()].hostRGBA, cpuFrames[i].hostRGBA, w, h)) {
                        keptRev.push_back(i);
                    }
                }
                for (auto it = keptRev.rbegin(); it != keptRev.rend(); ++it) {
                    pulldownFrames.push_back(cpuFrames[*it]);
                }
                renderCenterIdx = static_cast<int>(pulldownFrames.size()) - 1;
                if (keptRev.size() >= 2) {
                    m_params.bwdFrameStep = static_cast<float>(m_centerIdx - keptRev[1]);
                }
                if (m_params.mode == bmb::eModeCombined) {
                    for (int i = m_centerIdx + 1; i < static_cast<int>(cpuFrames.size()); ++i) {
                        if (bmb::CpuFallbackProcessor::areFramesDistinct(cpuFrames[m_centerIdx].hostRGBA, cpuFrames[i].hostRGBA, w, h)) {
                            m_params.fwdFrameStep = static_cast<float>(i - m_centerIdx);
                            pulldownFrames.push_back(cpuFrames[i]);
                            break;
                        }
                    }
                }
                if (pulldownFrames.size() >= 2) {
                    m_params.trailFrames = renderCenterIdx + 1;
                    renderFramesPtr = &pulldownFrames;
                }
            }
        }

        m_cpuOutput.resize(nPix * 4);
        bmb::CpuFallbackProcessor::renderFrame(
            m_instanceId, *renderFramesPtr, renderCenterIdx, m_cpuOutput.data(), m_params, bmb::FlowCache::global(), subPtr);
    }

    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow) override
    {
        if (m_cpuOutput.empty() || !_dstImg) return;
        const OfxRectI& bounds = _dstImg->getBounds();
        const int w = bounds.x2 - bounds.x1;
        const int copyW = p_ProcWindow.x2 - p_ProcWindow.x1;
        if (copyW <= 0) return;

        for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y) {
            if (_effect.abort()) break;
            int localY = y - bounds.y1;
            int localX = p_ProcWindow.x1 - bounds.x1;
            if (localY < 0 || localY >= m_params.height || localX < 0 || localX + copyW > w) continue;

            float* dstRow = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));
            const float* srcRow = m_cpuOutput.data() + (static_cast<size_t>(localY) * w + localX) * 4;
            if (dstRow) {
                std::memcpy(dstRow, srcRow, static_cast<size_t>(copyW) * 4 * sizeof(float));
            }
        }
    }

private:
    const void* m_instanceId = nullptr;
    std::vector<std::unique_ptr<OFX::Image>> m_srcImages;
    std::vector<std::unique_ptr<OFX::Image>> m_matteImages;
    std::vector<int> m_frameIndices;
    int m_centerIdx = 0;
    bmb::RenderParams m_params;
    bool m_probePulldown = false;
    std::vector<std::unique_ptr<OFX::Image>> m_subImages;
    std::vector<std::unique_ptr<OFX::Image>> m_subMatteImages;
    std::vector<int> m_subFrameIndices;
    std::vector<float> m_subRelTimes;
    std::vector<std::unique_ptr<OFX::Image>> m_trailSubImages;
    std::vector<std::unique_ptr<OFX::Image>> m_trailSubMatteImages;
    std::vector<int> m_trailSubFrameIndices;
    float m_trailPairTimeStep = 1.0f;

    std::vector<std::vector<float>> m_cpuPackedSrc;
    std::vector<std::vector<float>> m_cpuPackedMatte;
    std::vector<std::vector<float>> m_cpuPackedSubSrc;
    std::vector<std::vector<float>> m_cpuPackedSubMatte;
    std::vector<std::vector<float>> m_cpuPackedTrailSubSrc;
    std::vector<std::vector<float>> m_cpuPackedTrailSubMatte;
    std::vector<float> m_cpuOutput;
};

////////////////////////////////////////////////////////////////////////////////
class BetterMotionBlurPlugin : public OFX::ImageEffect
{
public:
    explicit BetterMotionBlurPlugin(OfxImageEffectHandle p_Handle);
    virtual ~BetterMotionBlurPlugin();

    virtual void render(const OFX::RenderArguments& p_Args) override;
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime) override;
    virtual void getFramesNeeded(const OFX::FramesNeededArguments& p_Args, OFX::FramesNeededSetter& p_FramesNeededSetter) override;
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName) override;

private:
    bmb::RenderParams readParamsAtTime(double time) const;
    void queryFrameRates(double& outTimelineFps, double& outClipFps) const;
    void updateParamVisibility();

    OFX::Clip* m_DstClip = nullptr;
    OFX::Clip* m_SrcClip = nullptr;
    OFX::Clip* m_MatteClip = nullptr;

    OFX::ChoiceParam* m_ModeParam = nullptr;
    OFX::ChoiceParam* m_ViewModeParam = nullptr;
    OFX::ChoiceParam* m_FlowEngineParam = nullptr;
    OFX::DoubleParam* m_FlowDetailParam = nullptr;
    OFX::DoubleParam* m_MotionSensitivityParam = nullptr;
    OFX::DoubleParam* m_MotionThresholdParam = nullptr;
    OFX::ChoiceParam* m_MatteModeParam = nullptr;
    OFX::DoubleParam* m_ShutterAngleParam = nullptr;
    OFX::DoubleParam* m_ShutterPhaseParam = nullptr;
    OFX::ChoiceParam* m_ShutterTimebaseParam = nullptr;
    OFX::BooleanParam* m_AutoShutterSamplesParam = nullptr;
    OFX::IntParam* m_ShutterSamplesParam = nullptr;
    OFX::IntParam* m_TrailFramesParam = nullptr;
    OFX::ChoiceParam* m_TrailTimebaseParam = nullptr;
    OFX::IntParam* m_SubFrameStepsParam = nullptr;
    OFX::ChoiceParam* m_TrailBlendModeParam = nullptr;
    OFX::DoubleParam* m_TrailDecayParam = nullptr;
    OFX::DoubleParam* m_RearCurtainCrispnessParam = nullptr;
    OFX::DoubleParam* m_LumaThresholdParam = nullptr;
    OFX::BooleanParam* m_SequentialCacheParam = nullptr;
    OFX::DoubleParam* m_HighlightBoostParam = nullptr;
    OFX::GroupParam* m_GroupShutter = nullptr;
    OFX::GroupParam* m_GroupTrail = nullptr;
    OFX::GroupParam* m_GroupPhoto = nullptr;
};

BetterMotionBlurPlugin::BetterMotionBlurPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    if (getContext() == OFX::eContextGeneral) {
        m_MatteClip = fetchClip(kClipMatteName);
    }

    m_ModeParam = fetchChoiceParam(kParamMode);
    m_ViewModeParam = fetchChoiceParam(kParamViewMode);
    m_FlowEngineParam = fetchChoiceParam(kParamFlowEngine);
    m_FlowDetailParam = fetchDoubleParam(kParamFlowDetail);
    m_MotionSensitivityParam = fetchDoubleParam(kParamMotionSensitivity);
    m_MotionThresholdParam = fetchDoubleParam(kParamMotionThreshold);
    m_MatteModeParam = fetchChoiceParam(kParamMatteMode);
    m_ShutterAngleParam = fetchDoubleParam(kParamShutterAngle);
    m_ShutterPhaseParam = fetchDoubleParam(kParamShutterPhase);
    m_ShutterTimebaseParam = fetchChoiceParam(kParamShutterTimebase);
    m_AutoShutterSamplesParam = fetchBooleanParam(kParamAutoShutterSamples);
    m_ShutterSamplesParam = fetchIntParam(kParamShutterSamples);
    m_TrailFramesParam = fetchIntParam(kParamTrailFrames);
    m_TrailTimebaseParam = fetchChoiceParam(kParamTrailTimebase);
    m_SubFrameStepsParam = fetchIntParam(kParamSubFrameSteps);
    m_TrailBlendModeParam = fetchChoiceParam(kParamTrailBlendMode);
    m_TrailDecayParam = fetchDoubleParam(kParamTrailDecay);
    m_RearCurtainCrispnessParam = fetchDoubleParam(kParamRearCurtainCrispness);
    m_LumaThresholdParam = fetchDoubleParam(kParamLumaThreshold);
    m_SequentialCacheParam = fetchBooleanParam(kParamSequentialCache);
    m_HighlightBoostParam = fetchDoubleParam(kParamHighlightBoost);
    m_GroupShutter = fetchGroupParam(kGroupShutter);
    m_GroupTrail = fetchGroupParam(kGroupTrail);
    m_GroupPhoto = fetchGroupParam(kGroupPhoto);

    updateParamVisibility();
}

BetterMotionBlurPlugin::~BetterMotionBlurPlugin()
{
    bmb::FlowCache::global().invalidateInstance(this);
}

void BetterMotionBlurPlugin::updateParamVisibility()
{
    auto setVisible = [](OFX::Param* p, bool visible) {
        if (p) {
            p->setIsSecret(!visible);
            p->setEnabled(visible);
        }
    };

    int mode = 0;
    if (m_ModeParam) m_ModeParam->getValue(mode);
    int blend = 0;
    if (m_TrailBlendModeParam) m_TrailBlendModeParam->getValue(blend);
    bool seqCache = false;
    if (m_SequentialCacheParam) m_SequentialCacheParam->getValue(seqCache);

    const bool showShutter = (mode == bmb::eModeShutterBlur || mode == bmb::eModeCombined);
    const bool showTrail = (mode == bmb::eModeLongExposureTrail || mode == bmb::eModeCombined);
    const bool showDecay = showTrail && (blend == bmb::eTrailCometDecay || seqCache);
    const bool showCrisp = showTrail && (blend == bmb::eTrailCometDecay);
    const bool showLuma = showTrail && (blend == bmb::eTrailLightPainting);
    const bool showHighlight = !(mode == bmb::eModeLongExposureTrail && blend == bmb::eTrailLightPainting);
    const bool showMatteMode = (getContext() == OFX::eContextGeneral);

    if (m_GroupShutter) m_GroupShutter->setIsSecret(!showShutter);
    setVisible(m_ShutterAngleParam, showShutter);
    setVisible(m_ShutterPhaseParam, showShutter);
    setVisible(m_ShutterTimebaseParam, showShutter);
    setVisible(m_AutoShutterSamplesParam, showShutter);
    setVisible(m_ShutterSamplesParam, showShutter);

    if (m_GroupTrail) m_GroupTrail->setIsSecret(!showTrail);
    setVisible(m_TrailFramesParam, showTrail);
    setVisible(m_TrailTimebaseParam, showTrail);
    setVisible(m_SubFrameStepsParam, showTrail);
    setVisible(m_TrailBlendModeParam, showTrail);
    setVisible(m_SequentialCacheParam, showTrail);
    setVisible(m_TrailDecayParam, showDecay);
    setVisible(m_RearCurtainCrispnessParam, showCrisp);
    setVisible(m_LumaThresholdParam, showLuma);

    if (m_GroupPhoto) m_GroupPhoto->setIsSecret(!showHighlight);
    setVisible(m_HighlightBoostParam, showHighlight);

    setVisible(m_MatteModeParam, showMatteMode);
}

void BetterMotionBlurPlugin::changedParam(const OFX::InstanceChangedArgs& /*p_Args*/, const std::string& p_ParamName)
{
    if (p_ParamName == kParamMode || p_ParamName == kParamTrailBlendMode || p_ParamName == kParamSequentialCache) {
        updateParamVisibility();
    }
}

void BetterMotionBlurPlugin::queryFrameRates(double& outTimelineFps, double& outClipFps) const
{
    double tl = getFrameRate();
    if (tl <= 0.01 && m_DstClip) tl = m_DstClip->getFrameRate();
    if (tl <= 0.01) tl = 24.0;

    double cl = tl;
    if (m_SrcClip && m_SrcClip->isConnected()) {
        double unmapped = m_SrcClip->getUnmappedFrameRate();
        if (unmapped > 0.01) {
            cl = unmapped;
        } else {
            double mapped = m_SrcClip->getFrameRate();
            if (mapped > 0.01) cl = mapped;
        }
    }
    outTimelineFps = tl;
    outClipFps = cl;
}

bmb::RenderParams BetterMotionBlurPlugin::readParamsAtTime(double time) const
{
    auto getChoice = [time](OFX::ChoiceParam* param) -> int {
        int val = 0;
        if (param) param->getValueAtTime(time, val);
        return val;
    };

    bmb::RenderParams p;
    p.mode = static_cast<bmb::EffectMode>(getChoice(m_ModeParam));
    p.viewMode = static_cast<bmb::DiagnosticView>(getChoice(m_ViewModeParam));
    p.flowEngine = static_cast<bmb::FlowEngineType>(getChoice(m_FlowEngineParam));
    p.flowDetail = static_cast<float>(m_FlowDetailParam->getValueAtTime(time));
    p.motionSensitivity = static_cast<float>(m_MotionSensitivityParam->getValueAtTime(time));
    p.motionThreshold = static_cast<float>(m_MotionThresholdParam->getValueAtTime(time));
    p.matteMode = static_cast<bmb::MatteMode>(getChoice(m_MatteModeParam));
    p.shutterAngle = static_cast<float>(m_ShutterAngleParam->getValueAtTime(time));
    p.shutterPhase = static_cast<float>(m_ShutterPhaseParam->getValueAtTime(time));
    p.shutterTimebase = static_cast<bmb::ShutterTimebase>(getChoice(m_ShutterTimebaseParam));
    if (p.shutterTimebase == bmb::eTimebaseClipFps) {
        double tlFps = 24.0, clFps = 24.0;
        queryFrameRates(tlFps, clFps);
        p.timebaseScale = std::clamp(static_cast<float>(tlFps / clFps), 0.05f, 8.0f);
    } else {
        p.timebaseScale = 1.0f;
    }
    p.autoShutterSamples = m_AutoShutterSamplesParam ? m_AutoShutterSamplesParam->getValueAtTime(time) : true;
    p.shutterSamples = std::max(4, m_ShutterSamplesParam->getValueAtTime(time));
    p.trailFrames = std::clamp(m_TrailFramesParam->getValueAtTime(time), 2, 24);
    p.trailTimebase = static_cast<bmb::ShutterTimebase>(getChoice(m_TrailTimebaseParam));
    p.subFrameSteps = std::clamp(m_SubFrameStepsParam->getValueAtTime(time), 1, 16);
    p.trailBlendMode = static_cast<bmb::TrailBlendMode>(getChoice(m_TrailBlendModeParam));
    p.trailDecay = static_cast<float>(m_TrailDecayParam->getValueAtTime(time));
    p.rearCurtainCrispness = static_cast<float>(m_RearCurtainCrispnessParam->getValueAtTime(time));
    p.lumaThreshold = static_cast<float>(m_LumaThresholdParam->getValueAtTime(time));
    p.sequentialCache = m_SequentialCacheParam->getValueAtTime(time);
    p.highlightBoost = static_cast<float>(m_HighlightBoostParam->getValueAtTime(time));
    return p;
}

bool BetterMotionBlurPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    const auto p = readParamsAtTime(p_Args.time);
    if (p.viewMode == bmb::eViewFinal && p.mode == bmb::eModeShutterBlur &&
        (p.shutterAngle <= 0.001f || p.motionSensitivity <= 0.0001f)) {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }
    return false;
}

void BetterMotionBlurPlugin::getFramesNeeded(const OFX::FramesNeededArguments& p_Args, OFX::FramesNeededSetter& p_FramesNeededSetter)
{
    const auto p = readParamsAtTime(p_Args.time);
    int lookback = 1;
    if (p.mode == bmb::eModeLongExposureTrail || p.mode == bmb::eModeCombined) {
        lookback = std::max(1, p.trailFrames - 1);
    }

    const bool needNextFrame = (p.mode != bmb::eModeLongExposureTrail) || (p.viewMode != bmb::eViewFinal);
    double minOff = -static_cast<double>(lookback);
    double maxOff = needNextFrame ? 1.0 : 0.0;

    if (p.viewMode == bmb::eViewFinal) {
        double tlFps = 24.0, clFps = 24.0;
        queryFrameRates(tlFps, clFps);
        const bool trailActive = (p.mode == bmb::eModeLongExposureTrail || p.mode == bmb::eModeCombined);
        const bool shutterActive = (p.mode == bmb::eModeShutterBlur || p.mode == bmb::eModeCombined);

        if (trailActive) {
            if (tlFps > clFps * 1.05) {
                int maxPull = std::clamp(static_cast<int>(std::ceil(tlFps / clFps)), 1, 4);
                int trailSpan = (p.trailTimebase == bmb::eTimebaseClipFps)
                                    ? std::min(60, static_cast<int>(std::ceil((p.trailFrames - 1) * (tlFps / clFps))) + maxPull)
                                    : std::min(60, lookback + maxPull);
                minOff = std::min(minOff, -static_cast<double>(trailSpan));
                if (shutterActive) maxOff = std::max(maxOff, static_cast<double>(maxPull));
            } else if (clFps > tlFps * 1.05 && p.trailTimebase == bmb::eTimebaseClipFps) {
                int scaledLookback = std::max(1, static_cast<int>(std::ceil((p.trailFrames - 1) * (tlFps / clFps))));
                minOff = std::min(minOff, -static_cast<double>(scaledLookback));
            }
        }
        if (shutterActive) {
            if (tlFps > clFps * 1.05) {
                int maxPull = std::clamp(static_cast<int>(std::ceil(tlFps / clFps)), 1, 4);
                minOff = std::min(minOff, -static_cast<double>(maxPull));
                maxOff = std::max(maxOff, static_cast<double>(maxPull));
            } else if (clFps > tlFps * 1.05) {
                double effSpan = (p.shutterAngle / 360.0) * p.timebaseScale;
                double effPhase = (p.shutterPhase / 360.0) * p.timebaseScale;
                double lo = std::clamp(std::floor(effPhase - 0.5 * effSpan - 1.0), -4.0, 0.0);
                double hi = std::clamp(std::ceil(effPhase + 0.5 * effSpan + 1.0), 0.0, 4.0);
                minOff = std::min(minOff, lo);
                maxOff = std::max(maxOff, hi);
            }
        }
    }

    OfxRangeD range;
    range.min = p_Args.time + minOff;
    range.max = p_Args.time + maxOff;

    p_FramesNeededSetter.setFramesNeeded(*m_SrcClip, range);
    if (m_MatteClip && m_MatteClip->isConnected() && p.matteMode != bmb::eMatteIgnore) {
        p_FramesNeededSetter.setFramesNeeded(*m_MatteClip, range);
    }
}

void BetterMotionBlurPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() != OFX::eBitDepthFloat) ||
        (m_DstClip->getPixelComponents() != OFX::ePixelComponentRGBA)) {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
        return;
    }

    bmb::RenderParams params = readParamsAtTime(p_Args.time);
    std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    if (!dst) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
        return;
    }

    const int currFrame = static_cast<int>(std::llround(p_Args.time));
    int lookback = (params.mode == bmb::eModeLongExposureTrail || params.mode == bmb::eModeCombined)
                       ? std::max(1, params.trailFrames - 1)
                       : 1;
    int maxOffset = (params.mode != bmb::eModeLongExposureTrail || params.viewMode != bmb::eViewFinal) ? 1 : 0;

    double tlFps = 24.0, clFps = 24.0;
    queryFrameRates(tlFps, clFps);
    const bool shutterActive = (params.viewMode == bmb::eViewFinal) &&
                               (params.mode == bmb::eModeShutterBlur || params.mode == bmb::eModeCombined);
    const bool trailActive = (params.viewMode == bmb::eViewFinal) &&
                             (params.mode == bmb::eModeLongExposureTrail || params.mode == bmb::eModeCombined);
    const bool probePulldown = (shutterActive || trailActive) && (tlFps > clFps * 1.05);
    const bool trySubFrameProbe = (shutterActive || trailActive) && (clFps > tlFps * 1.05);

    if (probePulldown) {
        int maxPull = std::clamp(static_cast<int>(std::ceil(tlFps / clFps)), 1, 4);
        if (trailActive) {
            lookback = (params.trailTimebase == bmb::eTimebaseClipFps)
                           ? std::min(60, static_cast<int>(std::ceil((params.trailFrames - 1) * (tlFps / clFps))) + maxPull)
                           : std::min(60, lookback + maxPull);
        } else {
            lookback = std::max(lookback, maxPull);
        }
        if (shutterActive) maxOffset = std::max(maxOffset, maxPull);
    } else if (trailActive && clFps > tlFps * 1.05 && params.trailTimebase == bmb::eTimebaseClipFps) {
        int scaled = std::clamp(static_cast<int>(std::lround(1.0 + (params.trailFrames - 1) * (tlFps / clFps))), 2, 24);
        lookback = std::max(1, scaled - 1);
    }

    std::vector<std::unique_ptr<OFX::Image>> srcImages;
    std::vector<std::unique_ptr<OFX::Image>> matteImages;
    std::vector<int> frameIndices;
    int centerIdx = -1;

    const bool useMatte = (m_MatteClip && m_MatteClip->isConnected() && params.matteMode != bmb::eMatteIgnore);

    for (int offset = -lookback; offset <= maxOffset; ++offset) {
        double sampleTime = p_Args.time + static_cast<double>(offset);
        std::unique_ptr<OFX::Image> sImg(m_SrcClip->fetchImage(sampleTime));
        if (!sImg) {
            if (offset == 0) {
                OFX::throwSuiteStatusException(kOfxStatFailed);
                return;
            }
            continue;
        }
        if (offset == 0) {
            centerIdx = static_cast<int>(srcImages.size());
        }
        srcImages.push_back(std::move(sImg));
        frameIndices.push_back(currFrame + offset);

        if (useMatte) {
            matteImages.push_back(std::unique_ptr<OFX::Image>(m_MatteClip->fetchImage(sampleTime)));
        } else {
            matteImages.push_back(nullptr);
        }
    }

    if (centerIdx < 0) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
        return;
    }

    // Fractional sub-frame probing when high-FPS clip is on a lower-FPS timeline
    std::vector<std::unique_ptr<OFX::Image>> subImages;
    std::vector<std::unique_ptr<OFX::Image>> subMatteImages;
    std::vector<int> subFrameIndices;
    std::vector<float> subRelTimes;
    std::vector<std::unique_ptr<OFX::Image>> trailSubImages;
    std::vector<std::unique_ptr<OFX::Image>> trailSubMatteImages;
    std::vector<int> trailSubFrameIndices;
    float trailPairTimeStep = 1.0f;

    if (trySubFrameProbe) {
        double subStep = std::clamp(tlFps / clFps, 1.0 / 16.0, 0.95);
        const std::string& currUid = srcImages[centerIdx]->getUniqueIdentifier();
        if (shutterActive) {
            std::unique_ptr<OFX::Image> probeImg(m_SrcClip->fetchImage(p_Args.time + subStep));
            bool hostSupportsSubFrames = probeImg &&
                                         (probeImg->getUniqueIdentifier().empty() ||
                                          currUid.empty() ||
                                          probeImg->getUniqueIdentifier() != currUid);
            if (hostSupportsSubFrames) {
                double effSpan = (params.shutterAngle / 360.0) * params.timebaseScale;
                double effPhase = (params.shutterPhase / 360.0) * params.timebaseScale;
                double tauMin = effPhase - 0.5 * effSpan;
                double tauMax = effPhase + 0.5 * effSpan;
                int kMin = std::max(-4, static_cast<int>(std::floor(tauMin / subStep)) - 1);
                int kMax = std::min(4, static_cast<int>(std::ceil(tauMax / subStep)) + 1);
                while (kMax - kMin + 1 > 9) {
                    if (-kMin > kMax) ++kMin;
                    else --kMax;
                }
                for (int k = kMin; k <= kMax; ++k) {
                    double relT = static_cast<double>(k) * subStep;
                    std::unique_ptr<OFX::Image> sfImg = (k == 1)
                        ? std::move(probeImg)
                        : std::unique_ptr<OFX::Image>(m_SrcClip->fetchImage(p_Args.time + relT));
                    if (!sfImg) continue;
                    subImages.push_back(std::move(sfImg));
                    subFrameIndices.push_back(currFrame * 16 + k);
                    subRelTimes.push_back(static_cast<float>(relT));
                    if (useMatte) {
                        subMatteImages.push_back(std::unique_ptr<OFX::Image>(m_MatteClip->fetchImage(p_Args.time + relT)));
                    } else {
                        subMatteImages.push_back(nullptr);
                    }
                }
            }
        }
        if (trailActive) {
            std::unique_ptr<OFX::Image> backProbe(m_SrcClip->fetchImage(p_Args.time - subStep));
            bool hostSupportsTrailSub = backProbe &&
                                        (backProbe->getUniqueIdentifier().empty() ||
                                         currUid.empty() ||
                                         backProbe->getUniqueIdentifier() != currUid);
            if (hostSupportsTrailSub) {
                int numSub = 2;
                double step = subStep;
                if (params.trailTimebase == bmb::eTimebaseClipFps) {
                    numSub = std::clamp(params.trailFrames, 2, 24);
                    trailPairTimeStep = 1.0f;
                } else {
                    int winTl = std::max(1, params.trailFrames - 1);
                    int rawN = static_cast<int>(std::ceil(static_cast<double>(winTl) / subStep)) + 1;
                    int stride = std::max(1, static_cast<int>(std::ceil(static_cast<double>(rawN - 1) / 24.0)));
                    step = subStep * static_cast<double>(stride);
                    numSub = std::clamp(static_cast<int>(std::llround(static_cast<double>(winTl) / step)) + 1, 2, 25);
                    trailPairTimeStep = static_cast<float>(step);
                }
                for (int m = numSub - 1; m >= 0; --m) {
                    double sampleTime = p_Args.time - static_cast<double>(m) * step;
                    std::unique_ptr<OFX::Image> tfImg = (m == 1 && std::fabs(step - subStep) < 1e-6 && backProbe)
                        ? std::move(backProbe)
                        : std::unique_ptr<OFX::Image>(m_SrcClip->fetchImage(sampleTime));
                    if (!tfImg) continue;
                    trailSubImages.push_back(std::move(tfImg));
                    trailSubFrameIndices.push_back(static_cast<int>(std::llround(sampleTime * 1000.0)));
                    if (useMatte) {
                        trailSubMatteImages.push_back(std::unique_ptr<OFX::Image>(m_MatteClip->fetchImage(sampleTime)));
                    } else {
                        trailSubMatteImages.push_back(nullptr);
                    }
                }
            } else if (params.trailTimebase == bmb::eTimebaseClipFps) {
                params.trailFrames = std::clamp(static_cast<int>(std::lround(1.0 + (params.trailFrames - 1) * (tlFps / clFps))), 2, 24);
            }
        }
    }

    BetterMotionBlurProcessor proc(*this);
    proc.setDstImg(dst.get());
    proc.setGPURenderArgs(p_Args);
    proc.setRenderWindow(p_Args.renderWindow);
    proc.setInputFrames(this, std::move(srcImages), std::move(matteImages), std::move(frameIndices), centerIdx, params,
                        probePulldown, std::move(subImages), std::move(subMatteImages), std::move(subFrameIndices), std::move(subRelTimes),
                        std::move(trailSubImages), std::move(trailSubMatteImages), std::move(trailSubFrameIndices), trailPairTimeStep);
    proc.process();
}

////////////////////////////////////////////////////////////////////////////////
BetterMotionBlurPluginFactory::BetterMotionBlurPluginFactory()
    : OFX::PluginFactoryHelper<BetterMotionBlurPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{}

void BetterMotionBlurPluginFactory::unload()
{
    bmb::FlowCache::global().clear();
    bmb::CudaEngine::instance().shutdown();
    bmb::NeuralFlowEngine::instance().shutdown();
}

void BetterMotionBlurPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    p_Desc.addSupportedContext(OFX::eContextFilter);
    p_Desc.addSupportedContext(OFX::eContextGeneral);

    p_Desc.addSupportedBitDepth(OFX::eBitDepthFloat);

    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(true);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

#ifndef __APPLE__
    p_Desc.setSupportsCudaRender(true);
    p_Desc.setSupportsCudaStream(true);
#endif
}

static OFX::DoubleParamDescriptor* defDouble(
    OFX::ImageEffectDescriptor& desc, const char* name, const char* label, const char* hint,
    double defVal, double minVal, double maxVal, double dispMin, double dispMax, OFX::GroupParamDescriptor* parent)
{
    auto* p = desc.defineDoubleParam(name);
    p->setLabels(label, label, label);
    p->setScriptName(name);
    p->setHint(hint);
    p->setDefault(defVal);
    p->setRange(minVal, maxVal);
    p->setDisplayRange(dispMin, dispMax);
    if (parent) p->setParent(*parent);
    return p;
}

static OFX::IntParamDescriptor* defInt(
    OFX::ImageEffectDescriptor& desc, const char* name, const char* label, const char* hint,
    int defVal, int minVal, int maxVal, int dispMin, int dispMax, OFX::GroupParamDescriptor* parent)
{
    auto* p = desc.defineIntParam(name);
    p->setLabels(label, label, label);
    p->setScriptName(name);
    p->setHint(hint);
    p->setDefault(defVal);
    p->setRange(minVal, maxVal);
    p->setDisplayRange(dispMin, dispMax);
    if (parent) p->setParent(*parent);
    return p;
}

static OFX::BooleanParamDescriptor* defBool(
    OFX::ImageEffectDescriptor& desc, const char* name, const char* label, const char* hint,
    bool defVal, OFX::GroupParamDescriptor* parent)
{
    auto* p = desc.defineBooleanParam(name);
    p->setLabels(label, label, label);
    p->setScriptName(name);
    p->setHint(hint);
    p->setDefault(defVal);
    if (parent) p->setParent(*parent);
    return p;
}

void BetterMotionBlurPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum p_Context)
{
    OFX::ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(OFX::ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(true);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    if (p_Context == OFX::eContextGeneral) {
        OFX::ClipDescriptor* matteClip = p_Desc.defineClip(kClipMatteName);
        matteClip->addSupportedComponent(OFX::ePixelComponentAlpha);
        matteClip->addSupportedComponent(OFX::ePixelComponentRGBA);
        matteClip->setTemporalClipAccess(true);
        matteClip->setSupportsTiles(kSupportsTiles);
        matteClip->setOptional(true);
        matteClip->setIsMask(true);
    }

    OFX::ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(OFX::ePixelComponentRGBA);
    dstClip->setSupportsTiles(kSupportsTiles);

    OFX::PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    // Mode & View
    OFX::ChoiceParamDescriptor* modeParam = p_Desc.defineChoiceParam(kParamMode);
    modeParam->setLabels("Render Mode", "Render Mode", "Render Mode");
    modeParam->setScriptName(kParamMode);
    modeParam->appendOption("Natural Shutter Blur", "Simulate cinematic shutter blur on high-shutter clips");
    modeParam->appendOption("Long-Exposure Trail", "Continuous multi-frame long-exposure trail");
    modeParam->appendOption("Combined (Shutter + Trail)", "Apply natural shutter blur followed by long-exposure trail");
    modeParam->setDefault(0);
    page->addChild(*modeParam);

    OFX::ChoiceParamDescriptor* viewParam = p_Desc.defineChoiceParam(kParamViewMode);
    viewParam->setLabels("Diagnostic View", "Diagnostic View", "Diagnostic View");
    viewParam->setScriptName(kParamViewMode);
    viewParam->appendOption("Final Output", "Render final composited image");
    viewParam->appendOption("Motion Vectors (HSV)", "Visualize optical flow angle & magnitude");
    viewParam->appendOption("Occlusion Confidence", "Green = reliable flow, Red = occluded");
    viewParam->appendOption("Trail Motion/Luma Mask", "Preview pixels triggering light trails");
    viewParam->setDefault(0);
    page->addChild(*viewParam);

    // Group: Motion Estimation
    OFX::GroupParamDescriptor* gFlow = p_Desc.defineGroupParam("groupMotion");
    gFlow->setLabels("Motion Estimation (Optical Flow)", "Motion Estimation", "Motion Estimation");
    page->addChild(*gFlow);

    OFX::ChoiceParamDescriptor* engineParam = p_Desc.defineChoiceParam(kParamFlowEngine);
    engineParam->setLabels("Flow Engine", "Flow Engine", "Flow Engine");
    engineParam->setScriptName(kParamFlowEngine);
    engineParam->appendOption("Neural Optical Flow (NeuFlow v2)", "Proxy neural refinement + 4K guided upsampling");
    engineParam->appendOption("Fast CUDA Pyramidal", "Real-time GPU PatchMatch + Lucas-Kanade");
    engineParam->setDefault(0);
    engineParam->setParent(*gFlow);
    page->addChild(*engineParam);

    page->addChild(*defDouble(p_Desc, kParamFlowDetail, "Edge Preservation", "Joint-bilateral edge stopping sharpness", 0.75, 0.0, 1.0, 0.0, 1.0, gFlow));
    page->addChild(*defDouble(p_Desc, kParamMotionSensitivity, "Motion Sensitivity", "Global multiplier on estimated motion vectors", 1.0, 0.0, 4.0, 0.0, 2.5, gFlow));
    page->addChild(*defDouble(p_Desc, kParamMotionThreshold, "Motion Threshold (px)", "Minimum velocity before blur/trail activates", 0.5, 0.0, 10.0, 0.0, 5.0, gFlow));

    OFX::ChoiceParamDescriptor* matteModeParam = p_Desc.defineChoiceParam(kParamMatteMode);
    matteModeParam->setLabels("Matte Handling", "Matte Handling", "Matte Handling");
    matteModeParam->setScriptName(kParamMatteMode);
    matteModeParam->appendOption("Ignore Matte", "Ignore optional Matte input");
    matteModeParam->appendOption("Foreground Isolation", "Prevent background bleed across foreground matte edge");
    matteModeParam->appendOption("Mask Effect", "Restrict blur/trail strictly inside Matte");
    matteModeParam->setDefault(1);
    matteModeParam->setParent(*gFlow);
    page->addChild(*matteModeParam);

    // Group: Natural Shutter Blur
    OFX::GroupParamDescriptor* gShutter = p_Desc.defineGroupParam(kGroupShutter);
    gShutter->setLabels("Natural Shutter Blur", "Natural Shutter Blur", "Natural Shutter Blur");
    page->addChild(*gShutter);

    page->addChild(*defDouble(p_Desc, kParamShutterAngle, "Shutter Angle (deg)", "Virtual camera shutter angle (180 = standard cinema)", 180.0, 0.0, 720.0, 0.0, 360.0, gShutter));
    page->addChild(*defDouble(p_Desc, kParamShutterPhase, "Shutter Phase (deg)", "Shutter temporal center offset", 0.0, -180.0, 180.0, -180.0, 180.0, gShutter));

    OFX::ChoiceParamDescriptor* timebaseParam = p_Desc.defineChoiceParam(kParamShutterTimebase);
    timebaseParam->setLabels("Shutter Timebase", "Shutter Timebase", "Shutter Timebase");
    timebaseParam->setScriptName(kParamShutterTimebase);
    timebaseParam->appendOption("Timeline FPS", "Reference shutter angle to timeline frame interval");
    timebaseParam->appendOption("Source Clip FPS", "Scale shutter angle to unmapped source clip frame interval");
    timebaseParam->setDefault(0);
    timebaseParam->setParent(*gShutter);
    page->addChild(*timebaseParam);

    page->addChild(*defBool(p_Desc, kParamAutoShutterSamples, "Auto Shutter Samples", "Adapt sample count per 16x16 tile from optical flow speed (uses Shutter Samples as max cap)", true, gShutter));
    page->addChild(*defInt(p_Desc, kParamShutterSamples, "Shutter Samples", "Sub-frame integration samples (acts as upper cap when Auto Shutter Samples is enabled)", 48, 8, 64, 8, 64, gShutter));

    // Group: Long-Exposure Trail
    OFX::GroupParamDescriptor* gTrail = p_Desc.defineGroupParam(kGroupTrail);
    gTrail->setLabels("Long-Exposure Trail", "Long-Exposure Trail", "Long-Exposure Trail");
    page->addChild(*gTrail);

    page->addChild(*defInt(p_Desc, kParamTrailFrames, "Trail Length (Frames)", "Lookback temporal window for long exposure", 8, 2, 24, 2, 24, gTrail));

    OFX::ChoiceParamDescriptor* trailTbParam = p_Desc.defineChoiceParam(kParamTrailTimebase);
    trailTbParam->setLabels("Trail Timebase", "Trail Timebase", "Trail Timebase");
    trailTbParam->setScriptName(kParamTrailTimebase);
    trailTbParam->appendOption("Timeline FPS", "Count Trail Length in timeline frames");
    trailTbParam->appendOption("Source Clip FPS", "Count Trail Length in distinct source clip frames");
    trailTbParam->setDefault(0);
    trailTbParam->setParent(*gTrail);
    page->addChild(*trailTbParam);

    page->addChild(*defInt(p_Desc, kParamSubFrameSteps, "Sub-Frame Steps", "Flow-bridged steps between consecutive frames (eliminates stepping)", 6, 1, 16, 1, 12, gTrail));

    OFX::ChoiceParamDescriptor* blendParam = p_Desc.defineChoiceParam(kParamTrailBlendMode);
    blendParam->setLabels("Trail Blend Mode", "Trail Blend Mode", "Trail Blend Mode");
    blendParam->setScriptName(kParamTrailBlendMode);
    blendParam->appendOption("Physical Exposure (Linear)", "Energy-conserving linear-light exposure integration");
    blendParam->appendOption("Light-Painting (Max/Screen)", "Continuous bright trail ribbons without altering static background");
    blendParam->appendOption("Comet / Rear-Curtain Decay", "Exponential comet tail decay with sharp current subject");
    blendParam->setDefault(0);
    blendParam->setParent(*gTrail);
    page->addChild(*blendParam);

    page->addChild(*defDouble(p_Desc, kParamTrailDecay, "Trail Decay", "Exponential falloff rate for Comet/Sequential modes", 0.35, 0.0, 1.0, 0.0, 1.0, gTrail));
    page->addChild(*defDouble(p_Desc, kParamRearCurtainCrispness, "Rear-Curtain Crispness", "Blends sharp current-frame subject over comet wake", 0.25, 0.0, 1.0, 0.0, 1.0, gTrail));
    page->addChild(*defDouble(p_Desc, kParamLumaThreshold, "Luma Threshold", "Restrict Light-Painting trails to highlights above threshold", 0.0, 0.0, 1.0, 0.0, 1.0, gTrail));
    page->addChild(*defBool(p_Desc, kParamSequentialCache, "Sequential Accumulation Cache", "Carry forward ultra-long trails during sequential playback", false, gTrail));

    // Group: Photometrics & Color
    OFX::GroupParamDescriptor* gPhoto = p_Desc.defineGroupParam(kGroupPhoto);
    gPhoto->setLabels("Photometrics & Color", "Photometrics & Color", "Photometrics & Color");
    page->addChild(*gPhoto);

    page->addChild(*defDouble(p_Desc, kParamHighlightBoost, "Highlight Streak Boost", "HDR linear-light weight in DaVinci Wide Gamut for realistic specular streaks", 1.5, 0.0, 5.0, 0.0, 3.0, gPhoto));
}

OFX::ImageEffect* BetterMotionBlurPluginFactory::createInstance(OfxImageEffectHandle p_Handle, OFX::ContextEnum /*p_Context*/)
{
    return new BetterMotionBlurPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static BetterMotionBlurPluginFactory factory;
    p_FactoryArray.push_back(&factory);
}
