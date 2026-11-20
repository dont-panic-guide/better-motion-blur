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
constexpr const char* kParamShutterSamples = "shutterSamples";
constexpr const char* kParamTrailFrames = "trailFrames";
constexpr const char* kParamSubFrameSteps = "subFrameSteps";
constexpr const char* kParamTrailBlendMode = "trailBlendMode";
constexpr const char* kParamTrailDecay = "trailDecay";
constexpr const char* kParamRearCurtainCrispness = "rearCurtainCrispness";
constexpr const char* kParamLumaThreshold = "lumaThreshold";
constexpr const char* kParamSequentialCache = "sequentialCache";
constexpr const char* kParamHighlightBoost = "highlightBoost";

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
                        const bmb::RenderParams& params)
    {
        m_instanceId = instanceId;
        m_srcImages = std::move(srcImages);
        m_matteImages = std::move(matteImages);
        m_frameIndices = std::move(frameIndices);
        m_centerIdx = centerIdx;
        m_params = params;
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

        std::vector<bmb::FrameInput> gpuFrames(m_srcImages.size());
        for (size_t i = 0; i < m_srcImages.size(); ++i) {
            gpuFrames[i].frameIndex = m_frameIndices[i];
            gpuFrames[i].deviceRGBA = reinterpret_cast<uint64_t>(m_srcImages[i]->getPixelData());
            if (i < m_matteImages.size() && m_matteImages[i]) {
                gpuFrames[i].deviceMatte = reinterpret_cast<uint64_t>(m_matteImages[i]->getPixelData());
                gpuFrames[i].matteComponents =
                    (m_matteImages[i]->getPixelComponents() == OFX::ePixelComponentAlpha) ? 1 : 4;
            }
        }

        uint64_t devOut = reinterpret_cast<uint64_t>(_dstImg->getPixelData());
        bmb::CudaEngine::instance().renderFrame(
            m_instanceId, gpuFrames, m_centerIdx, devOut, m_params, bmb::FlowCache::global(), _pCudaStream);
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

        m_cpuPackedSrc.resize(m_srcImages.size());
        m_cpuPackedMatte.resize(m_srcImages.size());
        std::vector<bmb::FrameInput> cpuFrames(m_srcImages.size());

        for (size_t i = 0; i < m_srcImages.size(); ++i) {
            m_cpuPackedSrc[i].resize(nPix * 4);
            OFX::Image* img = m_srcImages[i].get();
            for (int y = 0; y < h; ++y) {
                const float* row = static_cast<const float*>(img->getPixelAddress(bounds.x1, bounds.y1 + y));
                if (row) {
                    std::memcpy(m_cpuPackedSrc[i].data() + static_cast<size_t>(y) * w * 4, row, static_cast<size_t>(w) * 4 * sizeof(float));
                }
            }
            cpuFrames[i].frameIndex = m_frameIndices[i];
            cpuFrames[i].hostRGBA = m_cpuPackedSrc[i].data();

            if (i < m_matteImages.size() && m_matteImages[i]) {
                OFX::Image* mImg = m_matteImages[i].get();
                int comp = (mImg->getPixelComponents() == OFX::ePixelComponentAlpha) ? 1 : 4;
                m_cpuPackedMatte[i].resize(nPix * comp);
                for (int y = 0; y < h; ++y) {
                    const float* row = static_cast<const float*>(mImg->getPixelAddress(bounds.x1, bounds.y1 + y));
                    if (row) {
                        std::memcpy(m_cpuPackedMatte[i].data() + static_cast<size_t>(y) * w * comp, row, static_cast<size_t>(w) * comp * sizeof(float));
                    }
                }
                cpuFrames[i].hostMatte = m_cpuPackedMatte[i].data();
                cpuFrames[i].matteComponents = comp;
            }
        }

        m_cpuOutput.resize(nPix * 4);
        bmb::CpuFallbackProcessor::renderFrame(
            m_instanceId, cpuFrames, m_centerIdx, m_cpuOutput.data(), m_params, bmb::FlowCache::global());
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

    std::vector<std::vector<float>> m_cpuPackedSrc;
    std::vector<std::vector<float>> m_cpuPackedMatte;
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

private:
    bmb::RenderParams readParamsAtTime(double time) const;

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
    OFX::IntParam* m_ShutterSamplesParam = nullptr;
    OFX::IntParam* m_TrailFramesParam = nullptr;
    OFX::IntParam* m_SubFrameStepsParam = nullptr;
    OFX::ChoiceParam* m_TrailBlendModeParam = nullptr;
    OFX::DoubleParam* m_TrailDecayParam = nullptr;
    OFX::DoubleParam* m_RearCurtainCrispnessParam = nullptr;
    OFX::DoubleParam* m_LumaThresholdParam = nullptr;
    OFX::BooleanParam* m_SequentialCacheParam = nullptr;
    OFX::DoubleParam* m_HighlightBoostParam = nullptr;
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
    m_ShutterSamplesParam = fetchIntParam(kParamShutterSamples);
    m_TrailFramesParam = fetchIntParam(kParamTrailFrames);
    m_SubFrameStepsParam = fetchIntParam(kParamSubFrameSteps);
    m_TrailBlendModeParam = fetchChoiceParam(kParamTrailBlendMode);
    m_TrailDecayParam = fetchDoubleParam(kParamTrailDecay);
    m_RearCurtainCrispnessParam = fetchDoubleParam(kParamRearCurtainCrispness);
    m_LumaThresholdParam = fetchDoubleParam(kParamLumaThreshold);
    m_SequentialCacheParam = fetchBooleanParam(kParamSequentialCache);
    m_HighlightBoostParam = fetchDoubleParam(kParamHighlightBoost);
}

BetterMotionBlurPlugin::~BetterMotionBlurPlugin()
{
    bmb::FlowCache::global().invalidateInstance(this);
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
    p.shutterSamples = std::max(4, m_ShutterSamplesParam->getValueAtTime(time));
    p.trailFrames = std::clamp(m_TrailFramesParam->getValueAtTime(time), 2, 24);
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
        lookback = std::max(1, p.trailFrames);
    }

    const bool needNextFrame = (p.mode != bmb::eModeLongExposureTrail) || (p.viewMode != bmb::eViewFinal);
    OfxRangeD range;
    range.min = p_Args.time - static_cast<double>(lookback);
    range.max = p_Args.time + (needNextFrame ? 1.0 : 0.0);

    p_FramesNeededSetter.setFramesNeeded(*m_SrcClip, range);
    if (m_MatteClip && m_MatteClip->isConnected()) {
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

    const bmb::RenderParams params = readParamsAtTime(p_Args.time);
    std::unique_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    if (!dst) {
        OFX::throwSuiteStatusException(kOfxStatFailed);
        return;
    }

    const int currFrame = static_cast<int>(std::llround(p_Args.time));
    const int lookback = (params.mode == bmb::eModeLongExposureTrail || params.mode == bmb::eModeCombined)
                             ? std::max(1, params.trailFrames)
                             : 1;

    std::vector<std::unique_ptr<OFX::Image>> srcImages;
    std::vector<std::unique_ptr<OFX::Image>> matteImages;
    std::vector<int> frameIndices;
    int centerIdx = -1;

    const bool useMatte = (m_MatteClip && m_MatteClip->isConnected() && params.matteMode != bmb::eMatteIgnore);

    const int maxOffset = (params.mode != bmb::eModeLongExposureTrail || params.viewMode != bmb::eViewFinal) ? 1 : 0;

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

    BetterMotionBlurProcessor proc(*this);
    proc.setDstImg(dst.get());
    proc.setGPURenderArgs(p_Args);
    proc.setRenderWindow(p_Args.renderWindow);
    proc.setInputFrames(this, std::move(srcImages), std::move(matteImages), std::move(frameIndices), centerIdx, params);
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
    OFX::GroupParamDescriptor* gShutter = p_Desc.defineGroupParam("groupShutter");
    gShutter->setLabels("Natural Shutter Blur", "Natural Shutter Blur", "Natural Shutter Blur");
    page->addChild(*gShutter);

    page->addChild(*defDouble(p_Desc, kParamShutterAngle, "Shutter Angle (deg)", "Virtual camera shutter angle (180 = standard cinema)", 180.0, 0.0, 720.0, 0.0, 360.0, gShutter));
    page->addChild(*defDouble(p_Desc, kParamShutterPhase, "Shutter Phase (deg)", "Shutter temporal center offset", 0.0, -180.0, 180.0, -180.0, 180.0, gShutter));
    page->addChild(*defInt(p_Desc, kParamShutterSamples, "Shutter Samples", "Sub-frame trajectory integration samples", 24, 8, 64, 8, 48, gShutter));

    // Group: Long-Exposure Trail
    OFX::GroupParamDescriptor* gTrail = p_Desc.defineGroupParam("groupTrail");
    gTrail->setLabels("Long-Exposure Trail", "Long-Exposure Trail", "Long-Exposure Trail");
    page->addChild(*gTrail);

    page->addChild(*defInt(p_Desc, kParamTrailFrames, "Trail Length (Frames)", "Lookback temporal window for long exposure", 8, 2, 24, 2, 24, gTrail));
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
    OFX::GroupParamDescriptor* gPhoto = p_Desc.defineGroupParam("groupPhoto");
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
