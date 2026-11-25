#include "CommonTypes.h"
#include "CpuFallbackProcessor.h"
#include "CudaEngine.h"
#include "FlowCache.h"
#include "NeuralFlowEngine.h"
#include "OnnxModelGenerator.h"

#include "ofxCore.h"
#include "ofxImageEffect.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace bmb;

namespace {

int g_failures = 0;

void report(const char* name, bool ok, const std::string& detail) {
    if (ok) {
        std::cout << "[PASS] " << name << " -- " << detail << "\n";
    } else {
        std::cout << "[FAIL] " << name << " -- " << detail << "\n";
        ++g_failures;
    }
}

std::string getExecutableDir() {
    char buf[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, buf, MAX_PATH);
    return std::filesystem::path(buf).parent_path().string();
}

// Procedural multi-frequency texture for accurate optical flow testing
void evalTextureColor(float x, float y, float& r, float& g, float& b) {
    float v1 = 0.5f + 0.25f * std::sin(x * 0.13f) * std::cos(y * 0.11f)
                    + 0.15f * std::sin(x * 0.31f + y * 0.19f)
                    + 0.10f * std::cos(x * 0.05f - y * 0.07f);
    float v2 = 0.5f + 0.25f * std::cos(x * 0.09f - y * 0.14f)
                    + 0.15f * std::sin(x * 0.27f - y * 0.23f)
                    + 0.10f * std::sin(x * 0.06f + y * 0.04f);
    float v3 = 0.5f + 0.25f * std::sin(x * 0.17f + y * 0.08f)
                    + 0.15f * std::cos(x * 0.21f + y * 0.29f);
    r = std::min(1.0f, std::max(0.0f, v1));
    g = std::min(1.0f, std::max(0.0f, v2));
    b = std::min(1.0f, std::max(0.0f, v3));
}

std::vector<float> makeTranslatedTextureFrame(int w, int h, float shiftX, float shiftY) {
    std::vector<float> img(static_cast<size_t>(w) * h * 4, 1.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            float r, g, b;
            evalTextureColor(static_cast<float>(x) - shiftX, static_cast<float>(y) - shiftY, r, g, b);
            size_t idx = (static_cast<size_t>(y) * w + x) * 4;
            img[idx + 0] = r;
            img[idx + 1] = g;
            img[idx + 2] = b;
            img[idx + 3] = 1.0f;
        }
    }
    return img;
}

void makeMovingDiscScene(int w, int h, float cx, float cy, float radius,
                         std::vector<float>& outRGBA, std::vector<float>& outMatte)
{
    outRGBA.assign(static_cast<size_t>(w) * h * 4, 1.0f);
    outMatte.assign(static_cast<size_t>(w) * h, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t p = static_cast<size_t>(y) * w + x;
            float dx = static_cast<float>(x) - cx;
            float dy = static_cast<float>(y) - cy;
            float dist = std::hypot(dx, dy);

            if (dist <= radius) {
                // Distinct bright warm foreground disc with subtle internal texture
                float tex = 0.85f + 0.12f * std::sin(dx * 0.25f) * std::cos(dy * 0.25f);
                outRGBA[p * 4 + 0] = tex;
                outRGBA[p * 4 + 1] = 0.20f;
                outRGBA[p * 4 + 2] = 0.05f;
                outRGBA[p * 4 + 3] = 1.0f;
                outMatte[p] = 1.0f;
            } else {
                // Distinct cool blue-green textured background (zero red channel)
                float bg = 0.25f + 0.15f * std::sin(x * 0.15f) * std::cos(y * 0.15f);
                outRGBA[p * 4 + 0] = 0.02f;
                outRGBA[p * 4 + 1] = bg * 0.7f;
                outRGBA[p * 4 + 2] = bg + 0.25f;
                outRGBA[p * 4 + 3] = 1.0f;
                outMatte[p] = 0.0f;
            }
        }
    }
}

std::vector<float> makePointLightFrame(int w, int h, float cx, float cy, float radius, float peakIntensity) {
    std::vector<float> img(static_cast<size_t>(w) * h * 4, 0.0f);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t p = static_cast<size_t>(y) * w + x;
            float d = std::hypot(static_cast<float>(x) - cx, static_cast<float>(y) - cy);
            float val = peakIntensity * std::exp(-(d * d) / (2.0f * (radius * 0.45f) * (radius * 0.45f)));
            if (d > radius * 2.2f) val = 0.02f; // dark night background
            else val = std::max(0.02f, val);
            img[p * 4 + 0] = val;
            img[p * 4 + 1] = val * 0.75f;
            img[p * 4 + 2] = val * 0.25f;
            img[p * 4 + 3] = 1.0f;
        }
    }
    return img;
}

} // namespace

int main() {
    std::string exeDir = getExecutableDir();
    std::string bundlePath = exeDir + "\\BetterMotionBlur.ofx.bundle\\Contents\\Win64\\BetterMotionBlur.ofx";
    std::string onnxModelPath = exeDir + "\\BetterMotionBlur.ofx.bundle\\Contents\\Resources\\neuflow_v2.onnx";

    // Test 1: CUDA + NVRTC & ONNX Runtime Initialization
    bool modelWritten = writeResidualFlowOnnxToFile(onnxModelPath);
    bool cudaOk = CudaEngine::instance().initialize();
    bool ortOk = NeuralFlowEngine::instance().initialize(onnxModelPath);
    report("Test 1: CUDA+NVRTC & ONNX Runtime Init",
           modelWritten && cudaOk && ortOk,
           "GPU=" + CudaEngine::instance().deviceName() +
           " | ORT=" + NeuralFlowEngine::instance().runtimeVersion() +
           (cudaOk ? "" : (" | CudaErr=" + CudaEngine::instance().lastError())) +
           (ortOk ? "" : (" | OrtErr=" + NeuralFlowEngine::instance().lastError())));

    if (!cudaOk) {
        std::cerr << "Cannot continue GPU tests without CUDA.\n";
        return 1;
    }

    // Test 2: Optical Flow Accuracy (Neural & Pyramidal)
    {
        const int w = 512, h = 288;
        const float gtDx = 12.0f, gtDy = -6.0f;
        auto imgA = makeTranslatedTextureFrame(w, h, 0.0f, 0.0f);
        auto imgB = makeTranslatedTextureFrame(w, h, gtDx, gtDy);

        const size_t rgbaBytes = static_cast<size_t>(w) * h * 4 * sizeof(float);
        uint64_t devA = CudaEngine::instance().allocDevice(rgbaBytes);
        uint64_t devB = CudaEngine::instance().allocDevice(rgbaBytes);
        CudaEngine::instance().uploadToDevice(devA, imgA.data(), rgbaBytes);
        CudaEngine::instance().uploadToDevice(devB, imgB.data(), rgbaBytes);

        auto measureEpe = [&](FlowEngineType eng) -> double {
            FlowCache localCache;
            RenderParams p;
            p.width = w; p.height = h;
            p.flowEngine = eng;
            p.flowDetail = 0.8f;
            auto entry = CudaEngine::instance().getOrComputeFlowPair(
                &p, 0, devA, 0, 0, 1, devB, 0, 0, p, localCache, nullptr);
            CudaEngine::instance().syncStream(nullptr);
            if (!entry) return 999.0;

            std::vector<Float4> hostFlow;
            if (!CudaEngine::instance().downloadFlowToHost(*entry, hostFlow, nullptr)) return 999.0;

            double sumErr = 0.0;
            size_t count = 0;
            const int margin = 32;
            for (int y = margin; y < h - margin; ++y) {
                for (int x = margin; x < w - margin; ++x) {
                    const Float4& f = hostFlow[static_cast<size_t>(y) * w + x];
                    sumErr += std::hypot(f.x - gtDx, f.y - gtDy);
                    ++count;
                }
            }
            return sumErr / std::max<size_t>(1, count);
        };

        double epeNeural = measureEpe(eFlowNeural);
        double epePyr = measureEpe(eFlowCudaPyramidal);

        CudaEngine::instance().freeDevice(devA);
        CudaEngine::instance().freeDevice(devB);

        char buf[160];
        std::snprintf(buf, sizeof(buf), "Neural EPE=%.4f px, Pyramidal EPE=%.4f px (target < 0.50 px)", epeNeural, epePyr);
        report("Test 2: Optical Flow Accuracy (Neural & Pyramidal)",
               epeNeural < 0.5 && epePyr < 0.5, buf);
    }

    // Test 3: High-Shutter Natural Motion Blur & Matte Boundary Isolation
    {
        const int w = 512, h = 288;
        const float cy = 144.0f, radius = 48.0f;
        std::vector<float> fPrev, mPrev, fCurr, mCurr, fNext, mNext;
        makeMovingDiscScene(w, h, 240.0f, cy, radius, fPrev, mPrev);
        makeMovingDiscScene(w, h, 256.0f, cy, radius, fCurr, mCurr);
        makeMovingDiscScene(w, h, 272.0f, cy, radius, fNext, mNext);

        std::vector<FrameInput> frames(3);
        frames[0] = {0, fPrev.data(), 0, mPrev.data(), 0, 1};
        frames[1] = {1, fCurr.data(), 0, mCurr.data(), 0, 1};
        frames[2] = {2, fNext.data(), 0, mNext.data(), 0, 1};

        RenderParams p;
        p.width = w; p.height = h;
        p.mode = eModeShutterBlur;
        p.flowEngine = eFlowNeural;
        p.matteMode = eMatteForegroundIsolation;
        p.shutterAngle = 180.0f;
        p.shutterSamples = 24;

        std::vector<float> outRGBA(static_cast<size_t>(w) * h * 4, 0.0f);
        FlowCache cache;
        bool ok = CudaEngine::instance().renderHostFrames(&p, frames, 1, outRGBA.data(), p, cache);

        // Check interior foreground pixel (near center of disc at x=256, y=144): must have zero blue background bleed
        float maxBlueInCore = 0.0f;
        for (int y = 132; y <= 156; ++y) {
            for (int x = 244; x <= 268; ++x) {
                float b = outRGBA[(static_cast<size_t>(y) * w + x) * 4 + 2];
                maxBlueInCore = std::max(maxBlueInCore, b);
            }
        }
        // Check blur extension onto static background just outside leading edge (x=256+radius+3 = 307, y=144):
        // With 180 deg shutter & 16px/frame motion, foreground disc blurs +/-4px, so x=306 should show partial red > 0.15!
        float leadBlurRed = outRGBA[(144 * w + 306) * 4 + 0];

        char buf[192];
        std::snprintf(buf, sizeof(buf), "Core fg blue=%.4f (expected ~0.0500, no bg bleed), leading edge blur red=%.4f (>0.15)",
                      maxBlueInCore, leadBlurRed);
        report("Test 3: High-Shutter Blur & Matte Isolation",
               ok && maxBlueInCore < 0.06f && leadBlurRed > 0.15f, buf);
    }

    // Test 4: Gapless Long-Exposure Light Trail Synthesis (Sub-Frame Flow-Bridged vs Discrete Echo)
    {
        const int w = 512, h = 128;
        const int numFrames = 10;
        const float stepX = 36.0f; // 36 px/frame jump -> 9 wide dead zones in discrete mode
        const float startX = 80.0f;
        const float cy = 64.0f;

        std::vector<std::vector<float>> seq(numFrames);
        std::vector<FrameInput> frames(numFrames);
        for (int i = 0; i < numFrames; ++i) {
            seq[i] = makePointLightFrame(w, h, startX + i * stepX, cy, 5.5f, 1.0f);
            frames[i] = {i, seq[i].data(), 0, nullptr, 0, 0};
        }

        RenderParams p;
        p.width = w; p.height = h;
        p.mode = eModeLongExposureTrail;
        p.flowEngine = eFlowCudaPyramidal;
        p.trailFrames = 10;
        p.trailBlendMode = eTrailLightPainting;
        p.motionThreshold = 1.0f;
        p.lumaThreshold = 0.15f;
        p.highlightBoost = 2.5f;

        FlowCache cache;
        std::vector<float> discreteOut(static_cast<size_t>(w) * h * 4, 0.0f);
        p.subFrameSteps = 1;
        bool okDisc = CudaEngine::instance().renderHostFrames(&p, frames, numFrames - 1, discreteOut.data(), p, cache);

        std::vector<float> bridgedOut(static_cast<size_t>(w) * h * 4, 0.0f);
        p.subFrameSteps = 8;
        bool okBridged = CudaEngine::instance().renderHostFrames(&p, frames, numFrames - 1, bridgedOut.data(), p, cache);

        // Evaluate midpoints halfway between discrete frame centers (startX + (i + 0.5)*stepX)
        float minMidDiscrete = 10.0f;
        float minMidBridged = 10.0f;
        for (int i = 1; i < numFrames - 1; ++i) {
            int midX = static_cast<int>(std::lround(startX + (i + 0.5f) * stepX));
            float rDisc = discreteOut[(64 * w + midX) * 4 + 0];
            float rBrdg = bridgedOut[(64 * w + midX) * 4 + 0];
            minMidDiscrete = std::min(minMidDiscrete, rDisc);
            minMidBridged = std::min(minMidBridged, rBrdg);
        }

        char buf[192];
        std::snprintf(buf, sizeof(buf), "Discrete midpoint minR=%.4f (stroboscopic gap), Flow-Bridged midpoint minR=%.4f (>0.50 continuous)",
                      minMidDiscrete, minMidBridged);
        report("Test 4: Gapless Long-Exposure Trail Synthesis",
               okDisc && okBridged && minMidDiscrete < 0.10f && minMidBridged > 0.50f, buf);
    }

    // Test 5: LRU Temporal Flow Cache Efficiency
    {
        const int w = 256, h = 144;
        const int totalFrames = 12;
        std::vector<std::vector<float>> seq(totalFrames);
        for (int i = 0; i < totalFrames; ++i) {
            seq[i] = makeTranslatedTextureFrame(w, h, i * 4.0f, 0.0f);
        }

        FlowCache cache;
        cache.resetStats();

        RenderParams p;
        p.width = w; p.height = h;
        p.mode = eModeLongExposureTrail;
        p.flowEngine = eFlowCudaPyramidal;
        p.trailFrames = 8;
        p.subFrameSteps = 4;

        std::vector<float> out(static_cast<size_t>(w) * h * 4);
        const int tag = 42;

        // Render frame 8 (populates cache)
        {
            std::vector<FrameInput> win;
            for (int f = 0; f <= 9; ++f) win.push_back({f, seq[f].data(), 0, nullptr, 0, 0});
            CudaEngine::instance().renderHostFrames(&tag, win, 8, out.data(), p, cache);
        }

        cache.resetStats();
        // Render frame 9 and frame 10 sequentially: should reuse cached historical pairs
        for (int t : {9, 10}) {
            std::vector<FrameInput> win;
            for (int f = t - 8; f <= t + 1; ++f) win.push_back({f, seq[f].data(), 0, nullptr, 0, 0});
            CudaEngine::instance().renderHostFrames(&tag, win, 8, out.data(), p, cache);
        }

        uint64_t hits = cache.hits();
        uint64_t misses = cache.misses();
        double hitRate = (hits + misses > 0) ? (100.0 * hits / (hits + misses)) : 0.0;

        char buf[160];
        std::snprintf(buf, sizeof(buf), "Cache hits=%llu, misses=%llu, hitRate=%.1f%% (target >= 85%%)",
                      static_cast<unsigned long long>(hits),
                      static_cast<unsigned long long>(misses),
                      hitRate);
        report("Test 5: LRU Temporal Flow Cache Efficiency", hitRate >= 85.0, buf);
    }

    // Test 6: OFX Binary Symbol Export Validation
    {
        HMODULE mod = LoadLibraryA(bundlePath.c_str());
        bool loaded = (mod != nullptr);
        int numPlugins = 0;
        std::string pluginId;
        std::string pluginApi;

        if (loaded) {
            using FnGetNum = int(*)();
            using FnGetPlugin = OfxPlugin*(*)(int);
            auto getNum = reinterpret_cast<FnGetNum>(GetProcAddress(mod, "OfxGetNumberOfPlugins"));
            auto getPlug = reinterpret_cast<FnGetPlugin>(GetProcAddress(mod, "OfxGetPlugin"));
            if (getNum && getPlug) {
                numPlugins = getNum();
                if (numPlugins >= 1) {
                    OfxPlugin* plug = getPlug(0);
                    if (plug) {
                        pluginId = plug->pluginIdentifier ? plug->pluginIdentifier : "";
                        pluginApi = plug->pluginApi ? plug->pluginApi : "";
                    }
                }
            }
            FreeLibrary(mod);
        }

        report("Test 6: OFX Binary Export Validation",
               loaded && numPlugins == 1 &&
               pluginId == "com.BetterMotionBlur.OFX.BetterMotionBlur" &&
               pluginApi == kOfxImageEffectPluginApi,
               "Bundle=" + bundlePath + " | ID=" + pluginId);
    }

    // Test 8: GPU Performance & Canonical Bidirectional Cache Reuse (1080p & 4K)
    {
        auto benchSeq = [](int w, int h, FlowEngineType eng, int warmupIters, double& outColdMs, double& outWarmMs, double& outHitPct) -> bool {
            const int numFrames = 4;
            const size_t rgbaBytes = static_cast<size_t>(w) * h * 4 * sizeof(float);
            std::vector<uint64_t> devFrames(numFrames, 0);
            for (int i = 0; i < numFrames; ++i) {
                auto hostImg = makeTranslatedTextureFrame(w, h, i * 10.0f, -4.0f * i);
                devFrames[i] = CudaEngine::instance().allocDevice(rgbaBytes);
                if (!devFrames[i]) {
                    for (uint64_t p : devFrames) CudaEngine::instance().freeDevice(p);
                    return false;
                }
                CudaEngine::instance().uploadToDevice(devFrames[i], hostImg.data(), rgbaBytes);
            }
            uint64_t devOut = CudaEngine::instance().allocDevice(rgbaBytes);
            CudaEngine::instance().syncStream(nullptr);

            RenderParams p;
            p.width = w; p.height = h;
            p.mode = eModeShutterBlur;
            p.flowEngine = eng;
            p.shutterAngle = 180.0f;
            p.shutterSamples = 24;

            FlowCache cache(2048ULL * 1024ULL * 1024ULL);
            const int inst = 777;

            // Warmup frame to prime stream scratch pool
            if (warmupIters > 0) {
                std::vector<FrameInput> win = {
                    {0, nullptr, devFrames[0], 0, 0, 0},
                    {1, nullptr, devFrames[1], 0, 0, 0},
                    {2, nullptr, devFrames[2], 0, 0, 0}
                };
                CudaEngine::instance().renderFrame(&inst, win, 1, devOut, p, cache, nullptr);
                CudaEngine::instance().syncStream(nullptr);
            }
            cache.clear();
            cache.resetStats();

            // Frame 1 (cold pair (0,1) + (1,2))
            auto t0 = std::chrono::steady_clock::now();
            {
                std::vector<FrameInput> win = {
                    {0, nullptr, devFrames[0], 0, 0, 0},
                    {1, nullptr, devFrames[1], 0, 0, 0},
                    {2, nullptr, devFrames[2], 0, 0, 0}
                };
                CudaEngine::instance().renderFrame(&inst, win, 1, devOut, p, cache, nullptr);
                CudaEngine::instance().syncStream(nullptr);
            }
            auto t1 = std::chrono::steady_clock::now();
            outColdMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Sequential frame 2: reuses canonical pair (1,2) from frame 1!
            cache.resetStats();
            auto t2 = std::chrono::steady_clock::now();
            {
                std::vector<FrameInput> win = {
                    {1, nullptr, devFrames[1], 0, 0, 0},
                    {2, nullptr, devFrames[2], 0, 0, 0},
                    {3, nullptr, devFrames[3], 0, 0, 0}
                };
                CudaEngine::instance().renderFrame(&inst, win, 1, devOut, p, cache, nullptr);
                CudaEngine::instance().syncStream(nullptr);
            }
            auto t3 = std::chrono::steady_clock::now();
            outWarmMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

            uint64_t hts = cache.hits(), mxs = cache.misses();
            outHitPct = (hts + mxs > 0) ? (100.0 * hts / (hts + mxs)) : 0.0;

            cache.clear();
            CudaEngine::instance().freeDevice(devOut);
            for (uint64_t ptr : devFrames) CudaEngine::instance().freeDevice(ptr);
            return true;
        };

        double cold1080N = 0, warm1080N = 0, hit1080N = 0;
        double cold1080P = 0, warm1080P = 0, hit1080P = 0;
        double cold4KN = 0, warm4KN = 0, hit4KN = 0;
        bool b1 = benchSeq(1920, 1080, eFlowNeural, 1, cold1080N, warm1080N, hit1080N);
        bool b2 = benchSeq(1920, 1080, eFlowCudaPyramidal, 1, cold1080P, warm1080P, hit1080P);
        bool b3 = benchSeq(3840, 2160, eFlowNeural, 1, cold4KN, warm4KN, hit4KN);

        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "1080p Neural=%.1fms (%.0f fps, hit=%.0f%%), Pyr=%.1fms (%.0f fps) | 4K Neural=%.1fms (%.1f fps)",
                      warm1080N, warm1080N > 0 ? 1000.0 / warm1080N : 0.0, hit1080N,
                      warm1080P, warm1080P > 0 ? 1000.0 / warm1080P : 0.0,
                      warm4KN, warm4KN > 0 ? 1000.0 / warm4KN : 0.0);
        report("Test 8: GPU Performance & Canonical Cache Reuse",
               b1 && b2 && b3 && hit1080N >= 50.0 && hit1080P >= 50.0 && hit4KN >= 50.0, buf);
    }

    // Test 9: Long-Exposure Mode Parity, 4K Cache Non-Thrashing & Performance
    {
        // 9A: Numerical parity GPU vs exact scalar reference (using GPU flow) AND end-to-end CPU fallback
        //     across Physical Exposure & Comet Decay, plus diagonal >40px fallback path, Sequential Cache, and Combined.
        bool parityOk = true;
        double refDiffPhys = 0.0, refDiffComet = 0.0;
        double cpuDiffPhys = 0.0, cpuDiffComet = 0.0;
        float diagFallbackMinR = 0.0f, diagFallbackBg = 1.0f;
        float seqPersistR = 0.0f;
        double combinedPolluteDiff = 0.0;

        {
            const int pw = 256, ph = 144;
            const int nFrames = 6;
            std::vector<std::vector<float>> texFrames(nFrames);
            std::vector<FrameInput> fInputs(nFrames);
            for (int i = 0; i < nFrames; ++i) {
                texFrames[i] = makeTranslatedTextureFrame(pw, ph, i * 6.0f, -2.0f * i);
                fInputs[i] = {i, texFrames[i].data(), 0, nullptr, 0, 0};
            }

            auto evalMeanDiff = [&](const std::vector<float>& a, const std::vector<float>& b, int margin) {
                double sum = 0.0;
                size_t cnt = 0;
                for (int y = margin; y < ph - margin; ++y) {
                    for (int x = margin; x < pw - margin; ++x) {
                        size_t idx = (static_cast<size_t>(y) * pw + x) * 4;
                        for (int c = 0; c < 3; ++c) {
                            sum += std::abs(a[idx + c] - b[idx + c]);
                            ++cnt;
                        }
                    }
                }
                return sum / std::max<size_t>(1, cnt);
            };

            auto dwiToLin = [](float v) -> float {
                return (v <= 0.02740668f) ? (v / 10.44426855f) : (std::exp2(v / 0.07329248f - 7.0f) - 0.0075f);
            };
            auto linToDwi = [](float l) -> float {
                return (l <= 0.00262409f) ? (l * 10.44426855f) : ((std::log2(l + 0.0075f) + 7.0f) * 0.07329248f);
            };
            auto dwgLuma = [](float r, float g, float b) -> float {
                return std::max(0.0f, 0.27411851f * r + 0.87363190f * g - 0.14775041f * b);
            };
            auto sampleRGBA = [&](const float* img, float x, float y) -> Float4 {
                float cx = std::min(static_cast<float>(pw - 1), std::max(0.0f, x));
                float cy = std::min(static_cast<float>(ph - 1), std::max(0.0f, y));
                int x0 = static_cast<int>(std::floor(cx));
                int y0 = static_cast<int>(std::floor(cy));
                int x1 = std::min(x0 + 1, pw - 1);
                int y1 = std::min(y0 + 1, ph - 1);
                float fx = cx - x0, fy = cy - y0;
                float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
                const float* p00 = img + (y0 * pw + x0) * 4;
                const float* p10 = img + (y0 * pw + x1) * 4;
                const float* p01 = img + (y1 * pw + x0) * 4;
                const float* p11 = img + (y1 * pw + x1) * 4;
                return {
                    p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11,
                    p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11,
                    p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11,
                    p00[3] * w00 + p10[3] * w10 + p01[3] * w01 + p11[3] * w11
                };
            };
            auto sampleFwdXY = [&](const std::vector<Float4>& f4, float x, float y) -> Float2 {
                float cx = std::min(static_cast<float>(pw - 1), std::max(0.0f, x));
                float cy = std::min(static_cast<float>(ph - 1), std::max(0.0f, y));
                int x0 = static_cast<int>(std::floor(cx));
                int y0 = static_cast<int>(std::floor(cy));
                int x1 = std::min(x0 + 1, pw - 1);
                int y1 = std::min(y0 + 1, ph - 1);
                float fx = cx - x0, fy = cy - y0;
                float w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy), w01 = (1 - fx) * fy, w11 = fx * fy;
                const Float4& v00 = f4[y0 * pw + x0];
                const Float4& v10 = f4[y0 * pw + x1];
                const Float4& v01 = f4[y1 * pw + x0];
                const Float4& v11 = f4[y1 * pw + x1];
                return {v00.x * w00 + v10.x * w10 + v01.x * w01 + v11.x * w11,
                        v00.y * w00 + v10.y * w10 + v01.y * w01 + v11.y * w11};
            };

            auto evalScalarTrailReference = [&](const RenderParams& p, const std::vector<std::vector<Float4>>& pairFlows) {
                const int centerIdx = nFrames - 1;
                const int maxLookback = std::min(centerIdx, std::max(1, p.trailFrames - 1));
                const size_t nPix = static_cast<size_t>(pw) * ph;
                std::vector<float> accum(nPix * 4, 0.0f);
                const float* curr = fInputs[centerIdx].hostRGBA;
                for (size_t i = 0; i < nPix; ++i) {
                    accum[i * 4 + 0] = dwiToLin(curr[i * 4 + 0]) * 0.02f;
                    accum[i * 4 + 1] = dwiToLin(curr[i * 4 + 1]) * 0.02f;
                    accum[i * 4 + 2] = dwiToLin(curr[i * 4 + 2]) * 0.02f;
                    accum[i * 4 + 3] = 0.02f;
                }
                const float decayLambda = 0.25f + 3.2f * p.trailDecay;
                for (int k = 0; k <= maxLookback; ++k) {
                    int idxA = centerIdx - k;
                    int idxB = idxA - 1;
                    const float* rgbaA = fInputs[idxA].hostRGBA;
                    bool hasB = (idxB >= 0 && k < maxLookback);
                    const float* rgbaB = hasB ? fInputs[idxB].hostRGBA : rgbaA;
                    for (int y = 0; y < ph; ++y) {
                        for (int x = 0; x < pw; ++x) {
                            size_t idx = static_cast<size_t>(y) * pw + x;
                            float rA = dwiToLin(rgbaA[idx * 4 + 0]);
                            float gA = dwiToLin(rgbaA[idx * 4 + 1]);
                            float bA = dwiToLin(rgbaA[idx * 4 + 2]);
                            float luma = dwgLuma(rA, gA, bA);
                            float hdrW = 1.0f + p.highlightBoost * std::pow(std::max(0.0f, luma - 0.8f), 1.5f);
                            Float2 fl = hasB ? Float2{pairFlows[k][idx].z, pairFlows[k][idx].w} : Float2{0.0f, 0.0f};
                            float vx = fl.x * p.motionSensitivity;
                            float vy = fl.y * p.motionSensitivity;
                            float speed = std::hypot(vx, vy);
                            if (p.subFrameSteps <= 1 || !hasB || speed < 0.35f) {
                                float timeW = (p.trailBlendMode == eTrailCometDecay) ? std::exp(-decayLambda * k) : 1.0f;
                                float wTot = timeW * hdrW;
                                accum[idx * 4 + 0] += rA * wTot;
                                accum[idx * 4 + 1] += gA * wTot;
                                accum[idx * 4 + 2] += bA * wTot;
                                accum[idx * 4 + 3] += wTot;
                                continue;
                            }
                            int steps = std::max(p.subFrameSteps, std::min(72, static_cast<int>(std::ceil(speed * 1.25f))));
                            float invSteps = 1.0f / steps;
                            Float4 pB = sampleRGBA(rgbaB, x + fl.x, y + fl.y);
                            pB.x = dwiToLin(pB.x); pB.y = dwiToLin(pB.y); pB.z = dwiToLin(pB.z);
                            Float2 fDest = sampleFwdXY(pairFlows[k], x + fl.x, y + fl.y);
                            float errSq = (fl.x + fDest.x) * (fl.x + fDest.x) + (fl.y + fDest.y) * (fl.y + fDest.y);
                            float conf = std::round(std::min(1.0f, std::max(0.0f, std::exp(-errSq * 0.25f))) * 255.0f) / 255.0f;
                            for (int s = 0; s < steps; ++s) {
                                float alpha = s * invSteps;
                                float px = x + alpha * vx, py = y + alpha * vy;
                                float bb = alpha * conf;
                                float sr = rA * (1.0f - bb) + pB.x * bb;
                                float sg = gA * (1.0f - bb) + pB.y * bb;
                                float sb = bA * (1.0f - bb) + pB.z * bb;
                                float timeW = (p.trailBlendMode == eTrailCometDecay) ? std::exp(-decayLambda * (k + alpha)) : 1.0f;
                                int x0 = static_cast<int>(std::floor(px)), y0 = static_cast<int>(std::floor(py));
                                float fx = px - x0, fy = py - y0;
                                float bw[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
                                int ox[4] = {x0, x0 + 1, x0, x0 + 1}, oy[4] = {y0, y0, y0 + 1, y0 + 1};
                                float wStep = timeW * hdrW * invSteps;
                                for (int c = 0; c < 4; ++c) {
                                    if (bw[c] > 1e-4f && ox[c] >= 0 && ox[c] < pw && oy[c] >= 0 && oy[c] < ph) {
                                        size_t tidx = static_cast<size_t>(oy[c]) * pw + ox[c];
                                        float wt = wStep * bw[c];
                                        accum[tidx * 4 + 0] += sr * wt;
                                        accum[tidx * 4 + 1] += sg * wt;
                                        accum[tidx * 4 + 2] += sb * wt;
                                        accum[tidx * 4 + 3] += wt;
                                    }
                                }
                            }
                        }
                    }
                }
                std::vector<float> out(nPix * 4, 1.0f);
                for (size_t i = 0; i < nPix; ++i) {
                    float wSum = std::max(1e-5f, accum[i * 4 + 3]);
                    float or_ = accum[i * 4 + 0] / wSum, og_ = accum[i * 4 + 1] / wSum, ob_ = accum[i * 4 + 2] / wSum;
                    if (p.trailBlendMode == eTrailCometDecay && p.rearCurtainCrispness > 0.0f) {
                        float crisp = std::min(1.0f, std::max(0.0f, p.rearCurtainCrispness));
                        or_ = or_ * (1.0f - crisp) + dwiToLin(curr[i * 4 + 0]) * crisp;
                        og_ = og_ * (1.0f - crisp) + dwiToLin(curr[i * 4 + 1]) * crisp;
                        ob_ = ob_ * (1.0f - crisp) + dwiToLin(curr[i * 4 + 2]) * crisp;
                    }
                    out[i * 4 + 0] = linToDwi(or_);
                    out[i * 4 + 1] = linToDwi(og_);
                    out[i * 4 + 2] = linToDwi(ob_);
                }
                return out;
            };

            // Download GPU flow pairs once for exact scalar reference comparison
            std::vector<std::vector<Float4>> gpuPairFlows(nFrames - 1);
            {
                const size_t rgbaBytes = static_cast<size_t>(pw) * ph * 4 * sizeof(float);
                std::vector<uint64_t> devTex(nFrames, 0);
                for (int i = 0; i < nFrames; ++i) {
                    devTex[i] = CudaEngine::instance().allocDevice(rgbaBytes);
                    CudaEngine::instance().uploadToDevice(devTex[i], fInputs[i].hostRGBA, rgbaBytes);
                }
                FlowCache refFlowCache;
                RenderParams fp;
                fp.width = pw; fp.height = ph;
                fp.flowEngine = eFlowCudaPyramidal;
                for (int k = 0; k < nFrames - 1; ++k) {
                    int idxA = (nFrames - 1) - k;
                    int idxB = idxA - 1;
                    auto entry = CudaEngine::instance().getOrComputeFlowPair(
                        &fp, idxA, devTex[idxA], 0, 0, idxB, devTex[idxB], 0, 0, fp, refFlowCache, nullptr);
                    if (entry) {
                        CudaEngine::instance().downloadFlowToHost(*entry, gpuPairFlows[k], nullptr);
                    }
                }
                for (uint64_t d : devTex) CudaEngine::instance().freeDevice(d);
            }

            // 1) Physical Exposure parity
            {
                RenderParams p;
                p.width = pw; p.height = ph;
                p.mode = eModeLongExposureTrail;
                p.flowEngine = eFlowCudaPyramidal;
                p.trailFrames = 5;
                p.subFrameSteps = 6;
                p.trailBlendMode = eTrailPhysicalExposure;
                p.highlightBoost = 1.5f;

                std::vector<float> gpuOut(static_cast<size_t>(pw) * ph * 4, 0.0f);
                std::vector<float> cpuOut(static_cast<size_t>(pw) * ph * 4, 0.0f);
                FlowCache gCache, cCache;
                bool gOk = CudaEngine::instance().renderHostFrames(&p, fInputs, nFrames - 1, gpuOut.data(), p, gCache);
                bool cOk = CpuFallbackProcessor::renderFrame(&p, fInputs, nFrames - 1, cpuOut.data(), p, cCache);
                auto refOut = evalScalarTrailReference(p, gpuPairFlows);
                refDiffPhys = evalMeanDiff(gpuOut, refOut, 16);
                cpuDiffPhys = evalMeanDiff(gpuOut, cpuOut, 16);
                parityOk &= gOk && cOk && (refDiffPhys < 0.002) && (cpuDiffPhys < 0.05);
            }

            // 2) Comet Decay + Rear-Curtain Crispness parity
            {
                RenderParams p;
                p.width = pw; p.height = ph;
                p.mode = eModeLongExposureTrail;
                p.flowEngine = eFlowCudaPyramidal;
                p.trailFrames = 5;
                p.subFrameSteps = 6;
                p.trailBlendMode = eTrailCometDecay;
                p.trailDecay = 0.5f;
                p.rearCurtainCrispness = 0.3f;
                p.highlightBoost = 1.5f;

                std::vector<float> gpuOut(static_cast<size_t>(pw) * ph * 4, 0.0f);
                std::vector<float> cpuOut(static_cast<size_t>(pw) * ph * 4, 0.0f);
                FlowCache gCache, cCache;
                bool gOk = CudaEngine::instance().renderHostFrames(&p, fInputs, nFrames - 1, gpuOut.data(), p, gCache);
                bool cOk = CpuFallbackProcessor::renderFrame(&p, fInputs, nFrames - 1, cpuOut.data(), p, cCache);
                auto refOut = evalScalarTrailReference(p, gpuPairFlows);
                refDiffComet = evalMeanDiff(gpuOut, refOut, 16);
                cpuDiffComet = evalMeanDiff(gpuOut, cpuOut, 16);
                parityOk &= gOk && cOk && (refDiffComet < 0.002) && (cpuDiffComet < 0.05);
            }

            // 3) Combined mode must not pollute raw TrailSplatCache for subsequent frame
            {
                FlowCache sharedCache, cleanCache;
                const int tag = 909;
                RenderParams pComb;
                pComb.width = pw; pComb.height = ph;
                pComb.mode = eModeCombined;
                pComb.flowEngine = eFlowCudaPyramidal;
                pComb.trailFrames = 3;
                pComb.subFrameSteps = 4;
                pComb.shutterAngle = 360.0f;
                pComb.shutterSamples = 16;

                std::vector<float> tmpOut(static_cast<size_t>(pw) * ph * 4);
                std::vector<float> sharedTrailOut(static_cast<size_t>(pw) * ph * 4);
                std::vector<float> cleanTrailOut(static_cast<size_t>(pw) * ph * 4);

                // Frame 2 in Combined (splats shutter-blurred frame 2 over (2,1) and raw (1,0))
                std::vector<FrameInput> winComb = {fInputs[0], fInputs[1], fInputs[2], fInputs[3]};
                bool ok1 = CudaEngine::instance().renderHostFrames(&tag, winComb, 2, tmpOut.data(), pComb, sharedCache);

                // Frame 3 in pure Long-Exposure Trail (uses pair (3,2) and pair (2,1)!)
                RenderParams pTrail = pComb;
                pTrail.mode = eModeLongExposureTrail;
                std::vector<FrameInput> winTrail = {fInputs[1], fInputs[2], fInputs[3]};
                bool ok2 = CudaEngine::instance().renderHostFrames(&tag, winTrail, 2, sharedTrailOut.data(), pTrail, sharedCache);
                bool ok3 = CudaEngine::instance().renderHostFrames(&tag, winTrail, 2, cleanTrailOut.data(), pTrail, cleanCache);

                combinedPolluteDiff = evalMeanDiff(sharedTrailOut, cleanTrailOut, 0);
                parityOk &= ok1 && ok2 && ok3 && (combinedPolluteDiff < 1e-4);
            }
        }

        // 4) Diagonal > 40 px jump Light-Painting (triggers global fallback path boxArea > 1536)
        {
            const int lw = 384, lh = 256;
            const int numLightFrames = 5;
            const float startX = 64.0f, startY = 48.0f;
            const float stepX = 34.0f, stepY = 30.0f; // ~45.3 px diagonal jump -> boxArea > 1536
            std::vector<std::vector<float>> lSeq(numLightFrames);
            std::vector<FrameInput> lFrames(numLightFrames);
            for (int i = 0; i < numLightFrames; ++i) {
                lSeq[i] = makePointLightFrame(lw, lh, startX + i * stepX, startY + i * stepY, 6.0f, 1.0f);
                lFrames[i] = {i, lSeq[i].data(), 0, nullptr, 0, 0};
            }

            RenderParams lp;
            lp.width = lw; lp.height = lh;
            lp.mode = eModeLongExposureTrail;
            lp.flowEngine = eFlowCudaPyramidal;
            lp.trailFrames = 5;
            lp.subFrameSteps = 8;
            lp.trailBlendMode = eTrailLightPainting;
            lp.motionThreshold = 1.0f;
            lp.lumaThreshold = 0.15f;

            FlowCache lCache;
            std::vector<float> lOut(static_cast<size_t>(lw) * lh * 4, 0.0f);
            bool lOk = CudaEngine::instance().renderHostFrames(&lp, lFrames, numLightFrames - 1, lOut.data(), lp, lCache);

            diagFallbackMinR = 10.0f;
            for (int i = 0; i < numLightFrames - 1; ++i) {
                int mx = static_cast<int>(std::lround(startX + (i + 0.5f) * stepX));
                int my = static_cast<int>(std::lround(startY + (i + 0.5f) * stepY));
                diagFallbackMinR = std::min(diagFallbackMinR, lOut[(static_cast<size_t>(my) * lw + mx) * 4 + 0]);
            }
            diagFallbackBg = lOut[(20 * lw + 340) * 4 + 0];
            parityOk &= lOk && (diagFallbackMinR > 0.45f) && (diagFallbackBg <= 0.03f);
        }

        // 5) Sequential Accumulation Cache persistence across frames
        {
            const int sw = 256, sh = 128;
            std::vector<std::vector<float>> sSeq(4);
            for (int i = 0; i < 4; ++i) {
                sSeq[i] = makePointLightFrame(sw, sh, 40.0f + i * 32.0f, 64.0f, 6.0f, 1.0f);
            }
            RenderParams sp;
            sp.width = sw; sp.height = sh;
            sp.mode = eModeLongExposureTrail;
            sp.flowEngine = eFlowCudaPyramidal;
            sp.trailFrames = 2; // lookback = 1 pair -> frame 0 is outside frame 2's window without sequentialCache!
            sp.subFrameSteps = 6;
            sp.trailBlendMode = eTrailLightPainting;
            sp.trailDecay = 0.2f;
            sp.sequentialCache = true;

            FlowCache sCache;
            const int seqTag = 515;
            std::vector<float> sOut(static_cast<size_t>(sw) * sh * 4, 0.0f);
            bool okS = true;
            for (int f = 1; f <= 3; ++f) {
                std::vector<FrameInput> win = {
                    {f - 1, sSeq[f - 1].data(), 0, nullptr, 0, 0},
                    {f, sSeq[f].data(), 0, nullptr, 0, 0}
                };
                okS &= CudaEngine::instance().renderHostFrames(&seqTag, win, 1, sOut.data(), sp, sCache);
            }
            // Light peak from frame 0 at x=40, y=64 must persist into frame 3 via sequentialCache!
            seqPersistR = sOut[(64 * sw + 40) * 4 + 0];
            parityOk &= okS && (seqPersistR > 0.35f);
        }

        // 9B: 1080p & 4K Long-Exposure Performance & 4K trailFrames=16 Cache Non-Thrashing
        auto benchLongExposure = [](int w, int h, int trailLen, double& outColdMs, double& outWarmMs, double& outHitPct) -> bool {
            const int numFrames = trailLen + 1; // frames 0..trailLen -> frame trailLen-1 (cold) then trailLen (warm)
            const size_t rgbaBytes = static_cast<size_t>(w) * h * 4 * sizeof(float);
            std::vector<uint64_t> devFrames(numFrames, 0);
            for (int i = 0; i < numFrames; ++i) {
                auto hostImg = makeTranslatedTextureFrame(w, h, i * 14.0f, -3.0f * i);
                devFrames[i] = CudaEngine::instance().allocDevice(rgbaBytes);
                if (!devFrames[i]) {
                    for (uint64_t ptr : devFrames) CudaEngine::instance().freeDevice(ptr);
                    return false;
                }
                CudaEngine::instance().uploadToDevice(devFrames[i], hostImg.data(), rgbaBytes);
            }
            uint64_t devOut = CudaEngine::instance().allocDevice(rgbaBytes);
            if (!devOut) {
                for (uint64_t ptr : devFrames) CudaEngine::instance().freeDevice(ptr);
                return false;
            }
            CudaEngine::instance().syncStream(nullptr);

            RenderParams p;
            p.width = w; p.height = h;
            p.mode = eModeLongExposureTrail;
            p.flowEngine = eFlowNeural;
            p.trailFrames = trailLen;
            p.subFrameSteps = 8;
            p.trailBlendMode = eTrailPhysicalExposure;

            FlowCache& cache = FlowCache::global();
            const int inst = 888;

            // Warmup (2 iterations) to prime scratch pool and GPU boost clocks
            for (int wi = 0; wi < 2; ++wi) {
                std::vector<FrameInput> win = {
                    {0, nullptr, devFrames[0], 0, 0, 0},
                    {1, nullptr, devFrames[1], 0, 0, 0}
                };
                CudaEngine::instance().renderFrame(&inst, win, 1, devOut, p, cache, nullptr);
                CudaEngine::instance().syncStream(nullptr);
            }
            cache.clear();
            cache.resetStats();

            // Cold render at frame (trailLen - 1) using frames [0 .. trailLen - 1]
            std::vector<FrameInput> coldWin;
            coldWin.reserve(trailLen);
            for (int i = 0; i < trailLen; ++i) {
                coldWin.push_back({i, nullptr, devFrames[i], 0, 0, 0});
            }
            auto t0 = std::chrono::steady_clock::now();
            bool okCold = CudaEngine::instance().renderFrame(&inst, coldWin, trailLen - 1, devOut, p, cache, nullptr);
            CudaEngine::instance().syncStream(nullptr);
            auto t1 = std::chrono::steady_clock::now();
            outColdMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Warm sequential frame at frame (trailLen) using frames [1 .. trailLen]
            cache.resetStats();
            std::vector<FrameInput> warmWin;
            warmWin.reserve(trailLen);
            for (int i = 1; i <= trailLen; ++i) {
                warmWin.push_back({i, nullptr, devFrames[i], 0, 0, 0});
            }
            auto t2 = std::chrono::steady_clock::now();
            bool okWarm = CudaEngine::instance().renderFrame(&inst, warmWin, trailLen - 1, devOut, p, cache, nullptr);
            CudaEngine::instance().syncStream(nullptr);
            auto t3 = std::chrono::steady_clock::now();
            outWarmMs = std::chrono::duration<double, std::milli>(t3 - t2).count();

            uint64_t hts = cache.hits(), mxs = cache.misses();
            outHitPct = (hts + mxs > 0) ? (100.0 * hts / (hts + mxs)) : 0.0;

            cache.clear();
            CudaEngine::instance().freeDevice(devOut);
            for (uint64_t ptr : devFrames) CudaEngine::instance().freeDevice(ptr);
            return okCold && okWarm;
        };

        double cold1080_8 = 0, warm1080_8 = 0, hit1080_8 = 0;
        double cold4K_8 = 0, warm4K_8 = 0, hit4K_8 = 0;
        double cold4K_16 = 0, warm4K_16 = 0, hit4K_16 = 0;

        bool b1080_8 = benchLongExposure(1920, 1080, 8, cold1080_8, warm1080_8, hit1080_8);
        bool b4K_8 = benchLongExposure(3840, 2160, 8, cold4K_8, warm4K_8, hit4K_8);
        bool b4K_16 = benchLongExposure(3840, 2160, 16, cold4K_16, warm4K_16, hit4K_16);

        char buf[512];
        std::snprintf(buf, sizeof(buf),
                      "Parity(RefPhys=%.5f, RefComet=%.5f, CpuPhys=%.4f, CpuComet=%.4f, DiagMinR=%.3f, SeqR=%.3f, CombDiff=%.1e) | "
                      "1080p8=%.1fms (cold=%.1fms, %.0f fps, hit=%.0f%%) | "
                      "4K8=%.1fms (cold=%.1fms, %.1f fps, hit=%.0f%%) | "
                      "4K16=%.1fms (cold=%.1fms, %.1f fps, hit=%.0f%%)",
                      refDiffPhys, refDiffComet, cpuDiffPhys, cpuDiffComet,
                      diagFallbackMinR, seqPersistR, combinedPolluteDiff,
                      warm1080_8, cold1080_8, warm1080_8 > 0 ? 1000.0 / warm1080_8 : 0.0, hit1080_8,
                      warm4K_8, cold4K_8, warm4K_8 > 0 ? 1000.0 / warm4K_8 : 0.0, hit4K_8,
                      warm4K_16, cold4K_16, warm4K_16 > 0 ? 1000.0 / warm4K_16 : 0.0, hit4K_16);
        report("Test 9: Long-Exposure Parity, 4K Non-Thrashing & Performance",
               parityOk && b1080_8 && b4K_8 && b4K_16 &&
               hit1080_8 >= 85.0 && hit4K_8 >= 85.0 && hit4K_16 >= 85.0,
               buf);
    }

    // Test 10: Tile-Adaptive Auto Shutter Samples (GPU & CPU Parity)
    {
        const int aw = 256, ah = 128;
        std::vector<std::vector<float>> seq(3, std::vector<float>(static_cast<size_t>(aw) * ah * 4, 0.12f));
        for (int f = 0; f < 3; ++f) {
            for (int y = 0; y < ah; ++y) {
                for (int x = 0; x < aw; ++x) {
                    size_t idx = (static_cast<size_t>(y) * aw + x) * 4;
                    float r = 0.12f, g = 0.12f, b = 0.12f;
                    // Slow-moving bar (~2 px/frame) in left-middle region
                    float slowX = 48.0f + f * 2.0f;
                    if (y >= 32 && y < 64 && std::abs(x - slowX) <= 6.0f) {
                        r = 0.70f; g = 0.45f; b = 0.20f;
                    }
                    // Fast-moving bar (~24 px/frame) in right-middle region
                    float fastX = 140.0f + f * 24.0f;
                    if (y >= 64 && y < 96 && std::abs(x - fastX) <= 8.0f) {
                        r = 0.90f; g = 0.85f; b = 0.30f;
                    }
                    seq[f][idx + 0] = r;
                    seq[f][idx + 1] = g;
                    seq[f][idx + 2] = b;
                    seq[f][idx + 3] = 1.0f;
                }
            }
        }

        std::vector<FrameInput> win = {
            {0, seq[0].data(), 0, nullptr, 0, 0},
            {1, seq[1].data(), 0, nullptr, 0, 0},
            {2, seq[2].data(), 0, nullptr, 0, 0}
        };

        RenderParams pAuto;
        pAuto.width = aw; pAuto.height = ah;
        pAuto.mode = eModeShutterBlur;
        pAuto.flowEngine = eFlowCudaPyramidal;
        pAuto.shutterAngle = 180.0f;
        pAuto.autoShutterSamples = true;
        pAuto.shutterSamples = 48;

        RenderParams pFixed = pAuto;
        pFixed.autoShutterSamples = false;

        RenderParams pComb = pAuto;
        pComb.mode = eModeCombined;
        pComb.trailFrames = 3;

        FlowCache gpuCache, cpuCache;
        const int instId = 909;
        std::vector<float> gpuAutoOut(seq[1].size(), 0.0f);
        std::vector<float> cpuAutoOut(seq[1].size(), 0.0f);
        std::vector<float> gpuFixedOut(seq[1].size(), 0.0f);
        std::vector<float> gpuCombOut(seq[1].size(), 0.0f);

        bool okGpuAuto = CudaEngine::instance().renderHostFrames(&instId, win, 1, gpuAutoOut.data(), pAuto, gpuCache);
        bool okCpuAuto = CpuFallbackProcessor::renderFrame(&instId, win, 1, cpuAutoOut.data(), pAuto, cpuCache);
        bool okGpuFixed = CudaEngine::instance().renderHostFrames(&instId, win, 1, gpuFixedOut.data(), pFixed, gpuCache);
        bool okGpuComb = CudaEngine::instance().renderHostFrames(&instId, win, 1, gpuCombOut.data(), pComb, gpuCache);

        bool finiteOk = true;
        float staticMaxDiff = 0.0f;
        float gpuCpuDiffSum = 0.0f;
        for (size_t i = 0; i < gpuAutoOut.size(); ++i) {
            if (!std::isfinite(gpuAutoOut[i]) || !std::isfinite(cpuAutoOut[i]) ||
                !std::isfinite(gpuFixedOut[i]) || !std::isfinite(gpuCombOut[i])) {
                finiteOk = false;
                break;
            }
            gpuCpuDiffSum += std::abs(gpuAutoOut[i] - cpuAutoOut[i]);
        }
        // Top-left static corner (x < 24, y < 24) must remain untouched
        for (int y = 0; y < 24; ++y) {
            for (int x = 0; x < 24; ++x) {
                size_t idx = (static_cast<size_t>(y) * aw + x) * 4;
                for (int c = 0; c < 4; ++c) {
                    staticMaxDiff = std::max(staticMaxDiff, std::abs(gpuAutoOut[idx + c] - seq[1][idx + c]));
                }
            }
        }
        float gpuCpuMeanDiff = gpuCpuDiffSum / static_cast<float>(gpuAutoOut.size());

        char t10Buf[256];
        std::snprintf(t10Buf, sizeof(t10Buf),
                      "StaticDiff=%.2e, GpuCpuMeanDiff=%.4f, AutoOk=%d, FixedOk=%d, CombOk=%d",
                      staticMaxDiff, gpuCpuMeanDiff, okGpuAuto && okCpuAuto ? 1 : 0,
                      okGpuFixed ? 1 : 0, okGpuComb ? 1 : 0);
        report("Test 10: Tile-Adaptive Auto Shutter Samples (GPU & CPU Parity)",
               okGpuAuto && okCpuAuto && okGpuFixed && okGpuComb && finiteOk &&
               staticMaxDiff < 1e-4f && gpuCpuMeanDiff < 0.05f,
               t10Buf);
    }

    // Test 11: Physical 180 deg Shutter Ground-Truth, Large-Displacement (>120px), Curvature/Occlusion & Sub-Frame/Timebase
    {
        auto dwiToLin = [](float v) -> float {
            return (v <= 0.02740668f) ? (v / 10.44426855f) : (std::exp2(v / 0.07329248f - 7.0f) - 0.0075f);
        };
        auto linToDwi = [](float l) -> float {
            return (l <= 0.00262409f) ? (l * 10.44426855f) : ((std::log2(l + 0.0075f) + 7.0f) * 0.07329248f);
        };

        // 11A: Analytical 64-subframe 180 deg ground-truth profile (no-matte & matte, wide bar + thin fast bar)
        bool profileOk = false;
        double rmseGpuNoMatte = 999.0, rmseCpuNoMatte = 999.0, rmseGpuMatte = 999.0;
        float interiorEdgeR = 1.0f, exteriorRampR = 0.0f, thinBarCenterR = 1.0f;
        {
            const int pw = 512, ph = 256;
            const float speed = 24.0f; // 180 deg => +/-6 px blur
            const float wideCx = 180.0f, wideHalfW = 20.0f;
            const float thinCx = 340.0f, thinHalfW = 4.0f; // width 8 < 12 => semi-transparent across center

            auto renderSyntheticBarScene = [&](float tOffset, std::vector<float>& outMatte) {
                std::vector<float> img(static_cast<size_t>(pw) * ph * 4, 0.0f);
                outMatte.assign(static_cast<size_t>(pw) * ph, 0.0f);
                for (int y = 0; y < ph; ++y) {
                    for (int x = 0; x < pw; ++x) {
                        size_t idx = (static_cast<size_t>(y) * pw + x) * 4;
                        float bgR = 0.08f + 0.02f * std::sin(x * 0.19f) * std::cos(y * 0.17f);
                        float bgG = 0.18f + 0.03f * std::cos(x * 0.13f + y * 0.11f);
                        float bgB = 0.42f + 0.04f * std::sin(x * 0.11f - y * 0.15f);
                        float r = bgR, g = bgG, b = bgB, m = 0.0f;

                        float dxW = static_cast<float>(x) - (wideCx + tOffset * speed);
                        float dxT = static_cast<float>(x) - (thinCx + tOffset * speed);
                        if ((y >= 64 && y < 128 && std::fabs(dxW) <= wideHalfW) ||
                            (y >= 144 && y < 200 && std::fabs(dxT) <= thinHalfW)) {
                            r = 0.48f; g = 0.22f; b = 0.12f; m = 1.0f;
                        }
                        img[idx + 0] = r; img[idx + 1] = g; img[idx + 2] = b; img[idx + 3] = 1.0f;
                        outMatte[static_cast<size_t>(y) * pw + x] = m;
                    }
                }
                return img;
            };

            std::vector<float> mPrev, mCurr, mNext;
            auto fPrev = renderSyntheticBarScene(-1.0f, mPrev);
            auto fCurr = renderSyntheticBarScene(0.0f, mCurr);
            auto fNext = renderSyntheticBarScene(1.0f, mNext);

            // Analytical 64-subframe linear-light camera integral over tau in [-0.25, +0.25]
            std::vector<float> gtOut(static_cast<size_t>(pw) * ph * 4, 0.0f);
            std::vector<float> dummyM;
            const int gtN = 64;
            for (int s = 0; s < gtN; ++s) {
                float tau = -0.25f + (static_cast<float>(s) + 0.5f) / gtN * 0.5f;
                auto sub = renderSyntheticBarScene(tau, dummyM);
                for (size_t i = 0; i < static_cast<size_t>(pw) * ph; ++i) {
                    gtOut[i * 4 + 0] += dwiToLin(sub[i * 4 + 0]) / gtN;
                    gtOut[i * 4 + 1] += dwiToLin(sub[i * 4 + 1]) / gtN;
                    gtOut[i * 4 + 2] += dwiToLin(sub[i * 4 + 2]) / gtN;
                }
            }
            for (size_t i = 0; i < static_cast<size_t>(pw) * ph; ++i) {
                gtOut[i * 4 + 0] = linToDwi(gtOut[i * 4 + 0]);
                gtOut[i * 4 + 1] = linToDwi(gtOut[i * 4 + 1]);
                gtOut[i * 4 + 2] = linToDwi(gtOut[i * 4 + 2]);
            }

            std::vector<FrameInput> winNoMatte = {
                {0, fPrev.data(), 0, nullptr, 0, 0},
                {1, fCurr.data(), 0, nullptr, 0, 0},
                {2, fNext.data(), 0, nullptr, 0, 0}
            };
            std::vector<FrameInput> winMatte = {
                {0, fPrev.data(), 0, mPrev.data(), 0, 1},
                {1, fCurr.data(), 0, mCurr.data(), 0, 1},
                {2, fNext.data(), 0, mNext.data(), 0, 1}
            };

            RenderParams p;
            p.width = pw; p.height = ph;
            p.mode = eModeShutterBlur;
            p.flowEngine = eFlowCudaPyramidal;
            p.matteMode = eMatteForegroundIsolation;
            p.shutterAngle = 180.0f;
            p.shutterSamples = 48;
            p.autoShutterSamples = false;
            p.highlightBoost = 0.0f;

            std::vector<float> gpuNoMatte(gtOut.size()), cpuNoMatte(gtOut.size()), gpuMatte(gtOut.size());
            FlowCache c1, c2, c3;
            bool okG1 = CudaEngine::instance().renderHostFrames(&p, winNoMatte, 1, gpuNoMatte.data(), p, c1);
            bool okC1 = CpuFallbackProcessor::renderFrame(&p, winNoMatte, 1, cpuNoMatte.data(), p, c2);
            bool okG2 = CudaEngine::instance().renderHostFrames(&p, winMatte, 1, gpuMatte.data(), p, c3);

            auto evalProfileRmse = [&](const std::vector<float>& out) -> double {
                double sq = 0.0;
                size_t cnt = 0;
                for (int x = static_cast<int>(wideCx - wideHalfW - 10); x <= static_cast<int>(wideCx + wideHalfW + 10); ++x) {
                    size_t idx = (96 * pw + x) * 4;
                    double d = out[idx + 0] - gtOut[idx + 0];
                    sq += d * d;
                    ++cnt;
                }
                for (int x = static_cast<int>(thinCx - 12); x <= static_cast<int>(thinCx + 12); ++x) {
                    size_t idx = (172 * pw + x) * 4;
                    double d = out[idx + 0] - gtOut[idx + 0];
                    sq += d * d;
                    ++cnt;
                }
                return std::sqrt(sq / std::max<size_t>(1, cnt));
            };

            rmseGpuNoMatte = evalProfileRmse(gpuNoMatte);
            rmseCpuNoMatte = evalProfileRmse(cpuNoMatte);
            rmseGpuMatte = evalProfileRmse(gpuMatte);

            interiorEdgeR = gpuNoMatte[(96 * pw + static_cast<int>(wideCx + wideHalfW - 2)) * 4 + 0];
            exteriorRampR = gpuNoMatte[(96 * pw + static_cast<int>(wideCx + wideHalfW + 3)) * 4 + 0];
            thinBarCenterR = gpuNoMatte[(172 * pw + static_cast<int>(thinCx)) * 4 + 0];

            profileOk = okG1 && okC1 && okG2 &&
                        rmseGpuNoMatte < 0.035 && rmseCpuNoMatte < 0.055 && rmseGpuMatte < 0.035 &&
                        interiorEdgeR < 0.46f && exteriorRampR > 0.16f && exteriorRampR < 0.36f &&
                        thinBarCenterR < 0.46f;
        }

        // 11B: Large-displacement optical flow (>120 px/frame at 1920x1080)
        bool largeDispOk = false;
        double epe140Neural = 999.0, epe140Pyr = 999.0;
        {
            const int lw = 1920, lh = 1080;
            const float gtDx = 140.0f, gtDy = -32.0f;
            auto makeNonPeriodicFrame = [&](float shiftX, float shiftY) {
                std::vector<float> img(static_cast<size_t>(lw) * lh * 4, 1.0f);
                for (int y = 0; y < lh; ++y) {
                    for (int x = 0; x < lw; ++x) {
                        float sx = static_cast<float>(x) - shiftX;
                        float sy = static_cast<float>(y) - shiftY;
                        float r, g, b;
                        evalTextureColor(sx, sy, r, g, b);
                        // Add non-periodic multi-scale envelope & radial landmarks to prevent sine wrap-around
                        float env = 0.55f + 0.25f * std::sin(sx * 0.0029f + 0.4f) * std::cos(sy * 0.0037f)
                                          + 0.20f * std::cos((sx + sy) * 0.0041f);
                        float gx = std::fmod(std::fabs(sx) * 0.0071f, 1.0f);
                        float gy = std::fmod(std::fabs(sy) * 0.0053f, 1.0f);
                        float blob = std::exp(-((gx - 0.5f) * (gx - 0.5f) + (gy - 0.5f) * (gy - 0.5f)) * 18.0f);
                        size_t idx = (static_cast<size_t>(y) * lw + x) * 4;
                        img[idx + 0] = std::min(1.0f, std::max(0.0f, r * env + 0.25f * blob));
                        img[idx + 1] = std::min(1.0f, std::max(0.0f, g * env + 0.15f * std::sin(sx * 0.0063f)));
                        img[idx + 2] = std::min(1.0f, std::max(0.0f, b * env + 0.20f * std::cos(sy * 0.0079f)));
                    }
                }
                return img;
            };
            auto imgA = makeNonPeriodicFrame(0.0f, 0.0f);
            auto imgB = makeNonPeriodicFrame(gtDx, gtDy);
            const size_t rgbaBytes = static_cast<size_t>(lw) * lh * 4 * sizeof(float);
            uint64_t devA = CudaEngine::instance().allocDevice(rgbaBytes);
            uint64_t devB = CudaEngine::instance().allocDevice(rgbaBytes);
            CudaEngine::instance().uploadToDevice(devA, imgA.data(), rgbaBytes);
            CudaEngine::instance().uploadToDevice(devB, imgB.data(), rgbaBytes);

            auto measure1080 = [&](FlowEngineType eng) -> double {
                FlowCache fc;
                RenderParams fp;
                fp.width = lw; fp.height = lh;
                fp.flowEngine = eng;
                fp.flowDetail = 0.75f;
                auto entry = CudaEngine::instance().getOrComputeFlowPair(
                    &fp, 0, devA, 0, 0, 1, devB, 0, 0, fp, fc, nullptr);
                CudaEngine::instance().syncStream(nullptr);
                if (!entry) return 999.0;
                std::vector<Float4> hostFlow;
                if (!CudaEngine::instance().downloadFlowToHost(*entry, hostFlow, nullptr)) return 999.0;
                double sumErr = 0.0;
                size_t cnt = 0;
                for (int y = 96; y < lh - 96; y += 4) {
                    for (int x = 96; x < lw - 200; x += 4) {
                        const Float4& f = hostFlow[static_cast<size_t>(y) * lw + x];
                        sumErr += std::hypot(f.x - gtDx, f.y - gtDy);
                        ++cnt;
                    }
                }
                return sumErr / std::max<size_t>(1, cnt);
            };
            epe140Neural = measure1080(eFlowNeural);
            epe140Pyr = measure1080(eFlowCudaPyramidal);
            CudaEngine::instance().freeDevice(devA);
            CudaEngine::instance().freeDevice(devB);
            largeDispOk = (epe140Neural < 2.0 && epe140Pyr < 2.0);
        }

        // 11C: Curved trajectory sign (+y vs inverted -y) & single-sided occlusion velocity preservation
        bool curveAndOccOk = false;
        float curvePlusY = 0.0f, curveMinusY = 0.0f, occBlurR = 0.0f, unoccBlurR = 0.0f;
        {
            const int cw = 256, ch = 192;
            const float cx = 128.0f, cy = 96.0f;
            auto makeDiscAt = [&](float px, float py) {
                std::vector<float> img(static_cast<size_t>(cw) * ch * 4, 0.0f);
                for (int y = 0; y < ch; ++y) {
                    for (int x = 0; x < cw; ++x) {
                        size_t idx = (static_cast<size_t>(y) * cw + x) * 4;
                        float dx = static_cast<float>(x) - px;
                        float dy = static_cast<float>(y) - py;
                        float d = std::hypot(dx, dy);
                        if (d <= 20.0f) {
                            float tex = 0.72f + 0.10f * std::sin(dx * 0.35f) * std::cos(dy * 0.35f);
                            img[idx + 0] = tex; img[idx + 1] = 0.25f; img[idx + 2] = 0.08f;
                        } else {
                            float bg = 0.12f + 0.04f * std::sin(x * 0.23f) * std::cos(y * 0.21f);
                            img[idx + 0] = 0.04f; img[idx + 1] = bg; img[idx + 2] = bg + 0.18f;
                        }
                        img[idx + 3] = 1.0f;
                    }
                }
                return img;
            };
            // Parabolic motion: p(t) = (cx + 18*t, cy + 16*t^2) -> bends toward +y at t = +/-0.5
            auto cPrev = makeDiscAt(cx - 18.0f, cy + 16.0f);
            auto cCurr = makeDiscAt(cx, cy);
            auto cNext = makeDiscAt(cx + 18.0f, cy + 16.0f);
            auto cStaticBg = makeDiscAt(-100.0f, -100.0f); // occluded prev

            RenderParams cp;
            cp.width = cw; cp.height = ch;
            cp.mode = eModeShutterBlur;
            cp.flowEngine = eFlowCudaPyramidal;
            cp.shutterAngle = 360.0f; // tau in [-0.5, +0.5]
            cp.shutterSamples = 48;
            cp.autoShutterSamples = false;

            std::vector<FrameInput> curveWin = {
                {0, cPrev.data(), 0, nullptr, 0, 0},
                {1, cCurr.data(), 0, nullptr, 0, 0},
                {2, cNext.data(), 0, nullptr, 0, 0}
            };
            std::vector<FrameInput> occWin = {
                {0, cStaticBg.data(), 0, nullptr, 0, 0},
                {1, cCurr.data(), 0, nullptr, 0, 0},
                {2, cNext.data(), 0, nullptr, 0, 0}
            };
            std::vector<float> curveOut(cCurr.size()), occOut(cCurr.size());
            FlowCache fc1, fc2;
            bool ok1 = CudaEngine::instance().renderHostFrames(&cp, curveWin, 1, curveOut.data(), cp, fc1);
            bool ok2 = CudaEngine::instance().renderHostFrames(&cp, occWin, 1, occOut.data(), cp, fc2);

            // Check at (cx - 9, cy + 21) vs (cx - 9, cy - 21): true parabola passes through +y side (cy + 4 + 20 = cy + 24)
            curvePlusY = curveOut[(static_cast<int>(cy + 21) * cw + static_cast<int>(cx - 9)) * 4 + 0];
            curveMinusY = curveOut[(static_cast<int>(cy - 21) * cw + static_cast<int>(cx - 9)) * 4 + 0];

            // With linear velocity v0 = (18, 0), check blur extension at (cx + 25, cy)
            unoccBlurR = curveOut[(static_cast<int>(cy) * cw + static_cast<int>(cx + 25)) * 4 + 0];
            occBlurR = occOut[(static_cast<int>(cy) * cw + static_cast<int>(cx + 25)) * 4 + 0];
            curveAndOccOk = ok1 && ok2 && (curvePlusY > curveMinusY + 0.08f) && (occBlurR > 0.15f);
        }

        // 11D: Multi-subframe integration (5 real sub-frames) & eTimebaseClipFps scaling
        bool subFrameAndTimebaseOk = false;
        double subGpuCpuDiff = 999.0;
        float clipTimebaseSpanRatio = 0.0f;
        {
            const int sw = 256, sh = 128;
            const float totalShift = 40.0f; // 40 px per timeline frame = 8 px per sub-frame at 5:1
            auto makeBar = [&](float cx) {
                std::vector<float> img(static_cast<size_t>(sw) * sh * 4, 0.0f);
                for (int y = 0; y < sh; ++y) {
                    for (int x = 0; x < sw; ++x) {
                        size_t idx = (static_cast<size_t>(y) * sw + x) * 4;
                        bool inBar = (y >= 32 && y < 96 && std::fabs(x - cx) <= 10.0f);
                        img[idx + 0] = inBar ? 0.90f : 0.10f;
                        img[idx + 1] = inBar ? 0.30f : 0.15f;
                        img[idx + 2] = inBar ? 0.08f : 0.35f;
                        img[idx + 3] = 1.0f;
                    }
                }
                return img;
            };

            const float rels[5] = {-0.4f, -0.2f, 0.0f, 0.2f, 0.4f};
            std::vector<std::vector<float>> subHost(5);
            std::vector<FrameInput> subCpu(5), subGpu(5);
            std::vector<uint64_t> subDev(5, 0);
            const size_t bytes = static_cast<size_t>(sw) * sh * 4 * sizeof(float);
            for (int j = 0; j < 5; ++j) {
                subHost[j] = makeBar(128.0f + rels[j] * totalShift);
                subCpu[j] = {10 + j, subHost[j].data(), 0, nullptr, 0, 0, rels[j]};
                subDev[j] = CudaEngine::instance().allocDevice(bytes);
                CudaEngine::instance().uploadToDevice(subDev[j], subHost[j].data(), bytes);
                subGpu[j] = {10 + j, nullptr, subDev[j], 0, 0, 0, rels[j]};
            }
            uint64_t devOut1 = CudaEngine::instance().allocDevice(bytes);
            uint64_t devOut2 = CudaEngine::instance().allocDevice(bytes);
            CudaEngine::instance().syncStream(nullptr);

            RenderParams sp;
            sp.width = sw; sp.height = sh;
            sp.mode = eModeShutterBlur;
            sp.flowEngine = eFlowCudaPyramidal;
            sp.shutterAngle = 180.0f;
            sp.shutterSamples = 24;
            sp.shutterTimebase = eTimebaseTimelineFps;
            sp.timebaseScale = 1.0f;

            FlowCache gc, cc;
            std::vector<float> gpuOut(sw * sh * 4), cpuOut(sw * sh * 4), gpuClipTbOut(sw * sh * 4);
            bool okG = CudaEngine::instance().renderFrame(&sp, subGpu, 2, devOut1, sp, gc, nullptr, &subGpu);
            bool okC = CpuFallbackProcessor::renderFrame(&sp, subCpu, 2, cpuOut.data(), sp, cc, &subCpu);

            RenderParams spClip = sp;
            spClip.shutterTimebase = eTimebaseClipFps;
            spClip.timebaseScale = 0.2f; // 24fps / 120fps
            bool okClip = CudaEngine::instance().renderFrame(&spClip, subGpu, 2, devOut2, spClip, gc, nullptr, &subGpu);

            CudaEngine::instance().downloadFromDevice(gpuOut.data(), devOut1, bytes);
            CudaEngine::instance().downloadFromDevice(gpuClipTbOut.data(), devOut2, bytes);
            CudaEngine::instance().syncStream(nullptr);

            for (uint64_t ptr : subDev) CudaEngine::instance().freeDevice(ptr);
            CudaEngine::instance().freeDevice(devOut1);
            CudaEngine::instance().freeDevice(devOut2);

            double diffSum = 0.0;
            for (size_t i = 0; i < gpuOut.size(); ++i) diffSum += std::abs(gpuOut[i] - cpuOut[i]);
            subGpuCpuDiff = diffSum / gpuOut.size();

            // At x = 128 + 16 (16px from center: inside 180 deg timeline blur [-10-10 .. +10+10], outside 0.2x clip blur [-12 .. +12])
            float tlR = gpuOut[(64 * sw + 144) * 4 + 0];
            float clR = gpuClipTbOut[(64 * sw + 144) * 4 + 0];
            clipTimebaseSpanRatio = tlR - clR;
            subFrameAndTimebaseOk = okG && okC && okClip && (subGpuCpuDiff < 0.03) && (tlR > 0.25f) && (clR < 0.14f);
        }

        char t11Buf[384];
        std::snprintf(t11Buf, sizeof(t11Buf),
                      "11A(RMSE G=%.4f C=%.4f M=%.4f edge=%.3f ext=%.3f thin=%.3f) | "
                      "11B(140px EPE N=%.2f P=%.2f) | 11C(curve +y=%.3f/-y=%.3f occ=%.3f) | "
                      "11D(subDiff=%.4f tbDelta=%.3f)",
                      rmseGpuNoMatte, rmseCpuNoMatte, rmseGpuMatte,
                      interiorEdgeR, exteriorRampR, thinBarCenterR,
                      epe140Neural, epe140Pyr, curvePlusY, curveMinusY, occBlurR,
                      subGpuCpuDiff, clipTimebaseSpanRatio);
        report("Test 11: Physical 180 deg Shutter, >120px Flow, Curve/Occlusion & Sub-Frame/Timebase",
               profileOk && largeDispOk && curveAndOccOk && subFrameAndTimebaseOk, t11Buf);
    }

    // Test 12: Long-Exposure Curved Arcs, Foreground Matte Isolation, Wake Backfill & Sub-Frame TimeStep
    {
        // 12A: Curved circular point-light trail (GPU & CPU) vs straight-line chord
        bool curveTrailOk = false;
        float arcMidR = 0.0f, chordInR = 1.0f;
        double arcGpuCpuDiff = 999.0;
        {
            const int cw = 256, ch = 256;
            const float cx = 128.0f, cy = 128.0f, R = 52.0f, dTheta = 0.55f;
            const int nF = 5;
            std::vector<std::vector<float>> seq(nF);
            std::vector<FrameInput> win(nF);
            for (int f = 0; f < nF; ++f) {
                float th = f * dTheta;
                seq[f] = makePointLightFrame(cw, ch, cx + R * std::cos(th), cy + R * std::sin(th), 5.0f, 1.0f);
                win[f] = {f, seq[f].data(), 0, nullptr, 0, 0};
            }
            RenderParams cp;
            cp.width = cw; cp.height = ch;
            cp.mode = eModeLongExposureTrail;
            cp.flowEngine = eFlowCudaPyramidal;
            cp.trailFrames = nF;
            cp.subFrameSteps = 8;
            cp.trailBlendMode = eTrailLightPainting;
            cp.motionThreshold = 1.0f;
            cp.lumaThreshold = 0.15f;

            FlowCache gc, cc;
            std::vector<float> gpuOut(cw * ch * 4), cpuOut(cw * ch * 4);
            bool okG = CudaEngine::instance().renderHostFrames(&cp, win, nF - 1, gpuOut.data(), cp, gc);
            bool okC = CpuFallbackProcessor::renderFrame(&cp, win, nF - 1, cpuOut.data(), cp, cc);

            // Check midpoint between frame 2 and frame 3: true circle at R=52 vs chord interior at R*cos(0.275)*0.88
            float midTh = 2.5f * dTheta;
            int ax = static_cast<int>(std::lround(cx + R * std::cos(midTh)));
            int ay = static_cast<int>(std::lround(cy + R * std::sin(midTh)));
            int chx = static_cast<int>(std::lround(cx + (R * std::cos(0.5f * dTheta) - 5.0f) * std::cos(midTh)));
            int chy = static_cast<int>(std::lround(cy + (R * std::cos(0.5f * dTheta) - 5.0f) * std::sin(midTh)));
            arcMidR = gpuOut[(ay * cw + ax) * 4 + 0];
            chordInR = gpuOut[(chy * cw + chx) * 4 + 0];

            double diff = 0.0;
            for (size_t i = 0; i < gpuOut.size(); ++i) diff += std::abs(gpuOut[i] - cpuOut[i]);
            arcGpuCpuDiff = diff / gpuOut.size();
            curveTrailOk = okG && okC && (arcMidR > 0.45f) && (arcMidR > chordInR + 0.25f) && (arcGpuCpuDiff < 0.04);
        }

        // 12B: eMatteForegroundIsolation keeps shaky high-contrast background crisp while trailing foreground
        bool fgIsoOk = false;
        float bgMaxSmearPhys = 1.0f, bgMaxSmearLP = 1.0f, fgTrailR = 0.0f;
        double fgIsoGpuCpuDiff = 999.0;
        {
            const int mw = 256, mh = 192;
            const int nF = 4;
            std::vector<std::vector<float>> seq(nF), mattes(nF);
            std::vector<FrameInput> win(nF);
            for (int f = 0; f < nF; ++f) {
                seq[f].resize(static_cast<size_t>(mw) * mh * 4);
                mattes[f].resize(static_cast<size_t>(mw) * mh);
                float discX = 70.0f + f * 24.0f;
                float discY = 96.0f;
                float bgShiftX = f * 8.0f; // camera shake on background
                for (int y = 0; y < mh; ++y) {
                    for (int x = 0; x < mw; ++x) {
                        size_t p = static_cast<size_t>(y) * mw + x;
                        float d = std::hypot(x - discX, y - discY);
                        if (d <= 16.0f) {
                            seq[f][p * 4 + 0] = 0.72f; seq[f][p * 4 + 1] = 0.20f; seq[f][p * 4 + 2] = 0.06f;
                            mattes[f][p] = 1.0f;
                        } else {
                            float stripe = 0.12f + 0.10f * std::sin((x - bgShiftX) * 0.28f) * std::cos(y * 0.15f);
                            seq[f][p * 4 + 0] = 0.04f; seq[f][p * 4 + 1] = stripe; seq[f][p * 4 + 2] = stripe + 0.16f;
                            mattes[f][p] = 0.0f;
                        }
                        seq[f][p * 4 + 3] = 1.0f;
                    }
                }
                win[f] = {f, seq[f].data(), 0, mattes[f].data(), 0, 1};
            }

            RenderParams mp;
            mp.width = mw; mp.height = mh;
            mp.mode = eModeLongExposureTrail;
            mp.flowEngine = eFlowCudaPyramidal;
            mp.matteMode = eMatteForegroundIsolation;
            mp.trailFrames = nF;
            mp.subFrameSteps = 6;
            mp.trailBlendMode = eTrailPhysicalExposure;
            mp.highlightBoost = 0.0f;

            FlowCache gc1, gc2, cc1;
            std::vector<float> outPhys(mw * mh * 4), outCpuPhys(mw * mh * 4), outLP(mw * mh * 4);
            bool okP = CudaEngine::instance().renderHostFrames(&mp, win, nF - 1, outPhys.data(), mp, gc1);
            bool okCP = CpuFallbackProcessor::renderFrame(&mp, win, nF - 1, outCpuPhys.data(), mp, cc1);

            RenderParams lp = mp;
            lp.trailBlendMode = eTrailLightPainting;
            bool okL = CudaEngine::instance().renderHostFrames(&lp, win, nF - 1, outLP.data(), lp, gc2);

            bgMaxSmearPhys = 0.0f; bgMaxSmearLP = 0.0f;
            for (int y = 16; y < 48; ++y) {
                for (int x = 16; x < mw - 16; ++x) {
                    size_t idx = (static_cast<size_t>(y) * mw + x) * 4;
                    for (int c = 0; c < 3; ++c) {
                        bgMaxSmearPhys = std::max(bgMaxSmearPhys, std::abs(outPhys[idx + c] - seq[nF - 1][idx + c]));
                        bgMaxSmearLP = std::max(bgMaxSmearLP, std::abs(outLP[idx + c] - seq[nF - 1][idx + c]));
                    }
                }
            }
            // Foreground trail left behind at x = 94 (frame 1 disc center) while curr disc is at x = 142
            fgTrailR = outPhys[(96 * mw + 94) * 4 + 0];
            double diff = 0.0;
            for (size_t i = 0; i < outPhys.size(); ++i) diff += std::abs(outPhys[i] - outCpuPhys[i]);
            fgIsoGpuCpuDiff = diff / outPhys.size();
            fgIsoOk = okP && okCP && okL && (bgMaxSmearPhys < 1e-3f) && (bgMaxSmearLP < 1e-3f) &&
                      (fgTrailR > 0.14f) && (fgIsoGpuCpuDiff < 0.02);
        }

        // 12C: Short-window high-decay wake backfill & sub-frame pairTimeStep Comet scaling
        bool wakeAndStepOk = false;
        float wakeR = 1.0f, subStepDecayDiff = 999.0f;
        {
            const int ww = 192, wh = 128;
            auto makeBar = [&](float cx) {
                std::vector<float> img(static_cast<size_t>(ww) * wh * 4, 0.0f);
                for (int y = 0; y < wh; ++y) {
                    for (int x = 0; x < ww; ++x) {
                        size_t idx = (static_cast<size_t>(y) * ww + x) * 4;
                        bool inBar = (y >= 32 && y < 96 && std::fabs(x - cx) <= 10.0f);
                        img[idx + 0] = inBar ? 0.50f : 0.05f;
                        img[idx + 1] = inBar ? 0.20f : 0.12f;
                        img[idx + 2] = inBar ? 0.08f : 0.28f;
                        img[idx + 3] = 1.0f;
                    }
                }
                return img;
            };
            auto f0 = makeBar(64.0f), f1 = makeBar(100.0f);
            std::vector<FrameInput> win = {
                {0, f0.data(), 0, nullptr, 0, 0},
                {1, f1.data(), 0, nullptr, 0, 0}
            };
            RenderParams wp;
            wp.width = ww; wp.height = wh;
            wp.mode = eModeLongExposureTrail;
            wp.flowEngine = eFlowCudaPyramidal;
            wp.matteMode = eMatteIgnore;
            wp.trailFrames = 2;
            wp.subFrameSteps = 6;
            wp.trailBlendMode = eTrailCometDecay;
            wp.trailDecay = 1.0f;
            wp.rearCurtainCrispness = 0.0f;
            wp.highlightBoost = 0.0f;

            FlowCache gc1, gc2;
            std::vector<float> outComet(ww * wh * 4), outSubStep(ww * wh * 4);
            bool ok1 = CudaEngine::instance().renderHostFrames(&wp, win, 1, outComet.data(), wp, gc1);
            // At x = 109 (leading edge of curr bar with rearCurtainCrispness=0): acc.w < 0.25 triggers wake backfill from lastRGBA (bg 0.05)
            wakeR = outComet[(64 * ww + 109) * 4 + 0];

            RenderParams wpSub = wp;
            wpSub.pairTimeStep = 0.2f;
            bool ok2 = CudaEngine::instance().renderHostFrames(&wpSub, win, 1, outSubStep.data(), wpSub, gc2);
            // With 0.2x time step, decay is gentler so x = 76 tail is brighter than 1.0x time step
            subStepDecayDiff = outSubStep[(64 * ww + 76) * 4 + 0] - outComet[(64 * ww + 76) * 4 + 0];
            wakeAndStepOk = ok1 && ok2 && (wakeR < 0.38f) && (subStepDecayDiff > 0.02f);
        }

        char t12Buf[320];
        std::snprintf(t12Buf, sizeof(t12Buf),
                      "12A(arcR=%.3f chordInR=%.3f diff=%.4f) | 12B(smearP=%.1e smearL=%.1e fgR=%.3f diff=%.4f) | 12C(wakeR=%.3f subDelta=%.3f)",
                      arcMidR, chordInR, arcGpuCpuDiff, bgMaxSmearPhys, bgMaxSmearLP, fgTrailR, fgIsoGpuCpuDiff, wakeR, subStepDecayDiff);
        report("Test 12: Long-Exposure Curved Arcs, Matte Isolation, Wake Backfill & Sub-Frame Step",
               curveTrailOk && fgIsoOk && wakeAndStepOk, t12Buf);
    }

    // Test 7: Graceful Shutdown & Re-init Cycle
    {
        uint64_t ptr = CudaEngine::instance().allocDevice(4096);
        FlowCache::global().clear();
        CudaEngine::instance().shutdown();
        NeuralFlowEngine::instance().shutdown();

        // Calling freeDevice / invalidateInstance after shutdown must be a safe no-op
        CudaEngine::instance().freeDevice(ptr);
        FlowCache::global().invalidateInstance(reinterpret_cast<const void*>(0x1234));

        bool reCu = CudaEngine::instance().initialize();
        bool reOrt = NeuralFlowEngine::instance().initialize();

        // Clean final shutdown before process exit
        FlowCache::global().clear();
        CudaEngine::instance().shutdown();
        NeuralFlowEngine::instance().shutdown();

        report("Test 7: Graceful Shutdown & Re-init Cycle",
               ptr != 0 && reCu && reOrt,
               "Shutdown -> stale freeDevice no-op -> re-init OK");
    }

    std::cout << "\nSummary: " << (g_failures == 0 ? "ALL TESTS PASSED" : std::to_string(g_failures) + " FAILED") << "\n";
    return g_failures == 0 ? 0 : 1;
}
