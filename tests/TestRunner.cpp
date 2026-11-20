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

            std::vector<Float4> hostFlow(static_cast<size_t>(w) * h);
            CudaEngine::instance().downloadFromDevice(hostFlow.data(), entry->devFlow4, hostFlow.size() * sizeof(Float4));
            CudaEngine::instance().syncStream(nullptr);

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
