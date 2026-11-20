#include "CpuFallbackProcessor.h"
#include "NeuralFlowEngine.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace bmb {

namespace {

inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline float clampf(float v, float lo, float hi) {
    return std::min(hi, std::max(lo, v));
}

inline float smoothstepf(float edge0, float edge1, float x) {
    float t = clampf((x - edge0) / std::max(1e-6f, edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

constexpr float kDiA = 0.0075f;
constexpr float kDiB = 7.0f;
constexpr float kDiC = 0.07329248f;
constexpr float kDiM = 10.44426855f;
constexpr float kDiLinCut = 0.00262409f;
constexpr float kDiLogCut = 0.02740668f;

inline float dwiToLinearChannel(float v) {
    return (v <= kDiLogCut) ? (v / kDiM) : (std::exp2(v / kDiC - kDiB) - kDiA);
}

inline float linearToDwiChannel(float l) {
    return (l <= kDiLinCut) ? (l * kDiM) : ((std::log2(l + kDiA) + kDiB) * kDiC);
}

inline float dwgLuminance(float r, float g, float b) {
    return std::max(0.0f, 0.27411851f * r + 0.87363190f * g - 0.14775041f * b);
}

inline Float4 sampleRGBABilinear(const float* img, int w, int h, float x, float y) {
    float cx = clampf(x, 0.0f, static_cast<float>(w - 1));
    float cy = clampf(y, 0.0f, static_cast<float>(h - 1));
    int x0 = static_cast<int>(std::floor(cx));
    int y0 = static_cast<int>(std::floor(cy));
    int x1 = std::min(x0 + 1, w - 1);
    int y1 = std::min(y0 + 1, h - 1);
    float fx = cx - static_cast<float>(x0);
    float fy = cy - static_cast<float>(y0);

    const float* p00 = img + (y0 * w + x0) * 4;
    const float* p10 = img + (y0 * w + x1) * 4;
    const float* p01 = img + (y1 * w + x0) * 4;
    const float* p11 = img + (y1 * w + x1) * 4;

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    return {
        p00[0] * w00 + p10[0] * w10 + p01[0] * w01 + p11[0] * w11,
        p00[1] * w00 + p10[1] * w10 + p01[1] * w01 + p11[1] * w11,
        p00[2] * w00 + p10[2] * w10 + p01[2] * w01 + p11[2] * w11,
        p00[3] * w00 + p10[3] * w10 + p01[3] * w01 + p11[3] * w11
    };
}

inline float sampleFloat1Bilinear(const float* img, int w, int h, float x, float y) {
    float cx = clampf(x, 0.0f, static_cast<float>(w - 1));
    float cy = clampf(y, 0.0f, static_cast<float>(h - 1));
    int x0 = static_cast<int>(std::floor(cx));
    int y0 = static_cast<int>(std::floor(cy));
    int x1 = std::min(x0 + 1, w - 1);
    int y1 = std::min(y0 + 1, h - 1);
    float fx = cx - static_cast<float>(x0);
    float fy = cy - static_cast<float>(y0);

    float v00 = img[y0 * w + x0];
    float v10 = img[y0 * w + x1];
    float v01 = img[y1 * w + x0];
    float v11 = img[y1 * w + x1];
    return (v00 * (1.0f - fx) + v10 * fx) * (1.0f - fy) + (v01 * (1.0f - fx) + v11 * fx) * fy;
}

inline Float2 sampleFloat2Bilinear(const std::vector<Float2>& img, int w, int h, float x, float y) {
    float cx = clampf(x, 0.0f, static_cast<float>(w - 1));
    float cy = clampf(y, 0.0f, static_cast<float>(h - 1));
    int x0 = static_cast<int>(std::floor(cx));
    int y0 = static_cast<int>(std::floor(cy));
    int x1 = std::min(x0 + 1, w - 1);
    int y1 = std::min(y0 + 1, h - 1);
    float fx = cx - static_cast<float>(x0);
    float fy = cy - static_cast<float>(y0);

    const Float2& v00 = img[y0 * w + x0];
    const Float2& v10 = img[y0 * w + x1];
    const Float2& v01 = img[y1 * w + x0];
    const Float2& v11 = img[y1 * w + x1];
    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;
    return {
        v00.x * w00 + v10.x * w10 + v01.x * w01 + v11.x * w11,
        v00.y * w00 + v10.y * w10 + v01.y * w01 + v11.y * w11
    };
}

void extractLumaAndMatteCPU(int w, int h, const float* rgba, const float* matte, int matteComp,
                             std::vector<float>& outLuma, std::vector<float>& outMatte) {
    const size_t n = static_cast<size_t>(w) * h;
    outLuma.resize(n);
    outMatte.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const float* p = rgba + i * 4;
        outLuma[i] = dwgLuminance(p[0], p[1], p[2]);
        if (matte && matteComp > 0) {
            if (matteComp == 1) {
                outMatte[i] = clampf(matte[i], 0.0f, 1.0f);
            } else {
                const float* m = matte + i * 4;
                float mv = (m[3] > 0.0f && m[3] < 0.9999f) ? m[3] : (0.2126f * m[0] + 0.7152f * m[1] + 0.0722f * m[2]);
                outMatte[i] = clampf(mv, 0.0f, 1.0f);
            }
        } else {
            outMatte[i] = 1.0f;
        }
    }
}

std::vector<Float2> computeSingleFlowCPU(
    const float* rgbaA, const float* matteA, int matteAComp,
    const float* rgbaB, const float* matteB, int matteBComp,
    int width, int height, FlowEngineType engineType, float flowDetail, MatteMode matteMode)
{
    const int kLevels = 5;
    int lw[kLevels], lh[kLevels];
    std::vector<float> lumaA[kLevels], lumaB[kLevels], mattePyrA[kLevels];

    lw[0] = width;
    lh[0] = height;
    extractLumaAndMatteCPU(width, height, rgbaA, matteA, matteAComp, lumaA[0], mattePyrA[0]);

    std::vector<float> dummyMatteB;
    extractLumaAndMatteCPU(width, height, rgbaB, matteB, matteBComp, lumaB[0], dummyMatteB);

    for (int i = 1; i < kLevels; ++i) {
        lw[i] = std::max(16, lw[i - 1] / 2);
        lh[i] = std::max(16, lh[i - 1] / 2);
        lumaA[i].resize(static_cast<size_t>(lw[i]) * lh[i]);
        lumaB[i].resize(static_cast<size_t>(lw[i]) * lh[i]);
        mattePyrA[i].resize(static_cast<size_t>(lw[i]) * lh[i]);
        for (int y = 0; y < lh[i]; ++y) {
            int sy = std::min(y * 2, lh[i - 1] - 1);
            int sy1 = std::min(sy + 1, lh[i - 1] - 1);
            for (int x = 0; x < lw[i]; ++x) {
                int sx = std::min(x * 2, lw[i - 1] - 1);
                int sx1 = std::min(sx + 1, lw[i - 1] - 1);
                lumaA[i][y * lw[i] + x] = 0.25f * (lumaA[i - 1][sy * lw[i - 1] + sx] + lumaA[i - 1][sy * lw[i - 1] + sx1] +
                                                   lumaA[i - 1][sy1 * lw[i - 1] + sx] + lumaA[i - 1][sy1 * lw[i - 1] + sx1]);
                lumaB[i][y * lw[i] + x] = 0.25f * (lumaB[i - 1][sy * lw[i - 1] + sx] + lumaB[i - 1][sy * lw[i - 1] + sx1] +
                                                   lumaB[i - 1][sy1 * lw[i - 1] + sx] + lumaB[i - 1][sy1 * lw[i - 1] + sx1]);
                mattePyrA[i][y * lw[i] + x] = 0.25f * (mattePyrA[i - 1][sy * lw[i - 1] + sx] + mattePyrA[i - 1][sy * lw[i - 1] + sx1] +
                                                       mattePyrA[i - 1][sy1 * lw[i - 1] + sx] + mattePyrA[i - 1][sy1 * lw[i - 1] + sx1]);
            }
        }
    }

    bool hasMatte = (matteA != nullptr && matteAComp > 0 && matteMode == eMatteForegroundIsolation);
    float sigmaRange = 0.03f + (1.0f - flowDetail) * 0.12f;

    std::vector<Float2> flow;
    for (int lvl = kLevels - 1; lvl >= 0; --lvl) {
        int w = lw[lvl], h = lh[lvl];
        std::vector<Float2> inFlow(static_cast<size_t>(w) * h, {0.0f, 0.0f});
        if (lvl < kLevels - 1) {
            int pw = lw[lvl + 1], ph = lh[lvl + 1];
            float scaleX = static_cast<float>(w) / pw;
            float scaleY = static_cast<float>(h) / ph;
            for (int y = 0; y < h; ++y) {
                float sy = (y + 0.5f) / scaleY - 0.5f;
                for (int x = 0; x < w; ++x) {
                    float sx = (x + 0.5f) / scaleX - 0.5f;
                    Float2 f = sampleFloat2Bilinear(flow, pw, ph, sx, sy);
                    inFlow[y * w + x] = {f.x * scaleX, f.y * scaleY};
                }
            }
        }

        std::vector<Float2> outFlow(static_cast<size_t>(w) * h);
        int searchRadius = (lvl >= 3) ? 4 : ((lvl == 2) ? 2 : 1);
        int lkIters = (lvl <= 1) ? 4 : 3;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int fgC = (mattePyrA[lvl][y * w + x] >= 0.5f) ? 1 : 0;
                auto evalCost = [&](float vx, float vy) {
                    float cost = 0.0f, wsum = 0.0f;
                    for (int dy = -2; dy <= 2; ++dy) {
                        int ay = clampi(y + dy, 0, h - 1);
                        for (int dx = -2; dx <= 2; ++dx) {
                            int ax = clampi(x + dx, 0, w - 1);
                            float weight = (hasMatte && ((mattePyrA[lvl][ay * w + ax] >= 0.5f ? 1 : 0) != fgC)) ? 0.05f : 1.0f;
                            float a = lumaA[lvl][ay * w + ax];
                            float b = sampleFloat1Bilinear(lumaB[lvl].data(), w, h, ax + vx, ay + vy);
                            cost += weight * std::fabs(a - b);
                            wsum += weight;
                        }
                    }
                    return cost / std::max(1e-4f, wsum);
                };

                Float2 best = inFlow[y * w + x];
                float bestCost = evalCost(best.x, best.y);

                float c0 = evalCost(0.0f, 0.0f);
                if (c0 + 1e-4f < bestCost) {
                    bestCost = c0;
                    best = {0.0f, 0.0f};
                }

                const int offs[4][2] = {{-2, 0}, {2, 0}, {0, -2}, {0, 2}};
                for (int i = 0; i < 4; ++i) {
                    int nx = clampi(x + offs[i][0], 0, w - 1);
                    int ny = clampi(y + offs[i][1], 0, h - 1);
                    Float2 cand = inFlow[ny * w + nx];
                    float c = evalCost(cand.x, cand.y);
                    if (c + 1e-4f < bestCost) {
                        bestCost = c;
                        best = cand;
                    }
                }

                for (int r = searchRadius; r >= 1; r = (r > 2) ? (r / 2) : (r - 1)) {
                    Float2 base = best;
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            if (dx == 0 && dy == 0) continue;
                            Float2 cand = {base.x + dx * r, base.y + dy * r};
                            float c = evalCost(cand.x, cand.y);
                            if (c + 1e-4f < bestCost) {
                                bestCost = c;
                                best = cand;
                            }
                        }
                    }
                }

                float vx = best.x, vy = best.y;
                float Ixx = 0.0f, Ixy = 0.0f, Iyy = 0.0f;
                for (int dy = -2; dy <= 2; ++dy) {
                    int ay = clampi(y + dy, 0, h - 1);
                    int aym = clampi(ay - 1, 0, h - 1);
                    int ayp = clampi(ay + 1, 0, h - 1);
                    for (int dx = -2; dx <= 2; ++dx) {
                        int ax = clampi(x + dx, 0, w - 1);
                        int axm = clampi(ax - 1, 0, w - 1);
                        int axp = clampi(ax + 1, 0, w - 1);
                        float weight = (dx == 0 && dy == 0) ? 1.5f : 1.0f;
                        if (hasMatte && ((mattePyrA[lvl][ay * w + ax] >= 0.5f ? 1 : 0) != fgC)) weight = 0.02f;
                        float gx = 0.5f * (lumaA[lvl][ay * w + axp] - lumaA[lvl][ay * w + axm]);
                        float gy = 0.5f * (lumaA[lvl][ayp * w + ax] - lumaA[lvl][aym * w + ax]);
                        Ixx += weight * gx * gx;
                        Ixy += weight * gx * gy;
                        Iyy += weight * gy * gy;
                    }
                }
                float det = (Ixx + 0.0025f) * (Iyy + 0.0025f) - Ixy * Ixy;
                if (det > 1e-7f) {
                    float invDet = 1.0f / det;
                    for (int it = 0; it < lkIters; ++it) {
                        float bx = 0.0f, by = 0.0f;
                        for (int dy = -2; dy <= 2; ++dy) {
                            int ay = clampi(y + dy, 0, h - 1);
                            int aym = clampi(ay - 1, 0, h - 1);
                            int ayp = clampi(ay + 1, 0, h - 1);
                            for (int dx = -2; dx <= 2; ++dx) {
                                int ax = clampi(x + dx, 0, w - 1);
                                int axm = clampi(ax - 1, 0, w - 1);
                                int axp = clampi(ax + 1, 0, w - 1);
                                float weight = (dx == 0 && dy == 0) ? 1.5f : 1.0f;
                                if (hasMatte && ((mattePyrA[lvl][ay * w + ax] >= 0.5f ? 1 : 0) != fgC)) weight = 0.02f;
                                float gx = 0.5f * (lumaA[lvl][ay * w + axp] - lumaA[lvl][ay * w + axm]);
                                float gy = 0.5f * (lumaA[lvl][ayp * w + ax] - lumaA[lvl][aym * w + ax]);
                                float It = sampleFloat1Bilinear(lumaB[lvl].data(), w, h, ax + vx, ay + vy) - lumaA[lvl][ay * w + ax];
                                float wTot = weight * (0.08f / std::max(0.08f, std::fabs(It)));
                                bx -= wTot * gx * It;
                                by -= wTot * gy * It;
                            }
                        }
                        float dvx = invDet * ((Iyy + 0.0025f) * bx - Ixy * by);
                        float dvy = invDet * ((Ixx + 0.0025f) * by - Ixy * bx);
                        float sl = std::hypot(dvx, dvy);
                        if (sl > 1.25f) {
                            dvx *= 1.25f / sl;
                            dvy *= 1.25f / sl;
                        }
                        float cc = evalCost(vx + dvx, vy + dvy);
                        if (cc <= bestCost + 0.002f) {
                            vx += dvx;
                            vy += dvy;
                            bestCost = cc;
                        } else {
                            break;
                        }
                    }
                }
                outFlow[y * w + x] = {vx, vy};
            }
        }

        // Joint-Bilateral filter
        flow.resize(static_cast<size_t>(w) * h);
        float inv2S = 1.0f / 8.0f;
        float inv2R = 1.0f / (2.0f * sigmaRange * sigmaRange);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                int idx = y * w + x;
                float lc = lumaA[lvl][idx];
                int fgC = (mattePyrA[lvl][idx] >= 0.5f) ? 1 : 0;
                float sx = 0.0f, sy = 0.0f, sw = 0.0f;
                for (int dy = -2; dy <= 2; ++dy) {
                    int ny = clampi(y + dy, 0, h - 1);
                    for (int dx = -2; dx <= 2; ++dx) {
                        int nx = clampi(x + dx, 0, w - 1);
                        int nidx = ny * w + nx;
                        float dl = lc - lumaA[lvl][nidx];
                        float wgt = std::exp(-(dx * dx + dy * dy) * inv2S - dl * dl * inv2R);
                        if (hasMatte && ((mattePyrA[lvl][nidx] >= 0.5f ? 1 : 0) != fgC)) wgt *= 0.001f;
                        sx += wgt * outFlow[nidx].x;
                        sy += wgt * outFlow[nidx].y;
                        sw += wgt;
                    }
                }
                flow[idx] = {sx / std::max(1e-6f, sw), sy / std::max(1e-6f, sw)};
            }
        }
    }

    if (engineType == eFlowNeural && NeuralFlowEngine::instance().initialize()) {
        const size_t n = static_cast<size_t>(width) * height;
        std::vector<float> nchw4(4 * n), nchw2(2 * n);
        for (int y = 0; y < height; ++y) {
            int ym = std::max(0, y - 1), yp = std::min(height - 1, y + 1);
            for (int x = 0; x < width; ++x) {
                int xm = std::max(0, x - 1), xp = std::min(width - 1, x + 1);
                size_t i = static_cast<size_t>(y) * width + x;
                nchw4[0 * n + i] = flow[i].x;
                nchw4[1 * n + i] = flow[i].y;
                nchw4[2 * n + i] = 0.5f * (lumaA[0][y * width + xp] - lumaA[0][y * width + xm]);
                nchw4[3 * n + i] = 0.5f * (lumaA[0][yp * width + x] - lumaA[0][ym * width + x]);
            }
        }
        if (NeuralFlowEngine::instance().refineFlow(width, height, nchw4.data(), nchw2.data())) {
            for (size_t i = 0; i < n; ++i) {
                flow[i].x = nchw2[0 * n + i];
                flow[i].y = nchw2[1 * n + i];
            }
        }
    }

    return flow;
}

void hsvToRgbCPU(float h, float s, float v, float& r, float& g, float& b) {
    float c = v * s;
    float hp = std::fmod(h * 6.0f, 6.0f);
    float x = c * (1.0f - std::fabs(std::fmod(hp, 2.0f) - 1.0f));
    float m = v - c;
    if (hp < 1.0f)      { r = c; g = x; b = 0; }
    else if (hp < 2.0f) { r = x; g = c; b = 0; }
    else if (hp < 3.0f) { r = 0; g = c; b = x; }
    else if (hp < 4.0f) { r = 0; g = x; b = c; }
    else if (hp < 5.0f) { r = x; g = 0; b = c; }
    else                { r = c; g = 0; b = x; }
    r += m; g += m; b += m;
}

} // namespace

std::shared_ptr<FlowEntry> CpuFallbackProcessor::getOrComputeFlowPair(
    const void* instanceId,
    int frameAIndex, const float* rgbaA, const float* matteA, int matteAComp,
    int frameBIndex, const float* rgbaB, const float* matteB, int matteBComp,
    const RenderParams& params,
    FlowCache& cache)
{
    FlowCacheKey key{
        instanceId,
        frameAIndex,
        frameBIndex,
        params.width,
        params.height,
        static_cast<int>(params.flowEngine),
        static_cast<int>(std::lround(params.flowDetail * 20.0f)),
        false
    };

    if (auto cached = cache.get(key)) {
        return cached;
    }

    auto fwd = computeSingleFlowCPU(rgbaA, matteA, matteAComp, rgbaB, matteB, matteBComp,
                                    params.width, params.height, params.flowEngine, params.flowDetail, params.matteMode);
    auto bwd = computeSingleFlowCPU(rgbaB, matteB, matteBComp, rgbaA, matteA, matteAComp,
                                    params.width, params.height, params.flowEngine, params.flowDetail, params.matteMode);

    std::vector<float> dummyLuma, matteValA;
    extractLumaAndMatteCPU(params.width, params.height, rgbaA, matteA, matteAComp, dummyLuma, matteValA);

    const size_t n = static_cast<size_t>(params.width) * params.height;
    auto entry = std::make_shared<FlowEntry>();
    entry->width = params.width;
    entry->height = params.height;
    entry->isGpu = false;
    entry->hostFlow4.resize(n);
    entry->hostConf2.resize(n);
    entry->byteSize = n * (sizeof(Float4) + sizeof(Float2));

    for (int y = 0; y < params.height; ++y) {
        for (int x = 0; x < params.width; ++x) {
            size_t i = static_cast<size_t>(y) * params.width + x;
            Float2 f = fwd[i];
            Float2 bDest = sampleFloat2Bilinear(bwd, params.width, params.height, x + f.x, y + f.y);
            Float2 bLoc = bwd[i];
            float errSq = (f.x + bDest.x) * (f.x + bDest.x) + (f.y + bDest.y) * (f.y + bDest.y);
            float conf = std::exp(-errSq / 4.0f);
            entry->hostFlow4[i] = {f.x, f.y, bLoc.x, bLoc.y};
            entry->hostConf2[i] = {conf, matteValA[i]};
        }
    }

    cache.put(key, entry);
    return entry;
}

bool CpuFallbackProcessor::renderFrame(
    const void* instanceId,
    const std::vector<FrameInput>& frames,
    int centerIdx,
    float* dstRGBA,
    const RenderParams& params,
    FlowCache& cache)
{
    if (!dstRGBA || centerIdx < 0 || centerIdx >= static_cast<int>(frames.size())) return false;
    const int w = params.width;
    const int h = params.height;
    const size_t nPix = static_cast<size_t>(w) * h;
    const FrameInput& curr = frames[centerIdx];
    if (!curr.hostRGBA) return false;

    std::vector<float> dummyLuma, currMatte;
    extractLumaAndMatteCPU(w, h, curr.hostRGBA, curr.hostMatte, curr.matteComponents, dummyLuma, currMatte);
    bool hasMatte = (curr.hostMatte != nullptr && curr.matteComponents > 0 && params.matteMode != eMatteIgnore);

    std::shared_ptr<FlowEntry> flowFwd, flowBwd;
    if (centerIdx + 1 < static_cast<int>(frames.size()) && frames[centerIdx + 1].hostRGBA) {
        flowFwd = getOrComputeFlowPair(instanceId,
                                       curr.frameIndex, curr.hostRGBA, curr.hostMatte, curr.matteComponents,
                                       frames[centerIdx + 1].frameIndex, frames[centerIdx + 1].hostRGBA, frames[centerIdx + 1].hostMatte, frames[centerIdx + 1].matteComponents,
                                       params, cache);
    }
    if (centerIdx - 1 >= 0 && frames[centerIdx - 1].hostRGBA) {
        flowBwd = getOrComputeFlowPair(instanceId,
                                       curr.frameIndex, curr.hostRGBA, curr.hostMatte, curr.matteComponents,
                                       frames[centerIdx - 1].frameIndex, frames[centerIdx - 1].hostRGBA, frames[centerIdx - 1].hostMatte, frames[centerIdx - 1].matteComponents,
                                       params, cache);
    }

    if (params.viewMode != eViewFinal) {
        auto activeFlow = flowFwd ? flowFwd : flowBwd;
        for (size_t i = 0; i < nPix; ++i) {
            const float* p = curr.hostRGBA + i * 4;
            Float4 f = activeFlow ? activeFlow->hostFlow4[i] : Float4{0, 0, 0, 0};
            Float2 c = activeFlow ? activeFlow->hostConf2[i] : Float2{1, 1};
            float mag = std::hypot(f.x * params.motionSensitivity, f.y * params.motionSensitivity);
            if (params.viewMode == eViewFlowHSV) {
                float hue = (std::atan2(f.y, f.x) + 3.14159265f) / (2.0f * 3.14159265f);
                float r, g, b;
                hsvToRgbCPU(hue, 0.95f, clampf(mag / 24.0f, 0.0f, 1.0f), r, g, b);
                dstRGBA[i * 4 + 0] = r; dstRGBA[i * 4 + 1] = g; dstRGBA[i * 4 + 2] = b; dstRGBA[i * 4 + 3] = p[3];
            } else if (params.viewMode == eViewConfidence) {
                dstRGBA[i * 4 + 0] = 1.0f - c.x; dstRGBA[i * 4 + 1] = c.x; dstRGBA[i * 4 + 2] = 0.15f; dstRGBA[i * 4 + 3] = p[3];
            } else {
                float luma = dwgLuminance(p[0], p[1], p[2]);
                float mg = params.motionThreshold > 0 ? smoothstepf(params.motionThreshold, params.motionThreshold + 0.75f, mag) : 1.0f;
                float lg = params.lumaThreshold > 0 ? smoothstepf(params.lumaThreshold, params.lumaThreshold + 0.12f, luma) : 1.0f;
                float m = mg * lg * c.y;
                dstRGBA[i * 4 + 0] = m; dstRGBA[i * 4 + 1] = m; dstRGBA[i * 4 + 2] = m; dstRGBA[i * 4 + 3] = p[3];
            }
        }
        return true;
    }

    std::vector<float> shutterOut;
    const float* shutterSrc = curr.hostRGBA;

    if (params.mode == eModeShutterBlur || params.mode == eModeCombined) {
        shutterOut.resize(nPix * 4);
        const int tileSize = 16;
        const int tileW = (w + tileSize - 1) / tileSize;
        const int tileH = (h + tileSize - 1) / tileSize;
        std::vector<Float2> tileMax(static_cast<size_t>(tileW) * tileH, {0.0f, 0.0f});
        std::vector<Float2> neighMax(static_cast<size_t>(tileW) * tileH, {0.0f, 0.0f});

        for (int ty = 0; ty < tileH; ++ty) {
            for (int tx = 0; tx < tileW; ++tx) {
                float bestM = 0.0f;
                Float2 bestV{0.0f, 0.0f};
                for (int y = ty * tileSize; y < std::min(h, (ty + 1) * tileSize); ++y) {
                    for (int x = tx * tileSize; x < std::min(w, (tx + 1) * tileSize); ++x) {
                        size_t idx = static_cast<size_t>(y) * w + x;
                        Float2 vf = flowFwd ? Float2{flowFwd->hostFlow4[idx].x, flowFwd->hostFlow4[idx].y} : Float2{0, 0};
                        Float2 vb = flowBwd ? Float2{flowBwd->hostFlow4[idx].x, flowBwd->hostFlow4[idx].y} : Float2{0, 0};
                        Float2 v0{0.5f * (vf.x - vb.x) * params.motionSensitivity, 0.5f * (vf.y - vb.y) * params.motionSensitivity};
                        float m = std::hypot(v0.x, v0.y);
                        float g = smoothstepf(params.motionThreshold, params.motionThreshold + 0.75f, m);
                        if (m * g > bestM) {
                            bestM = m * g;
                            bestV = {v0.x * g, v0.y * g};
                        }
                    }
                }
                tileMax[ty * tileW + tx] = bestV;
            }
        }

        for (int ty = 0; ty < tileH; ++ty) {
            for (int tx = 0; tx < tileW; ++tx) {
                float bestM = 0.0f;
                Float2 bestV{0, 0};
                for (int dy = -1; dy <= 1; ++dy) {
                    int ny = clampi(ty + dy, 0, tileH - 1);
                    for (int dx = -1; dx <= 1; ++dx) {
                        int nx = clampi(tx + dx, 0, tileW - 1);
                        Float2 v = tileMax[ny * tileW + nx];
                        float m = std::hypot(v.x, v.y);
                        if (m > bestM) { bestM = m; bestV = v; }
                    }
                }
                neighMax[ty * tileW + tx] = bestV;
            }
        }

        float shutterSpan = params.shutterAngle / 360.0f;
        float halfSpan = 0.5f * shutterSpan;
        float phaseCenter = params.shutterPhase / 360.0f;

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t idx = static_cast<size_t>(y) * w + x;
                const float* cP = curr.hostRGBA + idx * 4;
                float mC = hasMatte ? currMatte[idx] : 1.0f;

                if (params.matteMode == eMatteMaskEffect && hasMatte && mC <= 0.001f) {
                    std::memcpy(shutterOut.data() + idx * 4, cP, 4 * sizeof(float));
                    continue;
                }

                Float2 vf = flowFwd ? Float2{flowFwd->hostFlow4[idx].x, flowFwd->hostFlow4[idx].y} : Float2{0, 0};
                Float2 vb = flowBwd ? Float2{flowBwd->hostFlow4[idx].x, flowBwd->hostFlow4[idx].y} : Float2{0, 0};
                if (!flowFwd && flowBwd) vb = {-flowBwd->hostFlow4[idx].x, -flowBwd->hostFlow4[idx].y};
                if (!flowBwd && flowFwd) vb = {-vf.x, -vf.y};
                vf.x *= params.motionSensitivity; vf.y *= params.motionSensitivity;
                vb.x *= params.motionSensitivity; vb.y *= params.motionSensitivity;
                Float2 v0{0.5f * (vf.x - vb.x), 0.5f * (vf.y - vb.y)};
                Float2 a0{vf.x + vb.x, vf.y + vb.y};
                float cSpeed = std::hypot(v0.x, v0.y);
                float cGate = smoothstepf(params.motionThreshold, params.motionThreshold + 0.75f, cSpeed);
                v0.x *= cGate; v0.y *= cGate; a0.x *= cGate; a0.y *= cGate; cSpeed *= cGate;

                Float2 vDom = neighMax[std::min(tileH - 1, y / tileSize) * tileW + std::min(tileW - 1, x / tileSize)];
                float domSpeed = std::hypot(vDom.x, vDom.y);
                if (std::max(cSpeed, domSpeed) * shutterSpan < 0.25f) {
                    std::memcpy(shutterOut.data() + idx * 4, cP, 4 * sizeof(float));
                    continue;
                }

                float jRaw = static_cast<float>(((x * 73856093) ^ (y * 19349663)) & 0xFFFF) * (1.0f / 65536.0f);
                float jitter = jRaw - 0.5f;
                float fgCenter = (hasMatte && params.matteMode == eMatteForegroundIsolation) ? mC : (cSpeed > 0.35f * domSpeed ? 1.0f : 0.0f);

                float ar = 0.0f, ag = 0.0f, ab = 0.0f, aa = 0.0f, aw = 0.0f, aaw = 0.0f;
                for (int i = 0; i < params.shutterSamples; ++i) {
                    float u = (i + 0.5f + 0.35f * jitter) / params.shutterSamples;
                    float tau = phaseCenter + (u - 0.5f) * shutterSpan;
                    bool useDom = ((i & 1) == 1) && (domSpeed > cSpeed + 0.75f) && (fgCenter < 0.5f);
                    Float2 disp = useDom ? Float2{vDom.x * tau, vDom.y * tau}
                                         : Float2{v0.x * tau + 0.5f * a0.x * tau * tau, v0.y * tau + 0.5f * a0.y * tau * tau};
                    float sx = x + disp.x, sy = y + disp.y;
                    int isx = clampi(static_cast<int>(std::round(sx)), 0, w - 1);
                    int isy = clampi(static_cast<int>(std::round(sy)), 0, h - 1);
                    size_t sIdx = static_cast<size_t>(isy) * w + isx;
                    float sMatte = hasMatte ? currMatte[sIdx] : 1.0f;

                    float wSample = 1.0f;
                    if (hasMatte && params.matteMode == eMatteForegroundIsolation) {
                        if (mC >= 0.5f && sMatte < 0.5f) wSample = 0.0f;
                        else if (mC < 0.5f && sMatte >= 0.5f) {
                            Float2 svf = flowFwd ? Float2{flowFwd->hostFlow4[sIdx].x, flowFwd->hostFlow4[sIdx].y} : Float2{0, 0};
                            Float2 svb = flowBwd ? Float2{flowBwd->hostFlow4[sIdx].x, flowBwd->hostFlow4[sIdx].y} : Float2{0, 0};
                            float sSpeed = std::hypot(0.5f * (svf.x - svb.x) * params.motionSensitivity, 0.5f * (svf.y - svb.y) * params.motionSensitivity);
                            wSample = (std::hypot(disp.x, disp.y) <= sSpeed * halfSpan + 1.0f) ? 1.0f : 0.0f;
                        }
                    }
                    if (wSample <= 0.0f) continue;

                    Float4 pix = sampleRGBABilinear(curr.hostRGBA, w, h, sx, sy);
                    pix.x = dwiToLinearChannel(pix.x);
                    pix.y = dwiToLinearChannel(pix.y);
                    pix.z = dwiToLinearChannel(pix.z);
                    float luma = dwgLuminance(pix.x, pix.y, pix.z);
                    float hdrW = 1.0f + params.highlightBoost * std::pow(std::max(0.0f, luma - 0.8f), 1.5f);
                    float totW = wSample * hdrW;
                    ar += totW * pix.x; ag += totW * pix.y; ab += totW * pix.z;
                    aa += wSample * pix.w; aaw += wSample; aw += totW;
                }

                if (aw < 1e-6f) {
                    std::memcpy(shutterOut.data() + idx * 4, cP, 4 * sizeof(float));
                } else {
                    float or_ = linearToDwiChannel(ar / aw);
                    float og_ = linearToDwiChannel(ag / aw);
                    float ob_ = linearToDwiChannel(ab / aw);
                    if (params.matteMode == eMatteMaskEffect && hasMatte) {
                        or_ = cP[0] * (1.0f - mC) + or_ * mC;
                        og_ = cP[1] * (1.0f - mC) + og_ * mC;
                        ob_ = cP[2] * (1.0f - mC) + ob_ * mC;
                    }
                    shutterOut[idx * 4 + 0] = or_;
                    shutterOut[idx * 4 + 1] = og_;
                    shutterOut[idx * 4 + 2] = ob_;
                    shutterOut[idx * 4 + 3] = clampf(aa / std::max(1e-6f, aaw), 0.0f, 1.0f);
                }
            }
        }

        if (params.mode == eModeShutterBlur) {
            std::memcpy(dstRGBA, shutterOut.data(), nPix * 4 * sizeof(float));
            return true;
        }
        shutterSrc = shutterOut.data();
    }

    // Long-Exposure Trail
    std::vector<float> accum(nPix * 4, 0.0f);
    for (size_t i = 0; i < nPix; ++i) {
        const float* p = shutterSrc + i * 4;
        float r = dwiToLinearChannel(p[0]);
        float g = dwiToLinearChannel(p[1]);
        float b = dwiToLinearChannel(p[2]);
        if (params.trailBlendMode == eTrailLightPainting) {
            accum[i * 4 + 0] = r; accum[i * 4 + 1] = g; accum[i * 4 + 2] = b; accum[i * 4 + 3] = 1.0f;
        } else {
            accum[i * 4 + 0] = r * 0.02f; accum[i * 4 + 1] = g * 0.02f; accum[i * 4 + 2] = b * 0.02f; accum[i * 4 + 3] = 0.02f;
        }
    }

    const int maxLookback = std::min(centerIdx, std::max(1, params.trailFrames - 1));
    float decayLambda = 0.25f + 3.2f * params.trailDecay;

    for (int k = 0; k <= maxLookback; ++k) {
        int idxA = centerIdx - k;
        int idxB = idxA - 1;
        const float* rgbaA = (k == 0) ? shutterSrc : frames[idxA].hostRGBA;
        bool hasB = (idxB >= 0 && k < maxLookback && frames[idxB].hostRGBA != nullptr);
        const float* rgbaB = hasB ? frames[idxB].hostRGBA : rgbaA;

        std::shared_ptr<FlowEntry> pairFlow;
        if (hasB) {
            pairFlow = getOrComputeFlowPair(instanceId,
                                            frames[idxA].frameIndex, rgbaA, frames[idxA].hostMatte, frames[idxA].matteComponents,
                                            frames[idxB].frameIndex, rgbaB, frames[idxB].hostMatte, frames[idxB].matteComponents,
                                            params, cache);
        }

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t idx = static_cast<size_t>(y) * w + x;
                const float* pA = rgbaA + idx * 4;
                float rA = dwiToLinearChannel(pA[0]);
                float gA = dwiToLinearChannel(pA[1]);
                float bA = dwiToLinearChannel(pA[2]);

                Float4 fl = pairFlow ? pairFlow->hostFlow4[idx] : Float4{0, 0, 0, 0};
                float vx = fl.x * params.motionSensitivity;
                float vy = fl.y * params.motionSensitivity;
                float speed = std::hypot(vx, vy);
                float luma = dwgLuminance(rA, gA, bA);
                float lumaDwi = linearToDwiChannel(luma);
                float mGate = params.motionThreshold > 0.0f ? smoothstepf(params.motionThreshold, params.motionThreshold + 0.75f, speed) : 1.0f;
                float lGate = params.lumaThreshold > 0.0f ? smoothstepf(params.lumaThreshold, params.lumaThreshold + 0.12f, lumaDwi) : 1.0f;
                float hdrW = 1.0f + params.highlightBoost * std::pow(std::max(0.0f, luma - 0.8f), 1.5f);

                if (params.subFrameSteps <= 1 || !hasB || speed < 0.35f) {
                    float timeW = (params.trailBlendMode == eTrailCometDecay) ? std::exp(-decayLambda * k) : 1.0f;
                    if (params.trailBlendMode == eTrailLightPainting) {
                        float gTot = mGate * lGate * timeW;
                        if (gTot > 0.01f) {
                            accum[idx * 4 + 0] = std::max(accum[idx * 4 + 0], rA * gTot);
                            accum[idx * 4 + 1] = std::max(accum[idx * 4 + 1], gA * gTot);
                            accum[idx * 4 + 2] = std::max(accum[idx * 4 + 2], bA * gTot);
                        }
                    } else {
                        float wTot = timeW * hdrW;
                        accum[idx * 4 + 0] += rA * wTot;
                        accum[idx * 4 + 1] += gA * wTot;
                        accum[idx * 4 + 2] += bA * wTot;
                        accum[idx * 4 + 3] += wTot;
                    }
                    continue;
                }

                int steps = std::max(params.subFrameSteps, std::min(72, static_cast<int>(std::ceil(speed * 1.25f))));
                float invSteps = 1.0f / steps;
                Float4 pB = sampleRGBABilinear(rgbaB, w, h, x + fl.x, y + fl.y);
                pB.x = dwiToLinearChannel(pB.x);
                pB.y = dwiToLinearChannel(pB.y);
                pB.z = dwiToLinearChannel(pB.z);
                float conf = pairFlow ? pairFlow->hostConf2[idx].x : 1.0f;

                for (int s = 0; s < steps; ++s) {
                    float alpha = s * invSteps;
                    float px = x + alpha * vx;
                    float py = y + alpha * vy;
                    float bb = alpha * conf;
                    float sr = rA * (1.0f - bb) + pB.x * bb;
                    float sg = gA * (1.0f - bb) + pB.y * bb;
                    float sb = bA * (1.0f - bb) + pB.z * bb;
                    float timeW = (params.trailBlendMode == eTrailCometDecay) ? std::exp(-decayLambda * (k + alpha)) : 1.0f;

                    int x0 = static_cast<int>(std::floor(px));
                    int y0 = static_cast<int>(std::floor(py));
                    float fx = px - x0, fy = py - y0;
                    float bw[4] = {(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy};
                    int ox[4] = {x0, x0 + 1, x0, x0 + 1};
                    int oy[4] = {y0, y0, y0 + 1, y0 + 1};

                    for (int c = 0; c < 4; ++c) {
                        if (ox[c] >= 0 && ox[c] < w && oy[c] >= 0 && oy[c] < h) {
                            size_t tidx = static_cast<size_t>(oy[c]) * w + ox[c];
                            if (params.trailBlendMode == eTrailLightPainting) {
                                float gate = mGate * lGate * timeW;
                                if (gate > 0.01f && bw[c] > 0.01f) {
                                    float sgGain = std::min(1.0f, bw[c] * 1.35f) * gate;
                                    accum[tidx * 4 + 0] = std::max(accum[tidx * 4 + 0], sr * sgGain);
                                    accum[tidx * 4 + 1] = std::max(accum[tidx * 4 + 1], sg * sgGain);
                                    accum[tidx * 4 + 2] = std::max(accum[tidx * 4 + 2], sb * sgGain);
                                }
                            } else {
                                float wt = timeW * hdrW * invSteps * bw[c];
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
    }

    for (size_t i = 0; i < nPix; ++i) {
        const float* cur = shutterSrc + i * 4;
        float cr = dwiToLinearChannel(cur[0]);
        float cg = dwiToLinearChannel(cur[1]);
        float cb = dwiToLinearChannel(cur[2]);

        float or_, og_, ob_;
        if (params.trailBlendMode == eTrailLightPainting) {
            or_ = std::max(cr, accum[i * 4 + 0]);
            og_ = std::max(cg, accum[i * 4 + 1]);
            ob_ = std::max(cb, accum[i * 4 + 2]);
        } else {
            float wSum = std::max(1e-5f, accum[i * 4 + 3]);
            or_ = accum[i * 4 + 0] / wSum;
            og_ = accum[i * 4 + 1] / wSum;
            ob_ = accum[i * 4 + 2] / wSum;
            if (params.trailBlendMode == eTrailCometDecay && params.rearCurtainCrispness > 0.0f) {
                float crisp = clampf(params.rearCurtainCrispness, 0.0f, 1.0f);
                or_ = or_ * (1.0f - crisp) + cr * crisp;
                og_ = og_ * (1.0f - crisp) + cg * crisp;
                ob_ = ob_ * (1.0f - crisp) + cb * crisp;
            }
        }

        or_ = linearToDwiChannel(or_);
        og_ = linearToDwiChannel(og_);
        ob_ = linearToDwiChannel(ob_);
        if (hasMatte && params.matteMode == eMatteMaskEffect) {
            float m = currMatte[i];
            or_ = cur[0] * (1.0f - m) + or_ * m;
            og_ = cur[1] * (1.0f - m) + og_ * m;
            ob_ = cur[2] * (1.0f - m) + ob_ * m;
        }
        dstRGBA[i * 4 + 0] = or_; dstRGBA[i * 4 + 1] = og_; dstRGBA[i * 4 + 2] = ob_; dstRGBA[i * 4 + 3] = cur[3];
    }

    return true;
}

} // namespace bmb
