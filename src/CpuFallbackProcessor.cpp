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
    const int kMaxLevels = 7;
    int lw[kMaxLevels], lh[kMaxLevels];
    std::vector<float> lumaA[kMaxLevels], lumaB[kMaxLevels], mattePyrA[kMaxLevels];

    lw[0] = width;
    lh[0] = height;
    extractLumaAndMatteCPU(width, height, rgbaA, matteA, matteAComp, lumaA[0], mattePyrA[0]);

    std::vector<float> dummyMatteB;
    extractLumaAndMatteCPU(width, height, rgbaB, matteB, matteBComp, lumaB[0], dummyMatteB);

    int numLevels = 1;
    for (int i = 1; i < kMaxLevels; ++i) {
        int nw = lw[i - 1] / 2;
        int nh = lh[i - 1] / 2;
        if (nw < 16 || nh < 16) break;
        lw[i] = nw;
        lh[i] = nh;
        ++numLevels;
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
    float sigmaRange = 0.012f + (1.0f - flowDetail) * 0.055f;

    std::vector<Float2> flow;
    for (int lvl = numLevels - 1; lvl >= 0; --lvl) {
        int w = lw[lvl], h = lh[lvl];
        std::vector<Float2> inFlow(static_cast<size_t>(w) * h, {0.0f, 0.0f});
        if (lvl < numLevels - 1) {
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
        int searchRadius = 0;
        if (lvl == numLevels - 1) searchRadius = (numLevels == 5) ? 3 : 5;
        else if (lvl >= 2) searchRadius = 1;
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

                float regW = (searchRadius >= 2) ? 8.0e-5f : 0.0f;
                if (searchRadius > 0) {
                    Float2 base = best;
                    float bestScore = bestCost + regW * (base.x * base.x + base.y * base.y);
                    for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
                        for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
                            if (dx == 0 && dy == 0) continue;
                            Float2 cand = {base.x + dx, base.y + dy};
                            float c = evalCost(cand.x, cand.y);
                            float score = c + regW * (cand.x * cand.x + cand.y * cand.y);
                            if (score + 1e-5f < bestScore) {
                                bestScore = score;
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
        static_cast<int>(params.matteMode),
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

bool CpuFallbackProcessor::areFramesDistinct(
    const float* rgbaA, const float* rgbaB, int width, int height)
{
    if (!rgbaA || !rgbaB || width <= 0 || height <= 0) return true;
    if (rgbaA == rgbaB) return false;
    for (int gy = 0; gy < 16; ++gy) {
        int sy = clampi((gy * height) / 16 + height / 32, 0, height - 1);
        for (int gx = 0; gx < 16; ++gx) {
            int sx = clampi((gx * width) / 16 + width / 32, 0, width - 1);
            size_t idx = (static_cast<size_t>(sy) * width + sx) * 4;
            float d = std::fabs(rgbaA[idx + 0] - rgbaB[idx + 0]) +
                      std::fabs(rgbaA[idx + 1] - rgbaB[idx + 1]) +
                      std::fabs(rgbaA[idx + 2] - rgbaB[idx + 2]);
            if (d > 3e-4f) return true;
        }
    }
    return false;
}

namespace {

inline void evalConfWeightedVelAccCPU(
    size_t idx,
    const FlowEntry* flowFwd, float dtInvFwd,
    const FlowEntry* flowBwd, float dtInvBwd,
    float sensitivity,
    Float2& outV0, Float2& outA0)
{
    Float2 vf = flowFwd ? Float2{flowFwd->hostFlow4[idx].x, flowFwd->hostFlow4[idx].y} : Float2{0.0f, 0.0f};
    Float2 vb = flowBwd ? Float2{flowBwd->hostFlow4[idx].x, flowBwd->hostFlow4[idx].y} : Float2{0.0f, 0.0f};
    vf.x *= dtInvFwd * sensitivity; vf.y *= dtInvFwd * sensitivity;
    vb.x *= dtInvBwd * sensitivity; vb.y *= dtInvBwd * sensitivity;

    float cf = flowFwd ? flowFwd->hostConf2[idx].x : 0.0f;
    float cb = flowBwd ? flowBwd->hostConf2[idx].x : 0.0f;
    float wf = flowFwd ? (cf * cf + 0.02f) : 0.0f;
    float wb = flowBwd ? (cb * cb + 0.02f) : 0.0f;
    float wSum = std::max(1e-5f, wf + wb);

    outV0 = {(wf * vf.x - wb * vb.x) / wSum, (wf * vf.y - wb * vb.y) / wSum};
    float accGate = (flowFwd && flowBwd) ? (std::min(cf, cb) * std::min(cf, cb)) : 0.0f;
    float dtF = (dtInvFwd > 1e-4f) ? (1.0f / dtInvFwd) : 1.0f;
    float dtB = (dtInvBwd > 1e-4f) ? (1.0f / dtInvBwd) : 1.0f;
    float accScale = (2.0f / std::max(1e-4f, dtF + dtB)) * accGate;
    outA0 = {(vf.x + vb.x) * accScale, (vf.y + vb.y) * accScale};
}

void renderSingleShutterPassCPU(
    const void* instanceId,
    const FrameInput* prevF,
    const FrameInput& currF,
    const FrameInput* nextF,
    float* outRGBA,
    const RenderParams& params,
    FlowCache& cache)
{
    const int w = params.width;
    const int h = params.height;
    const size_t nPix = static_cast<size_t>(w) * h;

    std::vector<float> dummyLuma, currMatte, prevMatte, nextMatte;
    extractLumaAndMatteCPU(w, h, currF.hostRGBA, currF.hostMatte, currF.matteComponents, dummyLuma, currMatte);
    bool hasMatte = (currF.hostMatte != nullptr && currF.matteComponents > 0 && params.matteMode != eMatteIgnore);
    bool hasPrevMatte = (hasMatte && prevF && prevF->hostMatte && prevF->matteComponents > 0);
    bool hasNextMatte = (hasMatte && nextF && nextF->hostMatte && nextF->matteComponents > 0);
    if (hasPrevMatte) extractLumaAndMatteCPU(w, h, prevF->hostRGBA, prevF->hostMatte, prevF->matteComponents, dummyLuma, prevMatte);
    if (hasNextMatte) extractLumaAndMatteCPU(w, h, nextF->hostRGBA, nextF->hostMatte, nextF->matteComponents, dummyLuma, nextMatte);

    std::shared_ptr<FlowEntry> flowFwd, flowBwd;
    if (nextF && nextF->hostRGBA) {
        flowFwd = CpuFallbackProcessor::getOrComputeFlowPair(
            instanceId,
            currF.frameIndex, currF.hostRGBA, currF.hostMatte, currF.matteComponents,
            nextF->frameIndex, nextF->hostRGBA, nextF->hostMatte, nextF->matteComponents,
            params, cache);
    }
    if (prevF && prevF->hostRGBA) {
        flowBwd = CpuFallbackProcessor::getOrComputeFlowPair(
            instanceId,
            currF.frameIndex, currF.hostRGBA, currF.hostMatte, currF.matteComponents,
            prevF->frameIndex, prevF->hostRGBA, prevF->hostMatte, prevF->matteComponents,
            params, cache);
    }

    float dtInvFwd = (params.fwdFrameStep > 1e-4f) ? (1.0f / params.fwdFrameStep) : 1.0f;
    float dtInvBwd = (params.bwdFrameStep > 1e-4f) ? (1.0f / params.bwdFrameStep) : 1.0f;

    const int tileSize = 16;
    const int tileW = (w + tileSize - 1) / tileSize;
    const int tileH = (h + tileSize - 1) / tileSize;
    std::vector<Float4> tileMax(static_cast<size_t>(tileW) * tileH, {0.0f, 0.0f, 0.0f, 0.0f});
    std::vector<Float4> tmpMax(static_cast<size_t>(tileW) * tileH, {0.0f, 0.0f, 0.0f, 0.0f});
    std::vector<Float4> neighMax(static_cast<size_t>(tileW) * tileH, {0.0f, 0.0f, 0.0f, 0.0f});

    for (int ty = 0; ty < tileH; ++ty) {
        for (int tx = 0; tx < tileW; ++tx) {
            float bestM = 0.0f;
            Float4 bestV{0.0f, 0.0f, 0.0f, 0.0f};
            for (int y = ty * tileSize; y < std::min(h, (ty + 1) * tileSize); ++y) {
                for (int x = tx * tileSize; x < std::min(w, (tx + 1) * tileSize); ++x) {
                    size_t idx = static_cast<size_t>(y) * w + x;
                    Float2 v0, a0;
                    evalConfWeightedVelAccCPU(idx, flowFwd.get(), dtInvFwd, flowBwd.get(), dtInvBwd,
                                              params.motionSensitivity, v0, a0);
                    float m = std::hypot(v0.x, v0.y);
                    float g = smoothstepf(params.motionThreshold, params.motionThreshold + 0.75f, m);
                    if (m * g > bestM) {
                        bestM = m * g;
                        bestV = {v0.x * g, v0.y * g, a0.x * g, a0.y * g};
                    }
                }
            }
            tileMax[ty * tileW + tx] = bestV;
        }
    }

    float shutterSpan = (params.shutterAngle / 360.0f) * params.timebaseScale;
    float phaseCenter = (params.shutterPhase / 360.0f) * params.timebaseScale;
    float maxHalfSpan = 0.5f * shutterSpan + std::fabs(phaseCenter);
    int tileRadius = std::clamp(static_cast<int>(std::ceil(0.14f * static_cast<float>(w) * maxHalfSpan / static_cast<float>(tileSize))), 1, 12);

    for (int ty = 0; ty < tileH; ++ty) {
        for (int tx = 0; tx < tileW; ++tx) {
            float bestM2 = 0.0f;
            Float4 bestV{0.0f, 0.0f, 0.0f, 0.0f};
            for (int nx = std::max(0, tx - tileRadius); nx <= std::min(tileW - 1, tx + tileRadius); ++nx) {
                Float4 v = tileMax[ty * tileW + nx];
                float m2 = v.x * v.x + v.y * v.y;
                if (m2 > bestM2) { bestM2 = m2; bestV = v; }
            }
            tmpMax[ty * tileW + tx] = bestV;
        }
    }
    for (int ty = 0; ty < tileH; ++ty) {
        for (int tx = 0; tx < tileW; ++tx) {
            float bestM2 = 0.0f;
            Float4 bestV{0.0f, 0.0f, 0.0f, 0.0f};
            for (int ny = std::max(0, ty - tileRadius); ny <= std::min(tileH - 1, ty + tileRadius); ++ny) {
                Float4 v = tmpMax[ny * tileW + tx];
                float m2 = v.x * v.x + v.y * v.y;
                if (m2 > bestM2) { bestM2 = m2; bestV = v; }
            }
            neighMax[ty * tileW + tx] = bestV;
        }
    }

    const float tauMin = phaseCenter - 0.5f * shutterSpan;
    const float tauMax = phaseCenter + 0.5f * shutterSpan;
    const bool matteIso = (hasMatte && params.matteMode == eMatteForegroundIsolation);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            size_t idx = static_cast<size_t>(y) * w + x;
            const float* cP = currF.hostRGBA + idx * 4;
            float mC = hasMatte ? currMatte[idx] : 1.0f;

            if (params.matteMode == eMatteMaskEffect && hasMatte && mC <= 0.001f) {
                std::memcpy(outRGBA + idx * 4, cP, 4 * sizeof(float));
                continue;
            }

            Float2 v0, a0;
            evalConfWeightedVelAccCPU(idx, flowFwd.get(), dtInvFwd, flowBwd.get(), dtInvBwd,
                                      params.motionSensitivity, v0, a0);
            float cSpeed = std::hypot(v0.x, v0.y);
            float cGate = smoothstepf(params.motionThreshold, params.motionThreshold + 0.75f, cSpeed);
            v0.x *= cGate; v0.y *= cGate; a0.x *= cGate; a0.y *= cGate; cSpeed *= cGate;

            Float4 domVA = neighMax[std::min(tileH - 1, y / tileSize) * tileW + std::min(tileW - 1, x / tileSize)];
            Float2 vDom{domVA.x, domVA.y};
            Float2 aDom{domVA.z, domVA.w};
            float domSpeed = std::hypot(vDom.x, vDom.y);
            if (std::max(cSpeed, domSpeed) * shutterSpan < 0.25f) {
                std::memcpy(outRGBA + idx * 4, cP, 4 * sizeof(float));
                continue;
            }

            float centerLinR = dwiToLinearChannel(cP[0]);
            float centerLinG = dwiToLinearChannel(cP[1]);
            float centerLinB = dwiToLinearChannel(cP[2]);
            float centerHdrW = 1.0f;
            if (params.highlightBoost > 0.0f) {
                float cl = dwgLuminance(centerLinR, centerLinG, centerLinB);
                if (cl > 0.8f) {
                    float ex = cl - 0.8f;
                    centerHdrW += params.highlightBoost * ex * std::sqrt(ex);
                }
            }

            Float2 vAtNext = flowFwd ? Float2{flowFwd->hostFlow4[idx].z * dtInvFwd * params.motionSensitivity,
                                              flowFwd->hostFlow4[idx].w * dtInvFwd * params.motionSensitivity}
                                     : Float2{0.0f, 0.0f};
            float nextSpeedAtX = std::hypot(vAtNext.x, vAtNext.y);

            Float2 vAtPrev = flowBwd ? Float2{flowBwd->hostFlow4[idx].z * dtInvBwd * params.motionSensitivity,
                                              flowBwd->hostFlow4[idx].w * dtInvBwd * params.motionSensitivity}
                                     : Float2{0.0f, 0.0f};
            float prevSpeedAtX = std::hypot(vAtPrev.x, vAtPrev.y);

            float mPrevAtX = hasPrevMatte ? prevMatte[idx] : 0.0f;
            float mNextAtX = hasNextMatte ? nextMatte[idx] : 0.0f;

            bool centerIsBg = matteIso ? (mC < 0.5f) : (cSpeed < 0.35f * domSpeed);
            bool useDomAll = (domSpeed > cSpeed + 0.75f) && centerIsBg;
            bool boundaryMix = !matteIso && !useDomAll && (domSpeed > cSpeed + 1.5f) && (cSpeed < 0.75f * domSpeed);

            float jRaw = static_cast<float>(((x * 73856093) ^ (y * 19349663)) & 0xFFFF) * (1.0f / 65536.0f);
            float jitter = jRaw - 0.5f;

            int effSamples = params.shutterSamples;
            if (params.autoShutterSamples) {
                int minS = std::min(params.shutterSamples, 8);
                int adaptive = static_cast<int>(std::ceil(std::max(cSpeed, domSpeed) * shutterSpan * 1.25f));
                effSamples = clampi(adaptive, minS, params.shutterSamples);
            }
            const float invSamples = 1.0f / static_cast<float>(effSamples);

            float fgR = 0.0f, fgG = 0.0f, fgB = 0.0f, fgA = 0.0f, fgRgbW = 0.0f, fgTime = 0.0f;
            float bgR = 0.0f, bgG = 0.0f, bgB = 0.0f, bgA = 0.0f, bgRgbW = 0.0f, bgTime = 0.0f;

            for (int i = 0; i < effSamples; ++i) {
                float u = (i + 0.5f + 0.35f * jitter) * invSamples;
                float tau = phaseCenter + (u - 0.5f) * shutterSpan;
                bool tapUseDom = useDomAll || (boundaryMix && ((i & 1) == 1));

                float halfTau2 = 0.5f * tau * tau;
                Float2 disp = tapUseDom
                    ? Float2{-vDom.x * tau - aDom.x * halfTau2, -vDom.y * tau - aDom.y * halfTau2}
                    : Float2{-v0.x * tau - a0.x * halfTau2, -v0.y * tau - a0.y * halfTau2};
                float sx = x + disp.x, sy = y + disp.y;
                int isx = clampi(static_cast<int>(std::round(sx)), 0, w - 1);
                int isy = clampi(static_cast<int>(std::round(sy)), 0, h - 1);
                size_t sIdx = static_cast<size_t>(isy) * w + isx;

                if (tapUseDom) {
                    Float2 sVel, sAcc;
                    evalConfWeightedVelAccCPU(sIdx, flowFwd.get(), dtInvFwd, flowBwd.get(), dtInvBwd,
                                              params.motionSensitivity, sVel, sAcc);
                    float sSpeed = std::hypot(sVel.x, sVel.y);
                    bool candFg = matteIso
                        ? (currMatte[sIdx] >= 0.5f && sSpeed > 0.25f)
                        : (sSpeed > cSpeed + 0.75f && sSpeed > 0.25f * domSpeed);
                    bool fgHit = false;
                    if (candFg) {
                        float dx = static_cast<float>(x) - sx;
                        float dy = static_cast<float>(y) - sy;
                        float v2 = std::max(1e-4f, sVel.x * sVel.x + sVel.y * sVel.y);
                        float tauStar = (dx * sVel.x + dy * sVel.y) / v2;
                        float epsTau = 0.75f / std::max(1.0f, sSpeed);
                        if (tauStar >= tauMin - epsTau && tauStar <= tauMax + epsTau) {
                            float halfTauS2 = 0.5f * tauStar * tauStar;
                            float perpDist = std::hypot(dx - tauStar * sVel.x - sAcc.x * halfTauS2,
                                                        dy - tauStar * sVel.y - sAcc.y * halfTauS2);
                            float distToSample = std::hypot(dx, dy);
                            if (perpDist <= 1.5f + 0.15f * distToSample) {
                                fgHit = true;
                            }
                        }
                    }

                    if (fgHit) {
                        Float4 pix = sampleRGBABilinear(currF.hostRGBA, w, h, sx, sy);
                        pix.x = dwiToLinearChannel(pix.x);
                        pix.y = dwiToLinearChannel(pix.y);
                        pix.z = dwiToLinearChannel(pix.z);
                        float hdrW = 1.0f;
                        if (params.highlightBoost > 0.0f) {
                            float luma = dwgLuminance(pix.x, pix.y, pix.z);
                            if (luma > 0.8f) {
                                float ex = luma - 0.8f;
                                hdrW += params.highlightBoost * ex * std::sqrt(ex);
                            }
                        }
                        float wJac = clampf(domSpeed / std::max(sSpeed, 1.0f), 1.0f, 2.5f);
                        float totW = wJac * hdrW;
                        fgR += totW * pix.x; fgG += totW * pix.y; fgB += totW * pix.z;
                        fgA += wJac * pix.w; fgRgbW += totW; fgTime += wJac;
                    } else {
                        if (cSpeed > 0.5f) {
                            Float4 pix = sampleRGBABilinear(currF.hostRGBA, w, h, x - v0.x * tau, y - v0.y * tau);
                            pix.x = dwiToLinearChannel(pix.x);
                            pix.y = dwiToLinearChannel(pix.y);
                            pix.z = dwiToLinearChannel(pix.z);
                            float hdrW = 1.0f;
                            if (params.highlightBoost > 0.0f) {
                                float luma = dwgLuminance(pix.x, pix.y, pix.z);
                                if (luma > 0.8f) {
                                    float ex = luma - 0.8f;
                                    hdrW += params.highlightBoost * ex * std::sqrt(ex);
                                }
                            }
                            bgR += hdrW * pix.x; bgG += hdrW * pix.y; bgB += hdrW * pix.z;
                            bgA += pix.w; bgRgbW += hdrW;
                        } else {
                            bgR += centerHdrW * centerLinR; bgG += centerHdrW * centerLinG; bgB += centerHdrW * centerLinB;
                            bgA += cP[3]; bgRgbW += centerHdrW;
                        }
                        bgTime += 1.0f;
                    }
                } else {
                    Float4 pix{};
                    float wSample = 1.0f;
                    bool steppedOff = false;
                    if (matteIso) {
                        steppedOff = (mC >= 0.5f && currMatte[sIdx] < 0.5f);
                    } else if (cSpeed > 1.5f) {
                        Float2 sVel, sAcc;
                        evalConfWeightedVelAccCPU(sIdx, flowFwd.get(), dtInvFwd, flowBwd.get(), dtInvBwd,
                                                  params.motionSensitivity, sVel, sAcc);
                        float sSpeed = std::hypot(sVel.x, sVel.y);
                        steppedOff = (sSpeed < 0.35f * cSpeed);
                    }

                    if (steppedOff) {
                        bool usePrev = (tau < 0.0f);
                        if (usePrev) {
                            bool uncovered = matteIso ? (hasPrevMatte ? (mPrevAtX < 0.5f) : true)
                                                      : (flowBwd ? (prevSpeedAtX < 0.5f * cSpeed) : true);
                            if (uncovered && prevF && prevF->hostRGBA) {
                                float dtStep = (dtInvBwd > 1e-4f) ? (1.0f / dtInvBwd) : 1.0f;
                                float pSub = clampf(dtStep + tau, 0.0f, dtStep);
                                pix = sampleRGBABilinear(prevF->hostRGBA, w, h, x - vAtPrev.x * pSub, y - vAtPrev.y * pSub);
                            } else if (uncovered) {
                                pix = sampleRGBABilinear(currF.hostRGBA, w, h, sx, sy);
                            } else {
                                if (matteIso) wSample = 0.0f;
                                else {
                                    wSample = 0.05f;
                                    pix = sampleRGBABilinear(currF.hostRGBA, w, h, sx, sy);
                                }
                            }
                        } else {
                            bool uncovered = matteIso ? (hasNextMatte ? (mNextAtX < 0.5f) : true)
                                                      : (flowFwd ? (nextSpeedAtX < 0.5f * cSpeed) : true);
                            if (uncovered && nextF && nextF->hostRGBA) {
                                float dtStep = (dtInvFwd > 1e-4f) ? (1.0f / dtInvFwd) : 1.0f;
                                float nSub = clampf(dtStep - tau, 0.0f, dtStep);
                                pix = sampleRGBABilinear(nextF->hostRGBA, w, h, x - vAtNext.x * nSub, y - vAtNext.y * nSub);
                            } else if (uncovered) {
                                pix = sampleRGBABilinear(currF.hostRGBA, w, h, sx, sy);
                            } else {
                                if (matteIso) wSample = 0.0f;
                                else {
                                    wSample = 0.05f;
                                    pix = sampleRGBABilinear(currF.hostRGBA, w, h, sx, sy);
                                }
                            }
                        }
                    } else {
                        pix = sampleRGBABilinear(currF.hostRGBA, w, h, sx, sy);
                    }

                    if (wSample <= 0.0f) continue;
                    pix.x = dwiToLinearChannel(pix.x);
                    pix.y = dwiToLinearChannel(pix.y);
                    pix.z = dwiToLinearChannel(pix.z);
                    float hdrW = 1.0f;
                    if (params.highlightBoost > 0.0f) {
                        float luma = dwgLuminance(pix.x, pix.y, pix.z);
                        if (luma > 0.8f) {
                            float ex = luma - 0.8f;
                            hdrW += params.highlightBoost * ex * std::sqrt(ex);
                        }
                    }
                    float totW = wSample * hdrW;
                    fgR += totW * pix.x; fgG += totW * pix.y; fgB += totW * pix.z;
                    fgA += wSample * pix.w; fgRgbW += totW; fgTime += wSample;
                }
            }

            float orLin = 0.0f, ogLin = 0.0f, obLin = 0.0f, outA = cP[3];
            if (fgTime < 1e-6f && bgTime < 1e-6f) {
                std::memcpy(outRGBA + idx * 4, cP, 4 * sizeof(float));
                continue;
            } else if (fgTime > 1e-6f && bgTime > 1e-6f) {
                float fgFrac = clampf(fgTime * invSamples, 0.0f, 1.0f);
                float bgFrac = 1.0f - fgFrac;
                float invFgW = 1.0f / std::max(1e-6f, fgRgbW);
                float invBgW = 1.0f / std::max(1e-6f, bgRgbW);
                float wFgBlend = fgFrac * (fgRgbW / std::max(1e-6f, fgTime));
                float wBgBlend = bgFrac * (bgRgbW / std::max(1e-6f, bgTime));
                float invBlend = 1.0f / std::max(1e-6f, wFgBlend + wBgBlend);
                orLin = (wFgBlend * (fgR * invFgW) + wBgBlend * (bgR * invBgW)) * invBlend;
                ogLin = (wFgBlend * (fgG * invFgW) + wBgBlend * (bgG * invBgW)) * invBlend;
                obLin = (wFgBlend * (fgB * invFgW) + wBgBlend * (bgB * invBgW)) * invBlend;
                outA = clampf(fgFrac * (fgA / fgTime) + bgFrac * (bgA / bgTime), 0.0f, 1.0f);
            } else if (fgTime > 1e-6f) {
                orLin = fgR / fgRgbW; ogLin = fgG / fgRgbW; obLin = fgB / fgRgbW;
                outA = clampf(fgA / fgTime, 0.0f, 1.0f);
            } else {
                orLin = bgR / bgRgbW; ogLin = bgG / bgRgbW; obLin = bgB / bgRgbW;
                outA = clampf(bgA / bgTime, 0.0f, 1.0f);
            }

            float or_ = linearToDwiChannel(orLin);
            float og_ = linearToDwiChannel(ogLin);
            float ob_ = linearToDwiChannel(obLin);
            if (params.matteMode == eMatteMaskEffect && hasMatte) {
                or_ = cP[0] * (1.0f - mC) + or_ * mC;
                og_ = cP[1] * (1.0f - mC) + og_ * mC;
                ob_ = cP[2] * (1.0f - mC) + ob_ * mC;
            }
            outRGBA[idx * 4 + 0] = or_;
            outRGBA[idx * 4 + 1] = og_;
            outRGBA[idx * 4 + 2] = ob_;
            outRGBA[idx * 4 + 3] = outA;
        }
    }
}

} // namespace

bool CpuFallbackProcessor::renderFrame(
    const void* instanceId,
    const std::vector<FrameInput>& frames,
    int centerIdx,
    float* dstRGBA,
    const RenderParams& params,
    FlowCache& cache,
    const std::vector<FrameInput>* shutterSubFrames)
{
    if (!dstRGBA || centerIdx < 0 || centerIdx >= static_cast<int>(frames.size())) return false;
    const int w = params.width;
    const int h = params.height;
    const size_t nPix = static_cast<size_t>(w) * h;
    const FrameInput& curr = frames[centerIdx];
    if (!curr.hostRGBA) return false;

    const bool useSubFrames = (params.viewMode == eViewFinal) &&
                              (params.mode == eModeShutterBlur || params.mode == eModeCombined) &&
                              (shutterSubFrames != nullptr) &&
                              (shutterSubFrames->size() >= 3);

    if (params.viewMode != eViewFinal) {
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
        if (useSubFrames) {
            const auto& sub = *shutterSubFrames;
            const int M = static_cast<int>(sub.size());
            const float effSpan = (params.shutterAngle / 360.0f) * params.timebaseScale;
            const float effPhase = (params.shutterPhase / 360.0f) * params.timebaseScale;
            const float tauMin = effPhase - 0.5f * effSpan;
            const float tauMax = effPhase + 0.5f * effSpan;

            std::vector<float> accumLin(nPix * 4, 0.0f);
            std::vector<float> segOut(nPix * 4, 0.0f);
            float totalSegDur = 0.0f;

            for (int j = 0; j < M; ++j) {
                float vLo = (j > 0) ? 0.5f * (sub[j - 1].relTime + sub[j].relTime) : tauMin;
                float vHi = (j + 1 < M) ? 0.5f * (sub[j].relTime + sub[j + 1].relTime) : tauMax;
                float segLo = std::max(tauMin, vLo);
                float segHi = std::min(tauMax, vHi);
                float segDur = segHi - segLo;
                if (segDur <= 1e-4f || !sub[j].hostRGBA) continue;

                RenderParams segP = params;
                segP.timebaseScale = 1.0f;
                segP.shutterAngle = segDur * 360.0f;
                segP.shutterPhase = (0.5f * (segLo + segHi) - sub[j].relTime) * 360.0f;
                segP.bwdFrameStep = (j > 0) ? std::max(1e-3f, sub[j].relTime - sub[j - 1].relTime) : 1.0f;
                segP.fwdFrameStep = (j + 1 < M) ? std::max(1e-3f, sub[j + 1].relTime - sub[j].relTime) : 1.0f;

                const FrameInput* pPrev = (j > 0) ? &sub[j - 1] : nullptr;
                const FrameInput* pNext = (j + 1 < M) ? &sub[j + 1] : nullptr;
                renderSingleShutterPassCPU(instanceId, pPrev, sub[j], pNext, segOut.data(), segP, cache);

                for (size_t i = 0; i < nPix; ++i) {
                    accumLin[i * 4 + 0] += dwiToLinearChannel(segOut[i * 4 + 0]) * segDur;
                    accumLin[i * 4 + 1] += dwiToLinearChannel(segOut[i * 4 + 1]) * segDur;
                    accumLin[i * 4 + 2] += dwiToLinearChannel(segOut[i * 4 + 2]) * segDur;
                    accumLin[i * 4 + 3] += segOut[i * 4 + 3] * segDur;
                }
                totalSegDur += segDur;
            }
            if (totalSegDur > 1e-6f) {
                float invW = 1.0f / totalSegDur;
                for (size_t i = 0; i < nPix; ++i) {
                    shutterOut[i * 4 + 0] = linearToDwiChannel(accumLin[i * 4 + 0] * invW);
                    shutterOut[i * 4 + 1] = linearToDwiChannel(accumLin[i * 4 + 1] * invW);
                    shutterOut[i * 4 + 2] = linearToDwiChannel(accumLin[i * 4 + 2] * invW);
                    shutterOut[i * 4 + 3] = clampf(accumLin[i * 4 + 3] * invW, 0.0f, 1.0f);
                }
            } else {
                std::memcpy(shutterOut.data(), curr.hostRGBA, nPix * 4 * sizeof(float));
            }
        } else {
            const FrameInput* prevF = (centerIdx - 1 >= 0) ? &frames[centerIdx - 1] : nullptr;
            const FrameInput* nextF = (centerIdx + 1 < static_cast<int>(frames.size())) ? &frames[centerIdx + 1] : nullptr;
            renderSingleShutterPassCPU(instanceId, prevF, curr, nextF, shutterOut.data(), params, cache);
        }

        if (params.mode == eModeShutterBlur) {
            std::memcpy(dstRGBA, shutterOut.data(), nPix * 4 * sizeof(float));
            return true;
        }
        shutterSrc = shutterOut.data();
    }

    // Long-Exposure Trail
    std::vector<float> dummyLuma, currMatte;
    extractLumaAndMatteCPU(w, h, curr.hostRGBA, curr.hostMatte, curr.matteComponents, dummyLuma, currMatte);
    bool hasMatte = (curr.hostMatte != nullptr && curr.matteComponents > 0 && params.matteMode != eMatteIgnore);
    const bool fgIso = (hasMatte && params.matteMode == eMatteForegroundIsolation);
    const int maxLookback = std::min(centerIdx, std::max(1, params.trailFrames - 1));
    float decayLambda = 0.25f + 3.2f * params.trailDecay;
    float pairStep = (params.pairTimeStep > 1e-4f) ? params.pairTimeStep : 1.0f;

    std::vector<float> accum(nPix * 4, 0.0f);
    for (size_t i = 0; i < nPix; ++i) {
        const float* p = shutterSrc + i * 4;
        float r = dwiToLinearChannel(p[0]);
        float g = dwiToLinearChannel(p[1]);
        float b = dwiToLinearChannel(p[2]);
        if (params.trailBlendMode == eTrailLightPainting) {
            accum[i * 4 + 0] = r; accum[i * 4 + 1] = g; accum[i * 4 + 2] = b; accum[i * 4 + 3] = 1.0f;
        } else if (!fgIso) {
            accum[i * 4 + 0] = r * 0.02f; accum[i * 4 + 1] = g * 0.02f; accum[i * 4 + 2] = b * 0.02f; accum[i * 4 + 3] = 0.02f;
        }
    }

    std::vector<std::shared_ptr<FlowEntry>> pairFlows(maxLookback + 1);
    for (int k = maxLookback - 1; k >= 0; --k) {
        int idxA = centerIdx - k;
        int idxB = idxA - 1;
        const float* rgbaA = (k == 0) ? shutterSrc : frames[idxA].hostRGBA;
        if (idxB >= 0 && frames[idxB].hostRGBA) {
            pairFlows[k] = getOrComputeFlowPair(instanceId,
                                                frames[idxA].frameIndex, rgbaA, frames[idxA].hostMatte, frames[idxA].matteComponents,
                                                frames[idxB].frameIndex, frames[idxB].hostRGBA, frames[idxB].hostMatte, frames[idxB].matteComponents,
                                                params, cache);
        }
    }

    std::vector<float> lastMatte;
    const FrameInput& fLast = frames[centerIdx - maxLookback];
    const float* lastRGBA = (maxLookback == 0) ? shutterSrc : fLast.hostRGBA;
    if (hasMatte && fLast.hostMatte && fLast.matteComponents > 0) {
        extractLumaAndMatteCPU(w, h, lastRGBA, fLast.hostMatte, fLast.matteComponents, dummyLuma, lastMatte);
    }

    float totalWindowW = 0.0f;
    for (int k = 0; k <= maxLookback; ++k) {
        totalWindowW += (params.trailBlendMode == eTrailCometDecay)
                            ? std::exp(-decayLambda * static_cast<float>(k) * pairStep)
                            : 1.0f;
        int idxA = centerIdx - k;
        int idxB = idxA - 1;
        const float* rgbaA = (k == 0) ? shutterSrc : frames[idxA].hostRGBA;
        bool hasB = (idxB >= 0 && k < maxLookback && frames[idxB].hostRGBA != nullptr);
        const float* rgbaB = hasB ? frames[idxB].hostRGBA : rgbaA;
        const auto& pairFlow = pairFlows[k];
        const auto& nextPairFlow = (k + 1 < maxLookback) ? pairFlows[k + 1] : nullptr;

        std::vector<float> matteA;
        if (hasMatte && frames[idxA].hostMatte && frames[idxA].matteComponents > 0) {
            extractLumaAndMatteCPU(w, h, rgbaA, frames[idxA].hostMatte, frames[idxA].matteComponents, dummyLuma, matteA);
        }

        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                size_t idx = static_cast<size_t>(y) * w + x;
                float mVal = (!matteA.empty()) ? matteA[idx] : 1.0f;
                if (fgIso && mVal < 0.5f) continue;

                if (hasMatte && params.matteMode == eMatteMaskEffect && mVal <= 0.001f) {
                    if (params.trailBlendMode != eTrailLightPainting) {
                        const float* pA = rgbaA + idx * 4;
                        float rA = dwiToLinearChannel(pA[0]);
                        float gA = dwiToLinearChannel(pA[1]);
                        float bA = dwiToLinearChannel(pA[2]);
                        float wBase = (params.trailBlendMode == eTrailCometDecay && k > 0)
                                          ? std::exp(-decayLambda * static_cast<float>(k) * pairStep)
                                          : 1.0f;
                        accum[idx * 4 + 0] += rA * wBase;
                        accum[idx * 4 + 1] += gA * wBase;
                        accum[idx * 4 + 2] += bA * wBase;
                        accum[idx * 4 + 3] += wBase;
                    }
                    continue;
                }

                Float4 fl = pairFlow ? pairFlow->hostFlow4[idx] : Float4{0, 0, 0, 0};
                float vx = fl.x * params.motionSensitivity;
                float vy = fl.y * params.motionSensitivity;
                float speed = std::hypot(vx, vy);

                float mGate = 1.0f;
                if (params.trailBlendMode == eTrailLightPainting && params.motionThreshold > 0.0f) {
                    mGate = smoothstepf(params.motionThreshold, params.motionThreshold + 0.75f, speed);
                    if (mGate <= 0.01f) continue;
                }

                const float* pA = rgbaA + idx * 4;
                float rA = dwiToLinearChannel(pA[0]);
                float gA = dwiToLinearChannel(pA[1]);
                float bA = dwiToLinearChannel(pA[2]);

                float lGate = 1.0f;
                if (params.trailBlendMode == eTrailLightPainting && params.lumaThreshold > 0.0f) {
                    float luma = dwgLuminance(rA, gA, bA);
                    float lumaDwi = linearToDwiChannel(luma);
                    lGate = smoothstepf(params.lumaThreshold, params.lumaThreshold + 0.12f, lumaDwi);
                }
                if (params.trailBlendMode == eTrailLightPainting && mGate * lGate <= 0.01f) {
                    continue;
                }

                float hdrW = 1.0f;
                if (params.trailBlendMode != eTrailLightPainting && params.highlightBoost > 0.0f) {
                    float luma = dwgLuminance(rA, gA, bA);
                    if (luma > 0.8f) {
                        float ex = luma - 0.8f;
                        hdrW += params.highlightBoost * ex * std::sqrt(ex);
                    }
                }
                float fgMatteW = fgIso ? mVal : 1.0f;

                if (params.subFrameSteps <= 1 || !hasB || speed < 0.35f) {
                    float timeW = (params.trailBlendMode == eTrailCometDecay)
                                      ? std::exp(-decayLambda * static_cast<float>(k) * pairStep)
                                      : 1.0f;
                    if (params.trailBlendMode == eTrailLightPainting) {
                        float gTot = mGate * lGate * timeW * fgMatteW;
                        if (gTot > 0.01f) {
                            accum[idx * 4 + 0] = std::max(accum[idx * 4 + 0], rA * gTot);
                            accum[idx * 4 + 1] = std::max(accum[idx * 4 + 1], gA * gTot);
                            accum[idx * 4 + 2] = std::max(accum[idx * 4 + 2], bA * gTot);
                        }
                    } else {
                        float wTot = timeW * hdrW * fgMatteW;
                        accum[idx * 4 + 0] += rA * wTot;
                        accum[idx * 4 + 1] += gA * wTot;
                        accum[idx * 4 + 2] += bA * wTot;
                        accum[idx * 4 + 3] += wTot;
                    }
                    continue;
                }

                float c1x = vx, c1y = vy, c2x = 0.0f, c2y = 0.0f;
                if (nextPairFlow) {
                    int bx = clampi(static_cast<int>(std::round(static_cast<float>(x) + fl.x)), 0, w - 1);
                    int by = clampi(static_cast<int>(std::round(static_cast<float>(y) + fl.y)), 0, h - 1);
                    size_t bIdx = static_cast<size_t>(by) * w + bx;
                    Float4 fl2 = nextPairFlow->hostFlow4[bIdx];
                    float u2x = fl2.x * params.motionSensitivity;
                    float u2y = fl2.y * params.motionSensitivity;
                    float c1 = pairFlow ? pairFlow->hostConf2[idx].x : 1.0f;
                    float c2 = nextPairFlow->hostConf2[bIdx].x;
                    float gCurv = std::min(c1, c2);
                    gCurv *= gCurv;
                    float dux = u2x - vx, duy = u2y - vy;
                    float duLen = std::hypot(dux, duy);
                    if (duLen > 1.5f * speed) {
                        float sc = (1.5f * speed) / std::max(1e-5f, duLen);
                        dux *= sc; duy *= sc;
                    }
                    float t1x = vx + 0.5f * gCurv * dux;
                    float t1y = vy + 0.5f * gCurv * duy;
                    c1x = 2.0f * vx - t1x;
                    c1y = 2.0f * vy - t1y;
                    c2x = t1x - vx;
                    c2y = t1y - vy;
                }
                float midX = x + 0.5f * c1x + 0.25f * c2x;
                float midY = y + 0.5f * c1y + 0.25f * c2y;
                float arcLen = std::hypot(midX - x, midY - y) + std::hypot((x + vx) - midX, (y + vy) - midY);

                int steps = std::max(params.subFrameSteps, std::min(72, static_cast<int>(std::ceil(std::max(speed, arcLen) * 1.25f))));
                float invSteps = 1.0f / steps;
                Float4 pB = sampleRGBABilinear(rgbaB, w, h, x + fl.x, y + fl.y);
                pB.x = dwiToLinearChannel(pB.x);
                pB.y = dwiToLinearChannel(pB.y);
                pB.z = dwiToLinearChannel(pB.z);
                float conf = pairFlow ? pairFlow->hostConf2[idx].x : 1.0f;

                float timeW = (params.trailBlendMode == eTrailCometDecay) ? std::exp(-decayLambda * static_cast<float>(k) * pairStep) : 1.0f;
                const float decayMul = (params.trailBlendMode == eTrailCometDecay) ? std::exp(-decayLambda * invSteps * pairStep) : 1.0f;

                for (int s = 0; s < steps; ++s) {
                    float alpha = s * invSteps;
                    float px = x + (c1x + c2x * alpha) * alpha;
                    float py = y + (c1y + c2y * alpha) * alpha;
                    float bb = alpha * conf;
                    float sr = rA * (1.0f - bb) + pB.x * bb;
                    float sg = gA * (1.0f - bb) + pB.y * bb;
                    float sb = bA * (1.0f - bb) + pB.z * bb;

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
                                float gate = mGate * lGate * timeW * fgMatteW;
                                if (gate > 0.01f && bw[c] > 0.01f) {
                                    float sgGain = std::min(1.0f, bw[c] * 1.35f) * gate;
                                    accum[tidx * 4 + 0] = std::max(accum[tidx * 4 + 0], sr * sgGain);
                                    accum[tidx * 4 + 1] = std::max(accum[tidx * 4 + 1], sg * sgGain);
                                    accum[tidx * 4 + 2] = std::max(accum[tidx * 4 + 2], sb * sgGain);
                                }
                            } else {
                                float wt = timeW * hdrW * fgMatteW * invSteps * bw[c];
                                accum[tidx * 4 + 0] += sr * wt;
                                accum[tidx * 4 + 1] += sg * wt;
                                accum[tidx * 4 + 2] += sb * wt;
                                accum[tidx * 4 + 3] += wt;
                            }
                        }
                    }
                    timeW *= decayMul;
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
            float accR = accum[i * 4 + 0], accG = accum[i * 4 + 1], accB = accum[i * 4 + 2], accW = accum[i * 4 + 3];
            float currM = hasMatte ? currMatte[i] : 1.0f;
            float lastM = (!lastMatte.empty()) ? lastMatte[i] : 1.0f;
            const float* pL = lastRGBA ? (lastRGBA + i * 4) : cur;
            float lr = dwiToLinearChannel(pL[0]), lg = dwiToLinearChannel(pL[1]), lb = dwiToLinearChannel(pL[2]);
            if (fgIso) {
                float bgW = std::max(0.0f, totalWindowW - accW);
                float bgR = (currM >= 0.5f && lastRGBA && lastM < 0.5f) ? lr : cr;
                float bgG = (currM >= 0.5f && lastRGBA && lastM < 0.5f) ? lg : cg;
                float bgB = (currM >= 0.5f && lastRGBA && lastM < 0.5f) ? lb : cb;
                accR += bgR * bgW; accG += bgG * bgW; accB += bgB * bgW; accW += bgW;
            } else if (accW < 0.25f && lastRGBA) {
                float fillW = 0.25f - accW;
                accR += lr * fillW; accG += lg * fillW; accB += lb * fillW; accW += fillW;
            }
            float wSum = std::max(1e-5f, accW);
            or_ = accR / wSum;
            og_ = accG / wSum;
            ob_ = accB / wSum;
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
