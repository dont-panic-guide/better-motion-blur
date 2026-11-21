#include "CudaEngine.h"
#include "NeuralFlowEngine.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace bmb {

namespace {

// CUDA Driver API types & function signatures
using CUresult = int;
using CUdevice = int;
using CUcontext = void*;
using CUmodule = void*;
using CUfunction = void*;
using CUstream = void*;
using CUdeviceptr = uint64_t;

constexpr CUresult CUDA_SUCCESS_VAL = 0;
constexpr int CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1;

using FnCuInit = CUresult(__stdcall*)(unsigned int Flags);
using FnCuDeviceGet = CUresult(__stdcall*)(CUdevice* device, int ordinal);
using FnCuDeviceGetName = CUresult(__stdcall*)(char* name, int len, CUdevice dev);
using FnCuDeviceGetAttribute = CUresult(__stdcall*)(int* pi, int attrib, CUdevice dev);
using FnCuCtxGetCurrent = CUresult(__stdcall*)(CUcontext* pctx);
using FnCuCtxCreate = CUresult(__stdcall*)(CUcontext* pctx, unsigned int flags, CUdevice dev);
using FnCuCtxDestroy = CUresult(__stdcall*)(CUcontext ctx);
using FnCuCtxSetCurrent = CUresult(__stdcall*)(CUcontext ctx);
using FnCuModuleLoadDataEx = CUresult(__stdcall*)(CUmodule* module, const void* image, unsigned int numOptions, int* options, void** optionValues);
using FnCuModuleUnload = CUresult(__stdcall*)(CUmodule hmod);
using FnCuModuleGetFunction = CUresult(__stdcall*)(CUfunction* hfunc, CUmodule hmod, const char* name);
using FnCuMemAlloc = CUresult(__stdcall*)(CUdeviceptr* dptr, size_t bytesize);
using FnCuMemFree = CUresult(__stdcall*)(CUdeviceptr dptr);
using FnCuMemGetInfo = CUresult(__stdcall*)(size_t* freeBytes, size_t* totalBytes);
using FnCuMemcpyHtoDAsync = CUresult(__stdcall*)(CUdeviceptr dstDevice, const void* srcHost, size_t ByteCount, CUstream hStream);
using FnCuMemcpyDtoHAsync = CUresult(__stdcall*)(void* dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
using FnCuMemcpyDtoDAsync = CUresult(__stdcall*)(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream);
using FnCuMemsetD8Async = CUresult(__stdcall*)(CUdeviceptr dstDevice, unsigned char uc, size_t N, CUstream hStream);
using FnCuLaunchKernel = CUresult(__stdcall*)(CUfunction f,
                                              unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                              unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                              unsigned int sharedMemBytes, CUstream hStream,
                                              void** kernelParams, void** extra);
using FnCuStreamSynchronize = CUresult(__stdcall*)(CUstream hStream);

// NVRTC types & function signatures
using nvrtcResult = int;
using nvrtcProgram = void*;
constexpr nvrtcResult NVRTC_SUCCESS_VAL = 0;

using FnNvrtcCreateProgram = nvrtcResult(__cdecl*)(nvrtcProgram* prog, const char* src, const char* name, int numHeaders, const char* const* headers, const char* const* includeNames);
using FnNvrtcDestroyProgram = nvrtcResult(__cdecl*)(nvrtcProgram* prog);
using FnNvrtcCompileProgram = nvrtcResult(__cdecl*)(nvrtcProgram prog, int numOptions, const char* const* options);
using FnNvrtcGetPTXSize = nvrtcResult(__cdecl*)(nvrtcProgram prog, size_t* ptxSizeRet);
using FnNvrtcGetPTX = nvrtcResult(__cdecl*)(nvrtcProgram prog, char* ptx);
using FnNvrtcGetProgramLogSize = nvrtcResult(__cdecl*)(nvrtcProgram prog, size_t* logSizeRet);
using FnNvrtcGetProgramLog = nvrtcResult(__cdecl*)(nvrtcProgram prog, char* log);

struct CudaDriverTable {
    FnCuInit cuInit = nullptr;
    FnCuDeviceGet cuDeviceGet = nullptr;
    FnCuDeviceGetName cuDeviceGetName = nullptr;
    FnCuDeviceGetAttribute cuDeviceGetAttribute = nullptr;
    FnCuCtxGetCurrent cuCtxGetCurrent = nullptr;
    FnCuCtxCreate cuCtxCreate = nullptr;
    FnCuCtxDestroy cuCtxDestroy = nullptr;
    FnCuCtxSetCurrent cuCtxSetCurrent = nullptr;
    FnCuModuleLoadDataEx cuModuleLoadDataEx = nullptr;
    FnCuModuleUnload cuModuleUnload = nullptr;
    FnCuModuleGetFunction cuModuleGetFunction = nullptr;
    FnCuMemAlloc cuMemAlloc = nullptr;
    FnCuMemFree cuMemFree = nullptr;
    FnCuMemGetInfo cuMemGetInfo = nullptr;
    FnCuMemcpyHtoDAsync cuMemcpyHtoDAsync = nullptr;
    FnCuMemcpyDtoHAsync cuMemcpyDtoHAsync = nullptr;
    FnCuMemcpyDtoDAsync cuMemcpyDtoDAsync = nullptr;
    FnCuMemsetD8Async cuMemsetD8Async = nullptr;
    FnCuLaunchKernel cuLaunchKernel = nullptr;
    FnCuStreamSynchronize cuStreamSynchronize = nullptr;

    FnNvrtcCreateProgram nvrtcCreateProgram = nullptr;
    FnNvrtcDestroyProgram nvrtcDestroyProgram = nullptr;
    FnNvrtcCompileProgram nvrtcCompileProgram = nullptr;
    FnNvrtcGetPTXSize nvrtcGetPTXSize = nullptr;
    FnNvrtcGetPTX nvrtcGetPTX = nullptr;
    FnNvrtcGetProgramLogSize nvrtcGetProgramLogSize = nullptr;
    FnNvrtcGetProgramLog nvrtcGetProgramLog = nullptr;
} g_cu;

const char* kEmbeddedCudaSource = R"CUDA(
extern "C" {

__device__ __forceinline__ int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

__device__ __forceinline__ float clampf(float v, float lo, float hi) {
    return fminf(hi, fmaxf(lo, v));
}

__device__ __forceinline__ float smoothstepf(float edge0, float edge1, float x) {
    float t = clampf((x - edge0) / fmaxf(1e-6f, edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

#define DI_A       0.0075f
#define DI_B       7.0f
#define DI_C       0.07329248f
#define DI_M       10.44426855f
#define DI_LIN_CUT 0.00262409f
#define DI_LOG_CUT 0.02740668f

__device__ __forceinline__ float dwiToLinearChannel(float v) {
    return (v <= DI_LOG_CUT) ? (v / DI_M) : (exp2f(v / DI_C - DI_B) - DI_A);
}

__device__ __forceinline__ float linearToDwiChannel(float l) {
    return (l <= DI_LIN_CUT) ? (l * DI_M) : ((log2f(l + DI_A) + DI_B) * DI_C);
}

__device__ __forceinline__ float dwgLuminance(float r, float g, float b) {
    return fmaxf(0.0f, 0.27411851f * r + 0.87363190f * g - 0.14775041f * b);
}

__device__ __forceinline__ float4 sampleRGBA_Bilinear(const float* img, int w, int h, float x, float y) {
    float cx = clampf(x, 0.0f, (float)(w - 1));
    float cy = clampf(y, 0.0f, (float)(h - 1));
    int x0 = (int)floorf(cx);
    int y0 = (int)floorf(cy);
    int x1 = min(x0 + 1, w - 1);
    int y1 = min(y0 + 1, h - 1);
    float fx = cx - (float)x0;
    float fy = cy - (float)y0;

    const float4* img4 = (const float4*)img;
    float4 p00 = __ldg(&img4[y0 * w + x0]);
    float4 p10 = __ldg(&img4[y0 * w + x1]);
    float4 p01 = __ldg(&img4[y1 * w + x0]);
    float4 p11 = __ldg(&img4[y1 * w + x1]);

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    float4 r;
    r.x = p00.x * w00 + p10.x * w10 + p01.x * w01 + p11.x * w11;
    r.y = p00.y * w00 + p10.y * w10 + p01.y * w01 + p11.y * w11;
    r.z = p00.z * w00 + p10.z * w10 + p01.z * w01 + p11.z * w11;
    r.w = p00.w * w00 + p10.w * w10 + p01.w * w01 + p11.w * w11;
    return r;
}

__device__ __forceinline__ float sampleFloat1_Bilinear(const float* img, int w, int h, float x, float y) {
    float cx = clampf(x, 0.0f, (float)(w - 1));
    float cy = clampf(y, 0.0f, (float)(h - 1));
    int x0 = (int)floorf(cx);
    int y0 = (int)floorf(cy);
    int x1 = min(x0 + 1, w - 1);
    int y1 = min(y0 + 1, h - 1);
    float fx = cx - (float)x0;
    float fy = cy - (float)y0;
    float v00 = __ldg(&img[y0 * w + x0]);
    float v10 = __ldg(&img[y0 * w + x1]);
    float v01 = __ldg(&img[y1 * w + x0]);
    float v11 = __ldg(&img[y1 * w + x1]);
    return (v00 * (1.0f - fx) + v10 * fx) * (1.0f - fy) + (v01 * (1.0f - fx) + v11 * fx) * fy;
}

__device__ __forceinline__ float2 sampleFloat2_Bilinear(const float2* img, int w, int h, float x, float y) {
    float cx = clampf(x, 0.0f, (float)(w - 1));
    float cy = clampf(y, 0.0f, (float)(h - 1));
    int x0 = (int)floorf(cx);
    int y0 = (int)floorf(cy);
    int x1 = min(x0 + 1, w - 1);
    int y1 = min(y0 + 1, h - 1);
    float fx = cx - (float)x0;
    float fy = cy - (float)y0;

    float2 v00 = __ldg(&img[y0 * w + x0]);
    float2 v10 = __ldg(&img[y0 * w + x1]);
    float2 v01 = __ldg(&img[y1 * w + x0]);
    float2 v11 = __ldg(&img[y1 * w + x1]);

    float w00 = (1.0f - fx) * (1.0f - fy);
    float w10 = fx * (1.0f - fy);
    float w01 = (1.0f - fx) * fy;
    float w11 = fx * fy;

    return make_float2(v00.x * w00 + v10.x * w10 + v01.x * w01 + v11.x * w11,
                       v00.y * w00 + v10.y * w10 + v01.y * w01 + v11.y * w11);
}

__device__ __forceinline__ unsigned int packHalf2(float a, float b) {
    unsigned short h0, h1;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h0) : "f"(a));
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h1) : "f"(b));
    return (unsigned int)h0 | ((unsigned int)h1 << 16);
}

__device__ __forceinline__ float2 unpackHalf2(unsigned int v) {
    unsigned short h0 = (unsigned short)(v & 0xFFFFu);
    unsigned short h1 = (unsigned short)(v >> 16);
    float a, b;
    asm("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(h0));
    asm("cvt.f32.f16 %0, %1;" : "=f"(b) : "h"(h1));
    return make_float2(a, b);
}

__device__ __forceinline__ float2 readFlowVec(const uint2* f4, int idx, int rev) {
    uint2 v = __ldg(&f4[idx]);
    return unpackHalf2(rev ? v.y : v.x);
}

__device__ __forceinline__ float readConfVal(const uchar2* c2, int idx, int rev) {
    uchar2 v = __ldg(&c2[idx]);
    return (float)(rev ? v.y : v.x) * (1.0f / 255.0f);
}

__device__ __forceinline__ float readMatteVal(const float* matte, int idx, int matteComp) {
    if (!matte || matteComp <= 0) return 1.0f;
    if (matteComp == 1) {
        return clampf(__ldg(&matte[idx]), 0.0f, 1.0f);
    }
    const float4* matte4 = (const float4*)matte;
    float4 m = __ldg(&matte4[idx]);
    float mv = (m.w > 0.0f && m.w < 0.9999f) ? m.w : (0.2126f * m.x + 0.7152f * m.y + 0.0722f * m.z);
    return clampf(mv, 0.0f, 1.0f);
}

__device__ __forceinline__ void atomicMaxPosFloat(float* addr, float val) {
    if (val > 0.0f) {
        atomicMax((int*)addr, __float_as_int(val));
    }
}

__global__ void extractLumaAndMatteKernel(
    int w, int h, const float* rgba, const float* matte, int matteComp,
    float* outLuma, float* outMatte)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    const float4* rgba4 = (const float4*)rgba;
    float4 p = __ldg(&rgba4[idx]);
    outLuma[idx] = dwgLuminance(p.x, p.y, p.z);

    if (outMatte) {
        if (matte && matteComp > 0) {
            if (matteComp == 1) {
                outMatte[idx] = clampf(__ldg(&matte[idx]), 0.0f, 1.0f);
            } else {
                const float4* matte4 = (const float4*)matte;
                float4 m = __ldg(&matte4[idx]);
                float mv = (m.w > 0.0f && m.w < 0.9999f) ? m.w : (0.2126f * m.x + 0.7152f * m.y + 0.0722f * m.z);
                outMatte[idx] = clampf(mv, 0.0f, 1.0f);
            }
        } else {
            outMatte[idx] = 1.0f;
        }
    }
}

__global__ void downsample2xKernel(
    int srcW, int srcH, int dstW, int dstH, const float* src, float* dst)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    const float kw[4] = {0.125f, 0.375f, 0.375f, 0.125f};
    int cx = x * 2;
    int cy = y * 2;
    float sum = 0.0f;
    for (int dy = -1; dy <= 2; ++dy) {
        int sy = clampi(cy + dy, 0, srcH - 1);
        float wy = kw[dy + 1];
        for (int dx = -1; dx <= 2; ++dx) {
            int sx = clampi(cx + dx, 0, srcW - 1);
            sum += wy * kw[dx + 1] * src[sy * srcW + sx];
        }
    }
    dst[y * dstW + x] = sum;
}

__global__ void initZeroFlowKernel(int w, int h, float2* flow) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    flow[y * w + x] = make_float2(0.0f, 0.0f);
}

__global__ void upsampleFlow2xKernel(
    int srcW, int srcH, int dstW, int dstH, const float2* srcFlow, float2* dstFlow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dstW || y >= dstH) return;

    float sx = ((float)x + 0.5f) * ((float)srcW / (float)dstW) - 0.5f;
    float sy = ((float)y + 0.5f) * ((float)srcH / (float)dstH) - 0.5f;
    float scaleX = (float)dstW / (float)srcW;
    float scaleY = (float)dstH / (float)srcH;

    float2 f = sampleFloat2_Bilinear(srcFlow, srcW, srcH, sx, sy);
    dstFlow[y * dstW + x] = make_float2(f.x * scaleX, f.y * scaleY);
}

// Coarse-to-Fine PatchMatch + Multi-Radius Search + Iterative Lucas-Kanade Kernel
__device__ __forceinline__ float evalPatchCostShared(
    int x, int y, int w, int h,
    int baseX, int baseY, int mBaseX, int mBaseY,
    const float sLumaA[22][22], const float sMatteA[20][20],
    const float* lumaB,
    int fgCenter, int hasMatte, float vx, float vy, float maxCost)
{
    float cost = 0.0f;
    float wsum = 0.0f;
    const float rvx = rintf(vx);
    const float rvy = rintf(vy);
    const bool isInt = (fabsf(vx - rvx) < 1e-4f) && (fabsf(vy - rvy) < 1e-4f);
    const int ivx = (int)rvx;
    const int ivy = (int)rvy;
    const float maxSum = (!hasMatte && maxCost < 1e8f) ? (maxCost * 25.0f + 1e-3f) : 1e30f;

    for (int dy = -2; dy <= 2; ++dy) {
        int ay = clampi(y + dy, 0, h - 1);
        int syA = ay - baseY;
        int smyA = ay - mBaseY;
        for (int dx = -2; dx <= 2; ++dx) {
            int ax = clampi(x + dx, 0, w - 1);
            float a = sLumaA[syA][ax - baseX];
            float diff;
            if (isInt) {
                int bx = ax + ivx;
                int by = ay + ivy;
                if (bx < 0 || bx >= w || by < 0 || by >= h) {
                    diff = 0.25f;
                } else {
                    float b = __ldg(&lumaB[by * w + bx]);
                    diff = fabsf(a - b);
                }
            } else {
                float bx = (float)ax + vx;
                float by = (float)ay + vy;
                if (bx < 0.0f || bx > (float)(w - 1) || by < 0.0f || by > (float)(h - 1)) {
                    diff = 0.25f;
                } else {
                    float b = sampleFloat1_Bilinear(lumaB, w, h, bx, by);
                    diff = fabsf(a - b);
                }
            }
            if (hasMatte) {
                int fgN = (sMatteA[smyA][ax - mBaseX] >= 0.5f) ? 1 : 0;
                float weight = (fgN != fgCenter) ? 0.05f : 1.0f;
                cost += weight * diff;
                wsum += weight;
            } else {
                cost += diff;
                if (cost > maxSum) return 1e9f;
            }
        }
    }
    return hasMatte ? (cost / fmaxf(1e-4f, wsum)) : (cost * 0.04f);
}

__global__ void patchMatchLKFlowKernel(
    int w, int h,
    const float* lumaA, const float* matteA,
    const float* lumaB,
    const float2* inFlow, float2* outFlow,
    int searchRadius, int lkIterations, int hasMatte)
{
    __shared__ float sLumaA[22][22];
    __shared__ float sMatteA[20][20];

    const int bx0 = (int)blockIdx.x * 16;
    const int by0 = (int)blockIdx.y * 16;
    const int baseX = bx0 - 3;
    const int baseY = by0 - 3;
    const int mBaseX = bx0 - 2;
    const int mBaseY = by0 - 2;
    const int tid = (int)threadIdx.y * 16 + (int)threadIdx.x;

    for (int k = tid; k < 22 * 22; k += 256) {
        int sy = k / 22;
        int sx = k - sy * 22;
        int gx = clampi(baseX + sx, 0, w - 1);
        int gy = clampi(baseY + sy, 0, h - 1);
        sLumaA[sy][sx] = __ldg(&lumaA[gy * w + gx]);
    }
    const int activeMatte = (hasMatte && matteA) ? 1 : 0;
    if (activeMatte) {
        for (int k = tid; k < 20 * 20; k += 256) {
            int sy = k / 20;
            int sx = k - sy * 20;
            int gx = clampi(mBaseX + sx, 0, w - 1);
            int gy = clampi(mBaseY + sy, 0, h - 1);
            sMatteA[sy][sx] = __ldg(&matteA[gy * w + gx]);
        }
    }
    __syncthreads();

    int x = bx0 + (int)threadIdx.x;
    int y = by0 + (int)threadIdx.y;
    if (x >= w || y >= h) return;

    float mCenter = activeMatte ? sMatteA[(int)threadIdx.y + 2][(int)threadIdx.x + 2] : 1.0f;
    int fgCenter = (mCenter >= 0.5f) ? 1 : 0;

    float2 bestFlow = __ldg(&inFlow[y * w + x]);
    float bestCost = evalPatchCostShared(
        x, y, w, h, baseX, baseY, mBaseX, mBaseY,
        sLumaA, sMatteA, lumaB, fgCenter, activeMatte, bestFlow.x, bestFlow.y, 1e9f);

    // Zero candidate helps static background snap cleanly
    bool testedZero = (fabsf(bestFlow.x) < 1e-4f && fabsf(bestFlow.y) < 1e-4f);
    if (!testedZero) {
        float c0 = evalPatchCostShared(
            x, y, w, h, baseX, baseY, mBaseX, mBaseY,
            sLumaA, sMatteA, lumaB, fgCenter, activeMatte, 0.0f, 0.0f, bestCost);
        if (c0 + 1e-4f < bestCost) {
            bestCost = c0;
            bestFlow = make_float2(0.0f, 0.0f);
        }
    }

    // Spatial neighbor propagation candidates (deduplicated against bestFlow / zero)
    const int nOffsets[4][2] = {{-2, 0}, {2, 0}, {0, -2}, {0, 2}};
    for (int i = 0; i < 4; ++i) {
        int nx = clampi(x + nOffsets[i][0], 0, w - 1);
        int ny = clampi(y + nOffsets[i][1], 0, h - 1);
        float2 cand = __ldg(&inFlow[ny * w + nx]);
        if (fabsf(cand.x - bestFlow.x) < 1e-4f && fabsf(cand.y - bestFlow.y) < 1e-4f) continue;
        if (fabsf(cand.x) < 1e-4f && fabsf(cand.y) < 1e-4f) continue;
        float c = evalPatchCostShared(
            x, y, w, h, baseX, baseY, mBaseX, mBaseY,
            sLumaA, sMatteA, lumaB, fgCenter, activeMatte, cand.x, cand.y, bestCost);
        if (c + 1e-4f < bestCost) {
            bestCost = c;
            bestFlow = cand;
        }
    }

    // Exhaustive 2D local search around base flow at coarse levels
    if (searchRadius > 0) {
        float2 base = bestFlow;
        float regW = (searchRadius >= 2) ? 1.0e-4f : 0.0f;
        float bestScore = bestCost + regW * (base.x * base.x + base.y * base.y);
        for (int dy = -searchRadius; dy <= searchRadius; ++dy) {
            for (int dx = -searchRadius; dx <= searchRadius; ++dx) {
                if (dx == 0 && dy == 0) continue;
                float cx = base.x + (float)dx;
                float cy = base.y + (float)dy;
                float regCost = regW * (cx * cx + cy * cy);
                float maxAllowed = bestScore - regCost;
                if (maxAllowed <= 0.0f) continue;
                float c = evalPatchCostShared(
                    x, y, w, h, baseX, baseY, mBaseX, mBaseY,
                    sLumaA, sMatteA, lumaB, fgCenter, activeMatte, cx, cy, maxAllowed);
                float score = c + regCost;
                if (score + 1e-5f < bestScore) {
                    bestScore = score;
                    bestCost = c;
                    bestFlow = make_float2(cx, cy);
                }
            }
        }
    }

    // Sub-pixel Iterative Lucas-Kanade / Structure Tensor Refinement
    float vx = bestFlow.x;
    float vy = bestFlow.y;

    if (lkIterations > 0) {
        float valA[25];
        float gxArr[25];
        float gyArr[25];
        float wArr[25];
        float Ixx = 0.0f, Ixy = 0.0f, Iyy = 0.0f;
        int k = 0;

        for (int dy = -2; dy <= 2; ++dy) {
            int ay = clampi(y + dy, 0, h - 1);
            int sy = ay - baseY;
            int sym = clampi(ay - 1, 0, h - 1) - baseY;
            int syp = clampi(ay + 1, 0, h - 1) - baseY;
            int smy = ay - mBaseY;
            for (int dx = -2; dx <= 2; ++dx) {
                int ax = clampi(x + dx, 0, w - 1);
                int sx = ax - baseX;
                int sxm = clampi(ax - 1, 0, w - 1) - baseX;
                int sxp = clampi(ax + 1, 0, w - 1) - baseX;
                float weight = (dx == 0 && dy == 0) ? 1.5f : 1.0f;
                if (activeMatte) {
                    int fgN = (sMatteA[smy][ax - mBaseX] >= 0.5f) ? 1 : 0;
                    if (fgN != fgCenter) weight = 0.02f;
                }
                float gx = 0.5f * (sLumaA[sy][sxp] - sLumaA[sy][sxm]);
                float gy = 0.5f * (sLumaA[syp][sx] - sLumaA[sym][sx]);
                valA[k] = sLumaA[sy][sx];
                gxArr[k] = gx;
                gyArr[k] = gy;
                wArr[k] = weight;
                ++k;

                Ixx += weight * gx * gx;
                Ixy += weight * gx * gy;
                Iyy += weight * gy * gy;
            }
        }

        float reg = 0.0025f;
        float a00 = Ixx + reg;
        float a11 = Iyy + reg;
        float det = a00 * a11 - Ixy * Ixy;

        if (det > 1e-7f) {
            float invDet = 1.0f / det;
            for (int iter = 0; iter < lkIterations; ++iter) {
                float bx = 0.0f, by = 0.0f;
                int kk = 0;
                for (int dy = -2; dy <= 2; ++dy) {
                    int ay = clampi(y + dy, 0, h - 1);
                    for (int dx = -2; dx <= 2; ++dx) {
                        int ax = clampi(x + dx, 0, w - 1);
                        float valB = sampleFloat1_Bilinear(lumaB, w, h, (float)ax + vx, (float)ay + vy);
                        float It = valB - valA[kk];
                        float wTot = wArr[kk] * (0.08f / fmaxf(0.08f, fabsf(It)));
                        bx -= wTot * gxArr[kk] * It;
                        by -= wTot * gyArr[kk] * It;
                        ++kk;
                    }
                }
                float dvx = invDet * (a11 * bx - Ixy * by);
                float dvy = invDet * (a00 * by - Ixy * bx);
                float stepLen = hypotf(dvx, dvy);
                if (stepLen > 1.25f) {
                    dvx *= 1.25f / stepLen;
                    dvy *= 1.25f / stepLen;
                }
                float candVx = vx + dvx;
                float candVy = vy + dvy;
                float candCost = evalPatchCostShared(
                    x, y, w, h, baseX, baseY, mBaseX, mBaseY,
                    sLumaA, sMatteA, lumaB, fgCenter, activeMatte, candVx, candVy, bestCost);
                if (candCost <= bestCost) {
                    vx = candVx;
                    vy = candVy;
                    bestCost = candCost;
                } else {
                    break;
                }
            }
        }
    }

    outFlow[y * w + x] = make_float2(vx, vy);
}

// Neural / All-Pairs Correlation Cost Volume Update at Proxy Resolution
__global__ void correlationFlowKernel(
    int w, int h,
    const float* lumaA, const float* matteA,
    const float* lumaB,
    const float2* inFlow, float2* outFlow, float* outGradXY)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    int xm = max(0, x - 1), xp = min(w - 1, x + 1);
    int ym = max(0, y - 1), yp = min(h - 1, y + 1);

    // Preload 3x3 patch in A into registers once
    float aPatch[9];
    aPatch[0] = __ldg(&lumaA[ym * w + xm]); aPatch[1] = __ldg(&lumaA[ym * w + x]); aPatch[2] = __ldg(&lumaA[ym * w + xp]);
    aPatch[3] = __ldg(&lumaA[y  * w + xm]); aPatch[4] = __ldg(&lumaA[y  * w + x]); aPatch[5] = __ldg(&lumaA[y  * w + xp]);
    aPatch[6] = __ldg(&lumaA[yp * w + xm]); aPatch[7] = __ldg(&lumaA[yp * w + x]); aPatch[8] = __ldg(&lumaA[yp * w + xp]);

    if (outGradXY) {
        outGradXY[idx * 2 + 0] = 0.5f * (aPatch[5] - aPatch[3]);
        outGradXY[idx * 2 + 1] = 0.5f * (aPatch[7] - aPatch[1]);
    }

    float2 cur = __ldg(&inFlow[idx]);
    float bestScore = 1e9f;
    float2 bestDelta = cur;

    const int axList[3] = {xm, x, xp};
    const int ayList[3] = {ym, y, yp};

    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            float sx = cur.x + (float)dx * 0.5f;
            float sy = cur.y + (float)dy * 0.5f;
            float err = 0.0f;
            for (int py = 0; py < 3; ++py) {
                float by = (float)ayList[py] + sy;
                int base = py * 3;
                for (int px = 0; px < 3; ++px) {
                    float b = sampleFloat1_Bilinear(lumaB, w, h, (float)axList[px] + sx, by);
                    err += fabsf(aPatch[base + px] - b);
                }
                if (err >= bestScore) break;
            }
            if (err < bestScore) {
                bestScore = err;
                bestDelta = make_float2(sx, sy);
            }
        }
    }

    outFlow[idx] = bestDelta;
}

// GPU-native 3x3 Laplacian residual flow refinement (exact match to embedded ONNX graph)
__global__ void embeddedResidualRefineKernel(
    int w, int h, const float2* inFlow, float2* outFlow)
{
    __shared__ float2 sFlow[18][18];
    const int baseX = (int)blockIdx.x * 16 - 1;
    const int baseY = (int)blockIdx.y * 16 - 1;
    const int tid = (int)threadIdx.y * 16 + (int)threadIdx.x;

    for (int k = tid; k < 18 * 18; k += 256) {
        int sy = k / 18;
        int sx = k - sy * 18;
        int gx = clampi(baseX + sx, 0, w - 1);
        int gy = clampi(baseY + sy, 0, h - 1);
        sFlow[sy][sx] = __ldg(&inFlow[gy * w + gx]);
    }
    __syncthreads();

    int x = (int)blockIdx.x * 16 + (int)threadIdx.x;
    int y = (int)blockIdx.y * 16 + (int)threadIdx.y;
    if (x >= w || y >= h) return;

    int stx = (int)threadIdx.x + 1;
    int sty = (int)threadIdx.y + 1;
    float2 c = sFlow[sty][stx];

    float2 tl = sFlow[sty - 1][stx - 1], tc = sFlow[sty - 1][stx], tr = sFlow[sty - 1][stx + 1];
    float2 ml = sFlow[sty    ][stx - 1],                            mr = sFlow[sty    ][stx + 1];
    float2 bl = sFlow[sty + 1][stx - 1], bc = sFlow[sty + 1][stx], br = sFlow[sty + 1][stx + 1];

    float lapX = 0.0625f * (tl.x + tr.x + bl.x + br.x) + 0.125f * (tc.x + ml.x + mr.x + bc.x) - 0.75f * c.x;
    float lapY = 0.0625f * (tl.y + tr.y + bl.y + br.y) + 0.125f * (tc.y + ml.y + mr.y + bc.y) - 0.75f * c.y;

    outFlow[y * w + x] = make_float2(c.x + 0.08f * lapX, c.y + 0.08f * lapY);
}

__global__ void jointBilateralFlowFilterKernel(
    int w, int h,
    const float2* inFlow, const float* guideLuma, const float* guideMatte,
    float sigmaRange, int hasMatte,
    float2* outFlow)
{
    __shared__ float2 sFlow[20][20];
    __shared__ float sLuma[20][20];
    __shared__ float sMatte[20][20];

    const int bx0 = (int)blockIdx.x * 16;
    const int by0 = (int)blockIdx.y * 16;
    const int baseX = bx0 - 2;
    const int baseY = by0 - 2;
    const int tid = (int)threadIdx.y * 16 + (int)threadIdx.x;
    const int activeMatte = (hasMatte && guideMatte) ? 1 : 0;

    for (int k = tid; k < 20 * 20; k += 256) {
        int sy = k / 20;
        int sx = k - sy * 20;
        int gx = clampi(baseX + sx, 0, w - 1);
        int gy = clampi(baseY + sy, 0, h - 1);
        int gidx = gy * w + gx;
        sFlow[sy][sx] = __ldg(&inFlow[gidx]);
        sLuma[sy][sx] = __ldg(&guideLuma[gidx]);
        if (activeMatte) {
            sMatte[sy][sx] = __ldg(&guideMatte[gidx]);
        }
    }
    __syncthreads();

    int x = bx0 + (int)threadIdx.x;
    int y = by0 + (int)threadIdx.y;
    if (x >= w || y >= h) return;

    int stx = (int)threadIdx.x + 2;
    int sty = (int)threadIdx.y + 2;
    float lc = sLuma[sty][stx];
    int fgC = activeMatte ? ((sMatte[sty][stx] >= 0.5f) ? 1 : 0) : 1;

    float inv2SigS2 = 1.0f / (2.0f * 2.0f * 2.0f);
    float inv2SigR2 = 1.0f / (2.0f * fmaxf(0.01f, sigmaRange) * fmaxf(0.01f, sigmaRange));

    float2 sumF = make_float2(0.0f, 0.0f);
    float sumW = 0.0f;

    for (int dy = -2; dy <= 2; ++dy) {
        int ny = clampi(y + dy, 0, h - 1) - baseY;
        for (int dx = -2; dx <= 2; ++dx) {
            int nx = clampi(x + dx, 0, w - 1) - baseX;
            float ln = sLuma[ny][nx];
            float dl = lc - ln;
            float weight = expf(-(float)(dx * dx + dy * dy) * inv2SigS2 - dl * dl * inv2SigR2);
            if (activeMatte) {
                int fgN = (sMatte[ny][nx] >= 0.5f) ? 1 : 0;
                if (fgN != fgC) weight *= 0.001f;
            }
            float2 nf = sFlow[ny][nx];
            sumF.x += weight * nf.x;
            sumF.y += weight * nf.y;
            sumW += weight;
        }
    }

    outFlow[y * w + x] = make_float2(sumF.x / fmaxf(1e-6f, sumW), sumF.y / fmaxf(1e-6f, sumW));
}

__global__ void jointBilateralUpsampleKernel(
    int lowW, int lowH, int highW, int highH,
    const float2* lowFlow, const float* lowLuma,
    const float* highLuma, const float* highMatte,
    float sigmaRange, int hasMatte,
    float2* highFlow)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= highW || y >= highH) return;

    int idx = y * highW + x;
    float lh = __ldg(&highLuma[idx]);
    float mh = highMatte ? __ldg(&highMatte[idx]) : 1.0f;

    float scaleX = (float)highW / (float)lowW;
    float scaleY = (float)highH / (float)lowH;
    float lx = ((float)x + 0.5f) / scaleX - 0.5f;
    float ly = ((float)y + 0.5f) / scaleY - 0.5f;

    int lx0 = (int)floorf(lx);
    int ly0 = (int)floorf(ly);
    float inv2SigR2 = 1.0f / (2.0f * fmaxf(0.015f, sigmaRange) * fmaxf(0.015f, sigmaRange));

    float2 sumF = make_float2(0.0f, 0.0f);
    float sumW = 0.0f;

    for (int dy = -1; dy <= 2; ++dy) {
        int sy = clampi(ly0 + dy, 0, lowH - 1);
        for (int dx = -1; dx <= 2; ++dx) {
            int sx = clampi(lx0 + dx, 0, lowW - 1);
            float dist2 = ((float)sx - lx) * ((float)sx - lx) + ((float)sy - ly) * ((float)sy - ly);
            float ll = __ldg(&lowLuma[sy * lowW + sx]);
            float dl = lh - ll;
            float w = expf(-dist2 * 0.5f - dl * dl * inv2SigR2);
            if (hasMatte && highMatte) {
                int hx = clampi((int)(((float)sx + 0.5f) * scaleX), 0, highW - 1);
                int hy = clampi((int)(((float)sy + 0.5f) * scaleY), 0, highH - 1);
                float dm = mh - __ldg(&highMatte[hy * highW + hx]);
                w *= expf(-dm * dm * 40.0f);
            }
            float2 lf = __ldg(&lowFlow[sy * lowW + sx]);
            sumF.x += w * lf.x * scaleX;
            sumF.y += w * lf.y * scaleY;
            sumW += w;
        }
    }

    highFlow[idx] = make_float2(sumF.x / fmaxf(1e-6f, sumW), sumF.y / fmaxf(1e-6f, sumW));
}

__global__ void packBidirectionalKernel(
    int w, int h,
    const float2* fwdFlow, const float2* bwdFlow,
    uint2* outFlow4, uchar2* outConf2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    float2 f = __ldg(&fwdFlow[idx]);
    float2 bLocal = __ldg(&bwdFlow[idx]);
    float2 bAtDest = sampleFloat2_Bilinear(bwdFlow, w, h, (float)x + f.x, (float)y + f.y);
    float2 fAtDest = sampleFloat2_Bilinear(fwdFlow, w, h, (float)x + bLocal.x, (float)y + bLocal.y);

    float errFX = f.x + bAtDest.x;
    float errFY = f.y + bAtDest.y;
    float confFwd = expf(-(errFX * errFX + errFY * errFY) * 0.25f);

    float errBX = bLocal.x + fAtDest.x;
    float errBY = bLocal.y + fAtDest.y;
    float confBwd = expf(-(errBX * errBX + errBY * errBY) * 0.25f);

    outFlow4[idx] = make_uint2(packHalf2(f.x, f.y), packHalf2(bLocal.x, bLocal.y));
    unsigned char cF = (unsigned char)(clampf(confFwd, 0.0f, 1.0f) * 255.0f + 0.5f);
    unsigned char cB = (unsigned char)(clampf(confBwd, 0.0f, 1.0f) * 255.0f + 0.5f);
    outConf2[idx] = make_uchar2(cF, cB);
}

// Coalesced 16x16 block-per-tile shared-memory parallel reduction
__global__ void tileMaxVelKernel(
    int w, int h, int tileW, int tileH, int tileSize,
    const uint2* flowFwd, int fwdRev,
    const uint2* flowBwd, int bwdRev,
    float sensitivity, float threshold,
    float2* tileMax)
{
    __shared__ float sMag[256];
    __shared__ float2 sVel[256];

    int tx = blockIdx.x;
    int ty = blockIdx.y;
    int x = tx * tileSize + threadIdx.x;
    int y = ty * tileSize + threadIdx.y;
    int tid = threadIdx.y * 16 + threadIdx.x;

    float mag = 0.0f;
    float2 vel = make_float2(0.0f, 0.0f);
    if (x < w && y < h) {
        int idx = y * w + x;
        float2 vf = flowFwd ? readFlowVec(flowFwd, idx, fwdRev) : make_float2(0.0f, 0.0f);
        float2 vb = flowBwd ? readFlowVec(flowBwd, idx, bwdRev) : make_float2(0.0f, 0.0f);
        if (!flowFwd && flowBwd) { vf = make_float2(-vb.x, -vb.y); }
        if (!flowBwd && flowFwd) { vb = make_float2(-vf.x, -vf.y); }
        float2 v0 = make_float2(0.5f * (vf.x - vb.x) * sensitivity, 0.5f * (vf.y - vb.y) * sensitivity);
        float m = hypotf(v0.x, v0.y);
        if (m > threshold) {
            float gate = smoothstepf(threshold, threshold + 0.75f, m);
            mag = m * gate;
            vel = make_float2(v0.x * gate, v0.y * gate);
        }
    }

    sMag[tid] = mag;
    sVel[tid] = vel;
    __syncthreads();

    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride && sMag[tid + stride] > sMag[tid]) {
            sMag[tid] = sMag[tid + stride];
            sVel[tid] = sVel[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0 && tx < tileW && ty < tileH) {
        tileMax[ty * tileW + tx] = sVel[0];
    }
}

__global__ void neighborMaxVelKernel(
    int tileW, int tileH, const float2* tileMax, float2* neighborMax)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= tileW || ty >= tileH) return;

    float bestMag = 0.0f;
    float2 bestVel = make_float2(0.0f, 0.0f);

    for (int dy = -1; dy <= 1; ++dy) {
        int ny = clampi(ty + dy, 0, tileH - 1);
        for (int dx = -1; dx <= 1; ++dx) {
            int nx = clampi(tx + dx, 0, tileW - 1);
            float2 v = tileMax[ny * tileW + nx];
            float m = hypotf(v.x, v.y);
            if (m > bestMag) {
                bestMag = m;
                bestVel = v;
            }
        }
    }

    neighborMax[ty * tileW + tx] = bestVel;
}

__global__ void synthesizeShutterBlurKernel(
    int w, int h, int tileW, int tileH, int tileSize,
    const float* srcRGBA,
    const uint2* flowFwd, int fwdRev, // (t->t+1)
    const uint2* flowBwd, int bwdRev, // (t->t-1)
    const uchar2* conf2,
    const float2* neighborMax,
    const float* matteTex,
    int hasMatte, int matteMode,
    float shutterAngle, float shutterPhase, int samples, int autoSamples,
    float sensitivity, float threshold,
    float highlightBoost,
    float* dstRGBA)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    const float4* src4 = (const float4*)srcRGBA;
    float4* dst4 = (float4*)dstRGBA;
    float4 centerP = __ldg(&src4[idx]);
    float mCenter = (hasMatte && matteTex) ? __ldg(&matteTex[idx]) : 1.0f;

    if (matteMode == 2 && hasMatte && mCenter <= 0.001f) {
        dst4[idx] = centerP;
        return;
    }

    float2 vf = flowFwd ? readFlowVec(flowFwd, idx, fwdRev) : make_float2(0.0f, 0.0f);
    float2 vb = flowBwd ? readFlowVec(flowBwd, idx, bwdRev) : make_float2(0.0f, 0.0f);
    if (!flowFwd && flowBwd) { vf = make_float2(-vb.x, -vb.y); }
    if (!flowBwd && flowFwd) { vb = make_float2(-vf.x, -vf.y); }

    vf.x *= sensitivity; vf.y *= sensitivity;
    vb.x *= sensitivity; vb.y *= sensitivity;

    float2 v0 = make_float2(0.5f * (vf.x - vb.x), 0.5f * (vf.y - vb.y));
    float2 a0 = make_float2(vf.x + vb.x, vf.y + vb.y);

    float centerSpeed = hypotf(v0.x, v0.y);
    float centerGate = smoothstepf(threshold, threshold + 0.75f, centerSpeed);
    v0.x *= centerGate; v0.y *= centerGate;
    a0.x *= centerGate; a0.y *= centerGate;
    centerSpeed *= centerGate;

    int tx = min(tileW - 1, x / tileSize);
    int ty = min(tileH - 1, y / tileSize);
    float2 vDom = __ldg(&neighborMax[ty * tileW + tx]);
    float domSpeed = hypotf(vDom.x, vDom.y);

    float shutterSpan = shutterAngle / 360.0f;
    float halfSpan = 0.5f * shutterSpan;
    float phaseCenter = shutterPhase / 360.0f;

    if (fmaxf(centerSpeed, domSpeed) * shutterSpan < 0.25f) {
        dst4[idx] = centerP;
        return;
    }

    // Interleaved hash jitter for banding-free temporal integration
    float jRaw = (float)(((x * 73856093) ^ (y * 19349663)) & 0xFFFF) * (1.0f / 65536.0f);
    float jitter = jRaw - 0.5f;

    float3 accumRGB = make_float3(0.0f, 0.0f, 0.0f);
    float accumA = 0.0f;
    float accumAlphaW = 0.0f;
    float accumW = 0.0f;

    float fgCenter = (hasMatte && matteMode == 1) ? mCenter : (centerSpeed > 0.35f * domSpeed ? 1.0f : 0.0f);
    const bool useDom = (domSpeed > centerSpeed + 0.75f) && (fgCenter < 0.5f);
    const bool matteIso = (hasMatte && matteMode == 1);
    int effSamples = samples;
    if (autoSamples) {
        int minS = min(samples, 8);
        int adaptive = (int)ceilf(domSpeed * shutterSpan * 1.25f);
        effSamples = clampi(adaptive, minS, samples);
    }
    const float invSamples = 1.0f / (float)effSamples;
    const float jOffset = 0.5f + 0.35f * jitter;

    for (int i = 0; i < effSamples; ++i) {
        float u = ((float)i + jOffset) * invSamples;
        float tau = phaseCenter + (u - 0.5f) * shutterSpan;

        float2 disp;
        if (useDom) {
            disp = make_float2(vDom.x * tau, vDom.y * tau);
        } else {
            float halfTau2 = 0.5f * tau * tau;
            disp = make_float2(v0.x * tau + a0.x * halfTau2,
                               v0.y * tau + a0.y * halfTau2);
        }

        float sx = (float)x + disp.x;
        float sy = (float)y + disp.y;

        int isx = clampi((int)roundf(sx), 0, w - 1);
        int isy = clampi((int)roundf(sy), 0, h - 1);
        int sIdx = isy * w + isx;

        float wSample = 1.0f;

        if (matteIso) {
            float sMatte = matteTex ? __ldg(&matteTex[sIdx]) : 1.0f;
            if (mCenter >= 0.5f) {
                if (sMatte < 0.5f) {
                    wSample = 0.0f;
                }
            } else if (sMatte >= 0.5f) {
                float2 svf = flowFwd ? readFlowVec(flowFwd, sIdx, fwdRev) : make_float2(0.0f, 0.0f);
                float2 svb = flowBwd ? readFlowVec(flowBwd, sIdx, bwdRev) : make_float2(0.0f, 0.0f);
                if (!flowFwd && flowBwd) { svf = make_float2(-svb.x, -svb.y); }
                if (!flowBwd && flowFwd) { svb = make_float2(-svf.x, -svf.y); }
                float sSpeed = hypotf(0.5f * (svf.x - svb.x) * sensitivity, 0.5f * (svf.y - svb.y) * sensitivity);
                float distToSample = hypotf(disp.x, disp.y);
                float reach = sSpeed * halfSpan + 1.0f;
                wSample = (distToSample <= reach) ? 1.0f : 0.0f;
            }
        } else if (centerSpeed > 1.5f || useDom) {
            float2 svf = flowFwd ? readFlowVec(flowFwd, sIdx, fwdRev) : make_float2(0.0f, 0.0f);
            float2 svb = flowBwd ? readFlowVec(flowBwd, sIdx, bwdRev) : make_float2(0.0f, 0.0f);
            if (!flowFwd && flowBwd) { svf = make_float2(-svb.x, -svb.y); }
            if (!flowBwd && flowFwd) { svb = make_float2(-svf.x, -svf.y); }
            float sSpeed = hypotf(0.5f * (svf.x - svb.x) * sensitivity, 0.5f * (svf.y - svb.y) * sensitivity);
            if (centerSpeed > 1.5f && sSpeed < 0.35f * centerSpeed) {
                wSample = 0.05f;
            } else if (useDom) {
                float distToSample = hypotf(disp.x, disp.y);
                float reach = sSpeed * halfSpan + 1.0f;
                wSample = (sSpeed > 0.4f * domSpeed && distToSample <= reach) ? 1.0f : 0.0f;
            }
        }

        if (wSample <= 0.0f) continue;

        float4 pix = sampleRGBA_Bilinear(srcRGBA, w, h, sx, sy);
        pix.x = dwiToLinearChannel(pix.x);
        pix.y = dwiToLinearChannel(pix.y);
        pix.z = dwiToLinearChannel(pix.z);

        float hdrW = 1.0f;
        if (highlightBoost > 0.0f) {
            float luma = dwgLuminance(pix.x, pix.y, pix.z);
            if (luma > 0.8f) {
                float ex = luma - 0.8f;
                hdrW += highlightBoost * ex * sqrtf(ex);
            }
        }
        float totW = wSample * hdrW;

        accumRGB.x += totW * pix.x;
        accumRGB.y += totW * pix.y;
        accumRGB.z += totW * pix.z;
        accumA += wSample * pix.w;
        accumAlphaW += wSample;
        accumW += totW;
    }

    if (accumW < 1e-6f) {
        dst4[idx] = centerP;
        return;
    }

    float3 outRGB = make_float3(accumRGB.x / accumW, accumRGB.y / accumW, accumRGB.z / accumW);
    float outA = clampf(accumA / fmaxf(1e-6f, accumAlphaW), 0.0f, 1.0f);

    outRGB.x = linearToDwiChannel(outRGB.x);
    outRGB.y = linearToDwiChannel(outRGB.y);
    outRGB.z = linearToDwiChannel(outRGB.z);

    if (matteMode == 2 && hasMatte) {
        outRGB.x = centerP.x * (1.0f - mCenter) + outRGB.x * mCenter;
        outRGB.y = centerP.y * (1.0f - mCenter) + outRGB.y * mCenter;
        outRGB.z = centerP.z * (1.0f - mCenter) + outRGB.z * mCenter;
    }

    dst4[idx] = make_float4(outRGB.x, outRGB.y, outRGB.z, outA);
}

struct TrailPairBatch {
    unsigned long long ptrs[24];
    float scales[24];
    int count;
    int pad;
};

__device__ __forceinline__ void flushCellCorner(
    int ox, int oy, int w, int h, int blendMode,
    int useShared, int blockMinX, int blockMinY, int boxW, int boxH,
    float* sR, float* sG, float* sB, float* sW, float* dstAccum,
    float cr, float cg, float cb, float cw)
{
    if (ox < 0 || ox >= w || oy < 0 || oy >= h) return;
    if (blendMode == 1) {
        if (cr <= 0.0f && cg <= 0.0f && cb <= 0.0f) return;
        if (useShared) {
            int lx = ox - blockMinX;
            int ly = oy - blockMinY;
            if ((unsigned int)lx < (unsigned int)boxW && (unsigned int)ly < (unsigned int)boxH) {
                int sidx = ly * boxW + lx;
                atomicMaxPosFloat(&sR[sidx], cr);
                atomicMaxPosFloat(&sG[sidx], cg);
                atomicMaxPosFloat(&sB[sidx], cb);
            }
        } else {
            int gidx = (oy * w + ox) * 4;
            atomicMaxPosFloat(&dstAccum[gidx + 0], cr);
            atomicMaxPosFloat(&dstAccum[gidx + 1], cg);
            atomicMaxPosFloat(&dstAccum[gidx + 2], cb);
        }
    } else {
        if (cw <= 0.0f) return;
        if (useShared) {
            int lx = ox - blockMinX;
            int ly = oy - blockMinY;
            if ((unsigned int)lx < (unsigned int)boxW && (unsigned int)ly < (unsigned int)boxH) {
                int sidx = ly * boxW + lx;
                atomicAdd(&sR[sidx], cr);
                atomicAdd(&sG[sidx], cg);
                atomicAdd(&sB[sidx], cb);
                atomicAdd(&sW[sidx], cw);
            }
        } else {
            int gidx = (oy * w + ox) * 4;
            atomicAdd(&dstAccum[gidx + 0], cr);
            atomicAdd(&dstAccum[gidx + 1], cg);
            atomicAdd(&dstAccum[gidx + 2], cb);
            atomicAdd(&dstAccum[gidx + 3], cw);
        }
    }
}

__device__ __forceinline__ void flushCell4(
    int cx, int cy, int w, int h, int blendMode,
    int useShared, int blockMinX, int blockMinY, int boxW, int boxH,
    float* sR, float* sG, float* sB, float* sW, float* dstAccum,
    float c00r, float c00g, float c00b, float c00w,
    float c10r, float c10g, float c10b, float c10w,
    float c01r, float c01g, float c01b, float c01w,
    float c11r, float c11g, float c11b, float c11w)
{
    flushCellCorner(cx,     cy,     w, h, blendMode, useShared, blockMinX, blockMinY, boxW, boxH, sR, sG, sB, sW, dstAccum, c00r, c00g, c00b, c00w);
    flushCellCorner(cx + 1, cy,     w, h, blendMode, useShared, blockMinX, blockMinY, boxW, boxH, sR, sG, sB, sW, dstAccum, c10r, c10g, c10b, c10w);
    flushCellCorner(cx,     cy + 1, w, h, blendMode, useShared, blockMinX, blockMinY, boxW, boxH, sR, sG, sB, sW, dstAccum, c01r, c01g, c01b, c01w);
    flushCellCorner(cx + 1, cy + 1, w, h, blendMode, useShared, blockMinX, blockMinY, boxW, boxH, sR, sG, sB, sW, dstAccum, c11r, c11g, c11b, c11w);
}

__global__ void trailPairInitStaticKernel(
    int w, int h,
    const float* frameA,
    const uint2* flow4, int flowRev,
    const float* matteA, int matteAComp, int hasMatte, int matteMode,
    int pairAgeIndex, int subFrameSteps,
    int blendMode, float trailDecay,
    float sensitivity, float motionThreshold, float lumaThreshold,
    float highlightBoost, int addToExisting,
    float4* pairAccum, unsigned char* tileActive)
{
    int x = (int)blockIdx.x * 16 + (int)threadIdx.x;
    int y = (int)blockIdx.y * 16 + (int)threadIdx.y;

    bool isMoving = false;

    if (x < w && y < h) {
        int idx = y * w + x;
        float mVal = (hasMatte && matteMode == 2 && matteA && matteAComp > 0)
                         ? readMatteVal(matteA, idx, matteAComp)
                         : 1.0f;

        if (hasMatte && matteMode == 2 && mVal <= 0.001f) {
            if (blendMode != 1) {
                const float4* frameA4 = (const float4*)frameA;
                float4 pA = __ldg(&frameA4[idx]);
                float rA = dwiToLinearChannel(pA.x);
                float gA = dwiToLinearChannel(pA.y);
                float bA = dwiToLinearChannel(pA.z);
                float decayLambda = 0.25f + 3.2f * trailDecay;
                float wBase = (blendMode == 2 && pairAgeIndex > 0) ? expf(-decayLambda * (float)pairAgeIndex) : 1.0f;
                if (addToExisting) {
                    float4 curAcc = pairAccum[idx];
                    curAcc.x += rA * wBase;
                    curAcc.y += gA * wBase;
                    curAcc.z += bA * wBase;
                    curAcc.w += wBase;
                    pairAccum[idx] = curAcc;
                } else {
                    pairAccum[idx] = make_float4(rA * wBase, gA * wBase, bA * wBase, wBase);
                }
            } else if (!addToExisting) {
                pairAccum[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        } else {
            float speed = 0.0f;
            if (flow4 && (subFrameSteps > 1 || (blendMode == 1 && motionThreshold > 0.0f))) {
                float2 fl = readFlowVec(flow4, idx, flowRev);
                speed = hypotf(fl.x * sensitivity, fl.y * sensitivity);
            }

            float motionGate = 1.0f;
            if (blendMode == 1 && motionThreshold > 0.0f) {
                motionGate = smoothstepf(motionThreshold, motionThreshold + 0.75f, speed);
            }

            if (blendMode == 1 && motionGate <= 0.01f) {
                if (!addToExisting) {
                    pairAccum[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                }
            } else {
                const float4* frameA4 = (const float4*)frameA;
                float4 pA = __ldg(&frameA4[idx]);
                float rA = dwiToLinearChannel(pA.x);
                float gA = dwiToLinearChannel(pA.y);
                float bA = dwiToLinearChannel(pA.z);

                float lumaGate = 1.0f;
                if (blendMode == 1 && lumaThreshold > 0.0f) {
                    float luma = dwgLuminance(rA, gA, bA);
                    float lumaDwi = linearToDwiChannel(luma);
                    lumaGate = smoothstepf(lumaThreshold, lumaThreshold + 0.12f, lumaDwi);
                }

                if (blendMode == 1 && motionGate * lumaGate <= 0.01f) {
                    if (!addToExisting) {
                        pairAccum[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                } else if (subFrameSteps <= 1 || speed < 0.35f) {
                    float decayLambda = 0.25f + 3.2f * trailDecay;
                    float timeW = (blendMode == 2 && pairAgeIndex > 0) ? expf(-decayLambda * (float)pairAgeIndex) : 1.0f;
                    if (blendMode == 1) {
                        float gate = motionGate * lumaGate * timeW;
                        float vr = fmaxf(0.0f, rA * gate);
                        float vg = fmaxf(0.0f, gA * gate);
                        float vb = fmaxf(0.0f, bA * gate);
                        if (addToExisting) {
                            float4 curAcc = pairAccum[idx];
                            curAcc.x = fmaxf(curAcc.x, vr);
                            curAcc.y = fmaxf(curAcc.y, vg);
                            curAcc.z = fmaxf(curAcc.z, vb);
                            pairAccum[idx] = curAcc;
                        } else {
                            pairAccum[idx] = make_float4(vr, vg, vb, 0.0f);
                        }
                    } else {
                        float hdrW = 1.0f;
                        if (highlightBoost > 0.0f) {
                            float luma = dwgLuminance(rA, gA, bA);
                            if (luma > 0.8f) {
                                float ex = luma - 0.8f;
                                hdrW += highlightBoost * ex * sqrtf(ex);
                            }
                        }
                        float wTot = timeW * hdrW;
                        if (addToExisting) {
                            float4 curAcc = pairAccum[idx];
                            curAcc.x += rA * wTot;
                            curAcc.y += gA * wTot;
                            curAcc.z += bA * wTot;
                            curAcc.w += wTot;
                            pairAccum[idx] = curAcc;
                        } else {
                            pairAccum[idx] = make_float4(rA * wTot, gA * wTot, bA * wTot, wTot);
                        }
                    }
                } else {
                    isMoving = true;
                    if (!addToExisting) {
                        pairAccum[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    }
                }
            }
        }
    }

    int anyMoving = __syncthreads_or(isMoving ? 1 : 0);
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tileActive[blockIdx.y * gridDim.x + blockIdx.x] = anyMoving ? 1 : 0;
    }
}

__global__ void trailPairSplatMovingKernel(
    int w, int h,
    const float* frameA, const float* frameB,
    const uint2* flow4, const uchar2* conf2, int flowRev,
    const float* matteA, int matteAComp, int hasMatte, int matteMode,
    int pairAgeIndex, int subFrameSteps,
    int blendMode, float trailDecay,
    float sensitivity, float motionThreshold, float lumaThreshold,
    float highlightBoost,
    const unsigned char* tileActive,
    float* dstAccum)
{
    int blockId = (int)blockIdx.y * (int)gridDim.x + (int)blockIdx.x;
    if (!tileActive[blockId]) return;

    __shared__ float sR[1536];
    __shared__ float sG[1536];
    __shared__ float sB[1536];
    __shared__ float sW[1536];
    __shared__ int sMinX[8];
    __shared__ int sMaxX[8];
    __shared__ int sMinY[8];
    __shared__ int sMaxY[8];
    __shared__ int sBbox[4];

    const int tid = (int)threadIdx.y * 16 + (int)threadIdx.x;
    const int x = (int)blockIdx.x * 16 + (int)threadIdx.x;
    const int y = (int)blockIdx.y * 16 + (int)threadIdx.y;
    const int idx = y * w + x;

    bool isActive = false;
    float2 fl = make_float2(0.0f, 0.0f);
    float vx = 0.0f, vy = 0.0f, speed = 0.0f;
    float rA = 0.0f, gA = 0.0f, bA = 0.0f;
    float motionGate = 1.0f, lumaGate = 1.0f;
    int adaptiveSteps = 0;

    if (x < w && y < h && subFrameSteps > 1 && flow4) {
        float mVal = (hasMatte && matteMode == 2 && matteA && matteAComp > 0)
                         ? readMatteVal(matteA, idx, matteAComp)
                         : 1.0f;
        if (!(hasMatte && matteMode == 2 && mVal <= 0.001f)) {
            fl = readFlowVec(flow4, idx, flowRev);
            vx = fl.x * sensitivity;
            vy = fl.y * sensitivity;
            speed = hypotf(vx, vy);
            if (speed >= 0.35f) {
                if (blendMode == 1) {
                    motionGate = (motionThreshold > 0.0f) ? smoothstepf(motionThreshold, motionThreshold + 0.75f, speed) : 1.0f;
                    if (motionGate > 0.01f) {
                        const float4* frameA4 = (const float4*)frameA;
                        float4 pA = __ldg(&frameA4[idx]);
                        rA = dwiToLinearChannel(pA.x);
                        gA = dwiToLinearChannel(pA.y);
                        bA = dwiToLinearChannel(pA.z);
                        if (lumaThreshold > 0.0f) {
                            float luma = dwgLuminance(rA, gA, bA);
                            float lumaDwi = linearToDwiChannel(luma);
                            lumaGate = smoothstepf(lumaThreshold, lumaThreshold + 0.12f, lumaDwi);
                        }
                        if (motionGate * lumaGate > 0.01f) {
                            isActive = true;
                        }
                    }
                } else {
                    const float4* frameA4 = (const float4*)frameA;
                    float4 pA = __ldg(&frameA4[idx]);
                    rA = dwiToLinearChannel(pA.x);
                    gA = dwiToLinearChannel(pA.y);
                    bA = dwiToLinearChannel(pA.z);
                    isActive = true;
                }
            }
        }
    }

    int txMin = w, txMax = -1, tyMin = h, tyMax = -1;
    if (isActive) {
        adaptiveSteps = max(subFrameSteps, min(72, (int)ceilf(speed * 1.25f)));
        float invSteps = 1.0f / (float)adaptiveSteps;
        float alphaEnd = (float)(adaptiveSteps - 1) * invSteps;
        float endX = (float)x + alphaEnd * vx;
        float endY = (float)y + alphaEnd * vy;
        float minPx = fminf((float)x, endX) - 0.01f;
        float maxPx = fmaxf((float)x, endX) + 0.01f;
        float minPy = fminf((float)y, endY) - 0.01f;
        float maxPy = fmaxf((float)y, endY) + 0.01f;
        txMin = clampi((int)floorf(minPx), 0, w - 1);
        txMax = clampi((int)floorf(maxPx) + 1, 0, w - 1);
        tyMin = clampi((int)floorf(minPy), 0, h - 1);
        tyMax = clampi((int)floorf(maxPy) + 1, 0, h - 1);
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        txMin = min(txMin, __shfl_down_sync(0xFFFFFFFFu, txMin, offset));
        txMax = max(txMax, __shfl_down_sync(0xFFFFFFFFu, txMax, offset));
        tyMin = min(tyMin, __shfl_down_sync(0xFFFFFFFFu, tyMin, offset));
        tyMax = max(tyMax, __shfl_down_sync(0xFFFFFFFFu, tyMax, offset));
    }
    if (lane == 0) {
        sMinX[warp] = txMin; sMaxX[warp] = txMax;
        sMinY[warp] = tyMin; sMaxY[warp] = tyMax;
    }
    __syncthreads();

    if (tid < 8) {
        int vx0 = sMinX[tid], vx1 = sMaxX[tid], vy0 = sMinY[tid], vy1 = sMaxY[tid];
        for (int offset = 4; offset > 0; offset >>= 1) {
            vx0 = min(vx0, __shfl_down_sync(0xFFu, vx0, offset));
            vx1 = max(vx1, __shfl_down_sync(0xFFu, vx1, offset));
            vy0 = min(vy0, __shfl_down_sync(0xFFu, vy0, offset));
            vy1 = max(vy1, __shfl_down_sync(0xFFu, vy1, offset));
        }
        if (tid == 0) {
            sBbox[0] = vx0; sBbox[1] = vx1; sBbox[2] = vy0; sBbox[3] = vy1;
        }
    }
    __syncthreads();

    const int blockMinX = sBbox[0], blockMaxX = sBbox[1];
    const int blockMinY = sBbox[2], blockMaxY = sBbox[3];
    if (blockMinX > blockMaxX || blockMinY > blockMaxY) return;

    const int boxW = blockMaxX - blockMinX + 1;
    const int boxH = blockMaxY - blockMinY + 1;
    const int boxArea = boxW * boxH;
    const int useShared = (boxArea <= 1536) ? 1 : 0;

    if (useShared) {
        for (int i = tid; i < boxArea; i += 256) {
            sR[i] = 0.0f;
            sG[i] = 0.0f;
            sB[i] = 0.0f;
            if (blendMode != 1) {
                sW[i] = 0.0f;
            }
        }
        __syncthreads();
    }

    if (isActive) {
        float4 pB = sampleRGBA_Bilinear(frameB, w, h, (float)x + fl.x, (float)y + fl.y);
        pB.x = dwiToLinearChannel(pB.x);
        pB.y = dwiToLinearChannel(pB.y);
        pB.z = dwiToLinearChannel(pB.z);
        float conf = conf2 ? readConfVal(conf2, idx, flowRev) : 1.0f;

        float hdrW = 1.0f;
        if (blendMode != 1 && highlightBoost > 0.0f) {
            float luma = dwgLuminance(rA, gA, bA);
            if (luma > 0.8f) {
                float ex = luma - 0.8f;
                hdrW += highlightBoost * ex * sqrtf(ex);
            }
        }

        const float invSteps = 1.0f / (float)adaptiveSteps;
        const float stepVx = vx * invSteps;
        const float stepVy = vy * invSteps;
        const float confStep = conf * invSteps;
        const float dr = (pB.x - rA) * confStep;
        const float dg = (pB.y - gA) * confStep;
        const float db = (pB.z - bA) * confStep;

        const float decayLambda = 0.25f + 3.2f * trailDecay;
        float timeW = (blendMode == 2) ? expf(-decayLambda * (float)pairAgeIndex) : 1.0f;
        const float decayMul = (blendMode == 2) ? expf(-decayLambda * invSteps) : 1.0f;
        const float baseGate = motionGate * lumaGate;

        float px = (float)x, py = (float)y;
        float sr = rA, sg = gA, sb = bA;

        int curX0 = -2147483647, curY0 = -2147483647;
        float c00r = 0.0f, c00g = 0.0f, c00b = 0.0f, c00w = 0.0f;
        float c10r = 0.0f, c10g = 0.0f, c10b = 0.0f, c10w = 0.0f;
        float c01r = 0.0f, c01g = 0.0f, c01b = 0.0f, c01w = 0.0f;
        float c11r = 0.0f, c11g = 0.0f, c11b = 0.0f, c11w = 0.0f;

        for (int s = 0; s < adaptiveSteps; ++s) {
            int x0 = (int)floorf(px);
            int y0 = (int)floorf(py);
            if (x0 != curX0 || y0 != curY0) {
                if (curX0 != -2147483647) {
                    flushCell4(curX0, curY0, w, h, blendMode, useShared, blockMinX, blockMinY, boxW, boxH,
                               sR, sG, sB, sW, dstAccum,
                               c00r, c00g, c00b, c00w, c10r, c10g, c10b, c10w,
                               c01r, c01g, c01b, c01w, c11r, c11g, c11b, c11w);
                }
                curX0 = x0;
                curY0 = y0;
                c00r = c00g = c00b = c00w = 0.0f;
                c10r = c10g = c10b = c10w = 0.0f;
                c01r = c01g = c01b = c01w = 0.0f;
                c11r = c11g = c11b = c11w = 0.0f;
            }

            float fx = px - (float)x0;
            float fy = py - (float)y0;
            float omfx = 1.0f - fx;
            float omfy = 1.0f - fy;
            float bw00 = omfx * omfy;
            float bw10 = fx * omfy;
            float bw01 = omfx * fy;
            float bw11 = fx * fy;

            if (blendMode == 1) {
                float gate = baseGate * timeW;
                if (gate > 0.01f) {
                    float pr = fmaxf(0.0f, sr);
                    float pg = fmaxf(0.0f, sg);
                    float pb = fmaxf(0.0f, sb);
                    if (bw00 > 0.01f) {
                        float g00 = fminf(1.0f, bw00 * 1.35f) * gate;
                        c00r = fmaxf(c00r, pr * g00); c00g = fmaxf(c00g, pg * g00); c00b = fmaxf(c00b, pb * g00);
                    }
                    if (bw10 > 0.01f) {
                        float g10 = fminf(1.0f, bw10 * 1.35f) * gate;
                        c10r = fmaxf(c10r, pr * g10); c10g = fmaxf(c10g, pg * g10); c10b = fmaxf(c10b, pb * g10);
                    }
                    if (bw01 > 0.01f) {
                        float g01 = fminf(1.0f, bw01 * 1.35f) * gate;
                        c01r = fmaxf(c01r, pr * g01); c01g = fmaxf(c01g, pg * g01); c01b = fmaxf(c01b, pb * g01);
                    }
                    if (bw11 > 0.01f) {
                        float g11 = fminf(1.0f, bw11 * 1.35f) * gate;
                        c11r = fmaxf(c11r, pr * g11); c11g = fmaxf(c11g, pg * g11); c11b = fmaxf(c11b, pb * g11);
                    }
                }
            } else {
                float wStep = timeW * hdrW * invSteps;
                if (bw00 > 1e-4f) {
                    float wt = wStep * bw00;
                    c00r += sr * wt; c00g += sg * wt; c00b += sb * wt; c00w += wt;
                }
                if (bw10 > 1e-4f) {
                    float wt = wStep * bw10;
                    c10r += sr * wt; c10g += sg * wt; c10b += sb * wt; c10w += wt;
                }
                if (bw01 > 1e-4f) {
                    float wt = wStep * bw01;
                    c01r += sr * wt; c01g += sg * wt; c01b += sb * wt; c01w += wt;
                }
                if (bw11 > 1e-4f) {
                    float wt = wStep * bw11;
                    c11r += sr * wt; c11g += sg * wt; c11b += sb * wt; c11w += wt;
                }
            }

            px += stepVx;
            py += stepVy;
            sr += dr;
            sg += dg;
            sb += db;
            timeW *= decayMul;
        }

        if (curX0 != -2147483647) {
            flushCell4(curX0, curY0, w, h, blendMode, useShared, blockMinX, blockMinY, boxW, boxH,
                       sR, sG, sB, sW, dstAccum,
                       c00r, c00g, c00b, c00w, c10r, c10g, c10b, c10w,
                       c01r, c01g, c01b, c01w, c11r, c11g, c11b, c11w);
        }
    }

    if (useShared) {
        __syncthreads();
        for (int i = tid; i < boxArea; i += 256) {
            if (blendMode == 1) {
                float vr = sR[i];
                float vg = sG[i];
                float vb = sB[i];
                if (vr > 0.0f || vg > 0.0f || vb > 0.0f) {
                    int ly = i / boxW;
                    int lx = i - ly * boxW;
                    int gidx = ((blockMinY + ly) * w + (blockMinX + lx)) * 4;
                    atomicMaxPosFloat(&dstAccum[gidx + 0], vr);
                    atomicMaxPosFloat(&dstAccum[gidx + 1], vg);
                    atomicMaxPosFloat(&dstAccum[gidx + 2], vb);
                }
            } else {
                float vw = sW[i];
                if (vw > 0.0f) {
                    int ly = i / boxW;
                    int lx = i - ly * boxW;
                    int gidx = ((blockMinY + ly) * w + (blockMinX + lx)) * 4;
                    atomicAdd(&dstAccum[gidx + 0], sR[i]);
                    atomicAdd(&dstAccum[gidx + 1], sG[i]);
                    atomicAdd(&dstAccum[gidx + 2], sB[i]);
                    atomicAdd(&dstAccum[gidx + 3], vw);
                }
            }
        }
    }
}

__global__ void trailCompositeAndFinalizeKernel(
    int w, int h,
    const float* currRGBA, const float* currMatte, int currMatteComp,
    const float* lastRGBA, const float* lastMatte, int lastMatteComp,
    int hasLastFrame, float lastTimeW,
    TrailPairBatch batch,
    const float4* uncachedAccum, int hasUncached,
    int hasMatte, int matteMode, int blendMode,
    float motionThreshold, float lumaThreshold, float highlightBoost,
    float rearCurtainCrispness,
    float seqDecay, const float4* seqPrevLin, float4* seqOutLin,
    float* dstRGBA)
{
    int x = (int)blockIdx.x * 16 + (int)threadIdx.x;
    int y = (int)blockIdx.y * 16 + (int)threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    const float4* cur4 = (const float4*)currRGBA;
    float4 cur = __ldg(&cur4[idx]);
    float cr = dwiToLinearChannel(cur.x);
    float cg = dwiToLinearChannel(cur.y);
    float cb = dwiToLinearChannel(cur.z);

    float4 acc = (blendMode == 1)
                     ? make_float4(cr, cg, cb, 1.0f)
                     : make_float4(cr * 0.02f, cg * 0.02f, cb * 0.02f, 0.02f);

    if (hasLastFrame && lastRGBA) {
        float lastM = (hasMatte && matteMode == 2 && lastMatte && lastMatteComp > 0)
                          ? readMatteVal(lastMatte, idx, lastMatteComp)
                          : 1.0f;
        if (hasMatte && matteMode == 2 && lastM <= 0.001f) {
            if (blendMode != 1) {
                const float4* last4 = (const float4*)lastRGBA;
                float4 pL = __ldg(&last4[idx]);
                acc.x += dwiToLinearChannel(pL.x) * lastTimeW;
                acc.y += dwiToLinearChannel(pL.y) * lastTimeW;
                acc.z += dwiToLinearChannel(pL.z) * lastTimeW;
                acc.w += lastTimeW;
            }
        } else if (blendMode == 1) {
            if (motionThreshold <= 0.0f) {
                const float4* last4 = (const float4*)lastRGBA;
                float4 pL = __ldg(&last4[idx]);
                float lr = dwiToLinearChannel(pL.x);
                float lg = dwiToLinearChannel(pL.y);
                float lb = dwiToLinearChannel(pL.z);
                float lumaGate = 1.0f;
                if (lumaThreshold > 0.0f) {
                    float luma = dwgLuminance(lr, lg, lb);
                    float lumaDwi = linearToDwiChannel(luma);
                    lumaGate = smoothstepf(lumaThreshold, lumaThreshold + 0.12f, lumaDwi);
                }
                if (lumaGate > 0.01f) {
                    float vr = lr * lumaGate, vg = lg * lumaGate, vb = lb * lumaGate;
                    if (vr > 0.0f) acc.x = fmaxf(acc.x, vr);
                    if (vg > 0.0f) acc.y = fmaxf(acc.y, vg);
                    if (vb > 0.0f) acc.z = fmaxf(acc.z, vb);
                }
            }
        } else {
            const float4* last4 = (const float4*)lastRGBA;
            float4 pL = __ldg(&last4[idx]);
            float lr = dwiToLinearChannel(pL.x);
            float lg = dwiToLinearChannel(pL.y);
            float lb = dwiToLinearChannel(pL.z);
            float hdrW = 1.0f;
            if (highlightBoost > 0.0f) {
                float luma = dwgLuminance(lr, lg, lb);
                if (luma > 0.8f) {
                    float ex = luma - 0.8f;
                    hdrW += highlightBoost * ex * sqrtf(ex);
                }
            }
            float wTot = lastTimeW * hdrW;
            acc.x += lr * wTot;
            acc.y += lg * wTot;
            acc.z += lb * wTot;
            acc.w += wTot;
        }
    }

    if (blendMode == 1) {
        for (int i = 0; i < batch.count; ++i) {
            const float4* pBuf = (const float4*)batch.ptrs[i];
            float4 pv = __ldg(&pBuf[idx]);
            if (pv.x > 0.0f) acc.x = fmaxf(acc.x, pv.x);
            if (pv.y > 0.0f) acc.y = fmaxf(acc.y, pv.y);
            if (pv.z > 0.0f) acc.z = fmaxf(acc.z, pv.z);
        }
        if (hasUncached && uncachedAccum) {
            float4 uv = __ldg(&uncachedAccum[idx]);
            if (uv.x > 0.0f) acc.x = fmaxf(acc.x, uv.x);
            if (uv.y > 0.0f) acc.y = fmaxf(acc.y, uv.y);
            if (uv.z > 0.0f) acc.z = fmaxf(acc.z, uv.z);
        }
    } else {
        for (int i = 0; i < batch.count; ++i) {
            const float4* pBuf = (const float4*)batch.ptrs[i];
            float4 pv = __ldg(&pBuf[idx]);
            float sc = batch.scales[i];
            acc.x += pv.x * sc;
            acc.y += pv.y * sc;
            acc.z += pv.z * sc;
            acc.w += pv.w * sc;
        }
        if (hasUncached && uncachedAccum) {
            float4 uv = __ldg(&uncachedAccum[idx]);
            acc.x += uv.x;
            acc.y += uv.y;
            acc.z += uv.z;
            acc.w += uv.w;
        }
    }

    float or_, og_, ob_;
    if (blendMode == 1) {
        or_ = acc.x;
        og_ = acc.y;
        ob_ = acc.z;
    } else {
        float invW = 1.0f / fmaxf(1e-5f, acc.w);
        or_ = acc.x * invW;
        og_ = acc.y * invW;
        ob_ = acc.z * invW;
        if (blendMode == 2 && rearCurtainCrispness > 0.0f) {
            float crisp = clampf(rearCurtainCrispness, 0.0f, 1.0f);
            or_ = or_ * (1.0f - crisp) + cr * crisp;
            og_ = og_ * (1.0f - crisp) + cg * crisp;
            ob_ = ob_ * (1.0f - crisp) + cb * crisp;
        }
    }

    float4* dst4 = (float4*)dstRGBA;
    if (seqPrevLin || seqOutLin) {
        if (hasMatte && matteMode == 2 && currMatte && currMatteComp > 0) {
            float m = readMatteVal(currMatte, idx, currMatteComp);
            or_ = cr * (1.0f - m) + or_ * m;
            og_ = cg * (1.0f - m) + og_ * m;
            ob_ = cb * (1.0f - m) + ob_ * m;
        }
        if (seqPrevLin) {
            float4 prevV = seqPrevLin[idx];
            if (blendMode == 1) {
                or_ = fmaxf(or_, prevV.x * seqDecay);
                og_ = fmaxf(og_, prevV.y * seqDecay);
                ob_ = fmaxf(ob_, prevV.z * seqDecay);
            } else {
                float alpha = 1.0f - seqDecay * 0.65f;
                or_ = prevV.x * (1.0f - alpha) + or_ * alpha;
                og_ = prevV.y * (1.0f - alpha) + og_ * alpha;
                ob_ = prevV.z * (1.0f - alpha) + ob_ * alpha;
            }
        }
        if (seqOutLin) {
            seqOutLin[idx] = make_float4(or_, og_, ob_, cur.w);
        }
        dst4[idx] = make_float4(linearToDwiChannel(or_), linearToDwiChannel(og_), linearToDwiChannel(ob_), cur.w);
    } else {
        float dwiR = linearToDwiChannel(or_);
        float dwiG = linearToDwiChannel(og_);
        float dwiB = linearToDwiChannel(ob_);
        if (hasMatte && matteMode == 2 && currMatte && currMatteComp > 0) {
            float m = readMatteVal(currMatte, idx, currMatteComp);
            dwiR = cur.x * (1.0f - m) + dwiR * m;
            dwiG = cur.y * (1.0f - m) + dwiG * m;
            dwiB = cur.z * (1.0f - m) + dwiB * m;
        }
        dst4[idx] = make_float4(dwiR, dwiG, dwiB, cur.w);
    }
}

__device__ __forceinline__ void hsvToRgb(float h, float s, float v, float& r, float& g, float& b) {
    float c = v * s;
    float hp = fmodf(h * 6.0f, 6.0f);
    float x = c * (1.0f - fabsf(fmodf(hp, 2.0f) - 1.0f));
    float m = v - c;
    if (hp < 1.0f)      { r = c; g = x; b = 0; }
    else if (hp < 2.0f) { r = x; g = c; b = 0; }
    else if (hp < 3.0f) { r = 0; g = c; b = x; }
    else if (hp < 4.0f) { r = 0; g = x; b = c; }
    else if (hp < 5.0f) { r = x; g = 0; b = c; }
    else                { r = c; g = 0; b = x; }
    r += m; g += m; b += m;
}

__global__ void renderDiagnosticViewKernel(
    int w, int h, int viewMode,
    const float* srcRGBA, const uint2* flow4, const uchar2* conf2, int flowRev,
    const float* matteTex, int hasMatte,
    float sensitivity, float motionThresh, float lumaThresh,
    float* dstRGBA)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int idx = y * w + x;
    const float4* src4 = (const float4*)srcRGBA;
    float4* dst4 = (float4*)dstRGBA;
    float4 p = __ldg(&src4[idx]);
    float2 f = flow4 ? readFlowVec(flow4, idx, flowRev) : make_float2(0.0f, 0.0f);
    float conf = conf2 ? clampf(readConfVal(conf2, idx, flowRev), 0.0f, 1.0f) : 1.0f;

    float vx = f.x * sensitivity;
    float vy = f.y * sensitivity;
    float mag = hypotf(vx, vy);

    if (viewMode == 1) {
        float angle = atan2f(vy, vx);
        float hue = (angle + 3.14159265f) / (2.0f * 3.14159265f);
        float val = clampf(mag / 24.0f, 0.0f, 1.0f);
        float r, g, b;
        hsvToRgb(hue, 0.95f, val, r, g, b);
        dst4[idx] = make_float4(r, g, b, p.w);
    } else if (viewMode == 2) {
        dst4[idx] = make_float4(1.0f - conf, conf, 0.15f, p.w);
    } else {
        float luma = dwgLuminance(p.x, p.y, p.z);
        float mg = (motionThresh > 0.0f) ? smoothstepf(motionThresh, motionThresh + 0.75f, mag) : 1.0f;
        float lg = (lumaThresh > 0.0f) ? smoothstepf(lumaThresh, lumaThresh + 0.12f, luma) : 1.0f;
        float fg = (hasMatte && matteTex) ? __ldg(&matteTex[idx]) : 1.0f;
        float mask = mg * lg * fg;
        dst4[idx] = make_float4(mask, mask, mask, p.w);
    }
}

} // extern "C"
)CUDA";

template <typename Fn>
bool loadSym(HMODULE mod, Fn& dst, const char* name) {
    dst = reinterpret_cast<Fn>(GetProcAddress(mod, name));
    return dst != nullptr;
}

bool isProcessShuttingDown() {
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    if (!ntdll) return false;
    using FnRtlShutdown = BOOLEAN(NTAPI*)();
    auto fn = reinterpret_cast<FnRtlShutdown>(GetProcAddress(ntdll, "RtlDllShutdownInProgress"));
    return fn && fn() != 0;
}

} // namespace

CudaEngine& CudaEngine::instance() {
    static CudaEngine inst;
    return inst;
}

CudaEngine::~CudaEngine() {
    shutdown();
    m_alive.store(false, std::memory_order_release);
}

bool CudaEngine::initialize() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_initialized) return true;
    if (m_attemptedInit && !m_initialized) return false;
    m_attemptedInit = true;

    HMODULE cuMod = GetModuleHandleA("nvcuda.dll");
    if (!cuMod) cuMod = LoadLibraryA("nvcuda.dll");
    if (!cuMod) {
        m_lastError = "Failed to load nvcuda.dll";
        return false;
    }
    m_cudaDll = cuMod;

    bool ok = true;
    ok &= loadSym(cuMod, g_cu.cuInit, "cuInit");
    ok &= loadSym(cuMod, g_cu.cuDeviceGet, "cuDeviceGet");
    ok &= loadSym(cuMod, g_cu.cuDeviceGetName, "cuDeviceGetName");
    loadSym(cuMod, g_cu.cuDeviceGetAttribute, "cuDeviceGetAttribute");
    ok &= loadSym(cuMod, g_cu.cuCtxGetCurrent, "cuCtxGetCurrent");
    ok &= loadSym(cuMod, g_cu.cuCtxCreate, "cuCtxCreate_v2");
    loadSym(cuMod, g_cu.cuCtxDestroy, "cuCtxDestroy_v2");
    ok &= loadSym(cuMod, g_cu.cuCtxSetCurrent, "cuCtxSetCurrent");
    ok &= loadSym(cuMod, g_cu.cuModuleLoadDataEx, "cuModuleLoadDataEx");
    ok &= loadSym(cuMod, g_cu.cuModuleUnload, "cuModuleUnload");
    ok &= loadSym(cuMod, g_cu.cuModuleGetFunction, "cuModuleGetFunction");
    ok &= loadSym(cuMod, g_cu.cuMemAlloc, "cuMemAlloc_v2");
    ok &= loadSym(cuMod, g_cu.cuMemFree, "cuMemFree_v2");
    loadSym(cuMod, g_cu.cuMemGetInfo, "cuMemGetInfo_v2");
    ok &= loadSym(cuMod, g_cu.cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2");
    ok &= loadSym(cuMod, g_cu.cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2");
    ok &= loadSym(cuMod, g_cu.cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2");
    ok &= loadSym(cuMod, g_cu.cuMemsetD8Async, "cuMemsetD8Async");
    ok &= loadSym(cuMod, g_cu.cuLaunchKernel, "cuLaunchKernel");
    ok &= loadSym(cuMod, g_cu.cuStreamSynchronize, "cuStreamSynchronize");
    if (!ok) {
        m_lastError = "Missing CUDA Driver API exports in nvcuda.dll";
        return false;
    }

    if (g_cu.cuInit(0) != CUDA_SUCCESS_VAL) {
        m_lastError = "cuInit(0) failed";
        return false;
    }

    HMODULE nvrtcMod = GetModuleHandleA("nvrtc64_120_0.dll");
    if (!nvrtcMod) {
        LoadLibraryExA("C:\\Program Files\\Blackmagic Design\\DaVinci Resolve\\nvrtc-builtins64_128.dll",
                       nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        nvrtcMod = LoadLibraryExA("C:\\Program Files\\Blackmagic Design\\DaVinci Resolve\\nvrtc64_120_0.dll",
                                  nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    if (!nvrtcMod) nvrtcMod = LoadLibraryA("nvrtc64_120_0.dll");
    if (!nvrtcMod) nvrtcMod = LoadLibraryA("nvrtc64_128_0.dll");
    if (!nvrtcMod) {
        m_lastError = "Failed to load nvrtc64_120_0.dll";
        return false;
    }
    m_nvrtcDll = nvrtcMod;

    ok = true;
    ok &= loadSym(nvrtcMod, g_cu.nvrtcCreateProgram, "nvrtcCreateProgram");
    ok &= loadSym(nvrtcMod, g_cu.nvrtcDestroyProgram, "nvrtcDestroyProgram");
    ok &= loadSym(nvrtcMod, g_cu.nvrtcCompileProgram, "nvrtcCompileProgram");
    ok &= loadSym(nvrtcMod, g_cu.nvrtcGetPTXSize, "nvrtcGetPTXSize");
    ok &= loadSym(nvrtcMod, g_cu.nvrtcGetPTX, "nvrtcGetPTX");
    ok &= loadSym(nvrtcMod, g_cu.nvrtcGetProgramLogSize, "nvrtcGetProgramLogSize");
    ok &= loadSym(nvrtcMod, g_cu.nvrtcGetProgramLog, "nvrtcGetProgramLog");
    if (!ok) {
        m_lastError = "Missing NVRTC exports";
        return false;
    }

    if (!ensureContext()) {
        return false;
    }
    if (!compileKernels()) {
        return false;
    }

    m_initialized = true;

    if (g_cu.cuMemGetInfo) {
        size_t freeMem = 0, totalMem = 0;
        if (g_cu.cuMemGetInfo(&freeMem, &totalMem) == CUDA_SUCCESS_VAL && totalMem > 0) {
            constexpr size_t kMinBudget = 1536ULL * 1024ULL * 1024ULL;
            constexpr size_t kMaxBudget = 5120ULL * 1024ULL * 1024ULL;
            const size_t target = std::min(totalMem / 4, freeMem / 2);
            FlowCache::global().setMaxBytes(std::clamp(target, kMinBudget, kMaxBudget));
        }
    }
    return true;
}

bool CudaEngine::ensureContext() {
    CUdevice dev = 0;
    if (g_cu.cuDeviceGet(&dev, 0) != CUDA_SUCCESS_VAL) {
        m_lastError = "cuDeviceGet(0) failed";
        return false;
    }
    char nameBuf[256] = {};
    if (g_cu.cuDeviceGetName(nameBuf, sizeof(nameBuf), dev) == CUDA_SUCCESS_VAL) {
        m_deviceName = nameBuf;
    }

    if (g_cu.cuDeviceGetAttribute) {
        int maj = 7, min = 5;
        if (g_cu.cuDeviceGetAttribute(&maj, 75, dev) == CUDA_SUCCESS_VAL &&
            g_cu.cuDeviceGetAttribute(&min, 76, dev) == CUDA_SUCCESS_VAL) {
            m_ccMajor = maj;
            m_ccMinor = min;
        }
    }

    CUcontext ctx = nullptr;
    g_cu.cuCtxGetCurrent(&ctx);
    if (!ctx) {
        if (g_cu.cuCtxCreate(&m_ownedCtx, 0, dev) != CUDA_SUCCESS_VAL) {
            m_lastError = "cuCtxCreate_v2 failed";
            return false;
        }
    }
    return true;
}

bool CudaEngine::compileKernels() {
    int sm = m_ccMajor * 10 + m_ccMinor;
    const char* archFlag = "--gpu-architecture=compute_75";
    if (sm >= 89)      archFlag = "--gpu-architecture=compute_89";
    else if (sm >= 86) archFlag = "--gpu-architecture=compute_86";
    else if (sm >= 80) archFlag = "--gpu-architecture=compute_80";

    nvrtcProgram prog = nullptr;
    if (g_cu.nvrtcCreateProgram(&prog, kEmbeddedCudaSource, "bmb_kernels.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS_VAL) {
        m_lastError = "nvrtcCreateProgram failed";
        return false;
    }

    const char* opts[] = {archFlag, "--use_fast_math"};
    nvrtcResult compRes = g_cu.nvrtcCompileProgram(prog, 2, opts);
    if (compRes != NVRTC_SUCCESS_VAL && std::strcmp(archFlag, "--gpu-architecture=compute_75") != 0) {
        g_cu.nvrtcDestroyProgram(&prog);
        prog = nullptr;
        if (g_cu.nvrtcCreateProgram(&prog, kEmbeddedCudaSource, "bmb_kernels.cu", 0, nullptr, nullptr) != NVRTC_SUCCESS_VAL) {
            m_lastError = "nvrtcCreateProgram fallback failed";
            return false;
        }
        const char* fallbackOpts[] = {"--gpu-architecture=compute_75", "--use_fast_math"};
        compRes = g_cu.nvrtcCompileProgram(prog, 2, fallbackOpts);
    }

    if (compRes != NVRTC_SUCCESS_VAL) {
        size_t logSize = 0;
        g_cu.nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        if (logSize > 1) g_cu.nvrtcGetProgramLog(prog, &log[0]);
        m_lastError = "NVRTC compile failed: " + log;
        g_cu.nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptxSize = 0;
    g_cu.nvrtcGetPTXSize(prog, &ptxSize);
    std::vector<char> ptx(ptxSize);
    g_cu.nvrtcGetPTX(prog, ptx.data());
    g_cu.nvrtcDestroyProgram(&prog);

    if (g_cu.cuModuleLoadDataEx(&m_module, ptx.data(), 0, nullptr, nullptr) != CUDA_SUCCESS_VAL) {
        m_lastError = "cuModuleLoadDataEx failed";
        return false;
    }

    bool ok = true;
    ok &= (g_cu.cuModuleGetFunction(&m_fnExtractLumaMatte, m_module, "extractLumaAndMatteKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnDownsample2x, m_module, "downsample2xKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnInitZeroFlow, m_module, "initZeroFlowKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnUpsampleFlow2x, m_module, "upsampleFlow2xKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnPatchMatchLK, m_module, "patchMatchLKFlowKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnCorrelationFlow, m_module, "correlationFlowKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnEmbeddedResidualRefine, m_module, "embeddedResidualRefineKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnJointBilateralFilter, m_module, "jointBilateralFlowFilterKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnJointBilateralUpsample, m_module, "jointBilateralUpsampleKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnPackBidirectional, m_module, "packBidirectionalKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnTileMaxVel, m_module, "tileMaxVelKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnNeighborMaxVel, m_module, "neighborMaxVelKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnShutterBlur, m_module, "synthesizeShutterBlurKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnTrailPairInitStatic, m_module, "trailPairInitStaticKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnTrailPairSplatMoving, m_module, "trailPairSplatMovingKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnTrailCompositeAndFinalize, m_module, "trailCompositeAndFinalizeKernel") == CUDA_SUCCESS_VAL);
    ok &= (g_cu.cuModuleGetFunction(&m_fnRenderDiagnostic, m_module, "renderDiagnosticViewKernel") == CUDA_SUCCESS_VAL);

    if (!ok) {
        m_lastError = "cuModuleGetFunction failed for one or more kernels";
        return false;
    }
    return true;
}

uint64_t CudaEngine::allocDevice(size_t bytes) {
    if (!initialize() || bytes == 0) return 0;
    CUdeviceptr ptr = 0;
    if (g_cu.cuMemAlloc(&ptr, bytes) != CUDA_SUCCESS_VAL) return 0;
    return ptr;
}

void CudaEngine::freeDevice(uint64_t dptr) {
    if (dptr && m_alive.load(std::memory_order_acquire) && m_initialized &&
        !isProcessShuttingDown() && g_cu.cuMemFree) {
        g_cu.cuMemFree(dptr);
    }
}

uint64_t CudaEngine::acquireScratch(size_t bytes, void* stream) {
    if (bytes == 0) return 0;
    const uintptr_t skey = reinterpret_cast<uintptr_t>(stream);
    {
        std::lock_guard<std::mutex> lock(m_scratchMutex);
        auto sit = m_scratchPool.find(skey);
        if (sit != m_scratchPool.end()) {
            auto bit = sit->second.find(bytes);
            if (bit != sit->second.end() && !bit->second.empty()) {
                uint64_t ptr = bit->second.back();
                bit->second.pop_back();
                m_scratchPoolBytes -= bytes;
                return ptr;
            }
        }
    }
    return allocDevice(bytes);
}

void CudaEngine::releaseScratch(uint64_t dptr, size_t bytes, void* stream) {
    if (!dptr) return;
    if (!m_alive.load(std::memory_order_acquire) || !m_initialized || isProcessShuttingDown()) {
        return;
    }
    const uintptr_t skey = reinterpret_cast<uintptr_t>(stream);
    {
        std::lock_guard<std::mutex> lock(m_scratchMutex);
        if (m_scratchPoolBytes + bytes <= kMaxScratchPoolBytes) {
            m_scratchPool[skey][bytes].push_back(dptr);
            m_scratchPoolBytes += bytes;
            return;
        }
    }
    freeDevice(dptr);
}

void CudaEngine::clearScratchPool() {
    decltype(m_scratchPool) localPool;
    {
        std::lock_guard<std::mutex> lock(m_scratchMutex);
        localPool.swap(m_scratchPool);
        m_scratchPoolBytes = 0;
    }
    if (!isProcessShuttingDown() && g_cu.cuMemFree && m_initialized) {
        for (const auto& skv : localPool) {
            for (const auto& bkv : skv.second) {
                for (uint64_t ptr : bkv.second) {
                    if (ptr) g_cu.cuMemFree(ptr);
                }
            }
        }
    }
}

bool CudaEngine::uploadToDevice(uint64_t dstDev, const void* srcHost, size_t bytes, void* stream) {
    if (!m_initialized || !dstDev || !srcHost) return false;
    return g_cu.cuMemcpyHtoDAsync(dstDev, srcHost, bytes, stream) == CUDA_SUCCESS_VAL;
}

bool CudaEngine::downloadFromDevice(void* dstHost, uint64_t srcDev, size_t bytes, void* stream) {
    if (!m_initialized || !dstHost || !srcDev) return false;
    return g_cu.cuMemcpyDtoHAsync(dstHost, srcDev, bytes, stream) == CUDA_SUCCESS_VAL;
}

bool CudaEngine::copyDeviceToDevice(uint64_t dstDev, uint64_t srcDev, size_t bytes, void* stream) {
    if (!m_initialized || !dstDev || !srcDev) return false;
    return g_cu.cuMemcpyDtoDAsync(dstDev, srcDev, bytes, stream) == CUDA_SUCCESS_VAL;
}

bool CudaEngine::memsetDevice(uint64_t dstDev, unsigned char val, size_t bytes, void* stream) {
    if (!m_initialized || !dstDev) return false;
    return g_cu.cuMemsetD8Async(dstDev, val, bytes, stream) == CUDA_SUCCESS_VAL;
}

bool CudaEngine::syncStream(void* stream) {
    if (!m_initialized) return false;
    return g_cu.cuStreamSynchronize(stream) == CUDA_SUCCESS_VAL;
}

static inline void grid2D(int w, int h, unsigned int bx, unsigned int by, unsigned int& gx, unsigned int& gy) {
    gx = static_cast<unsigned int>((w + bx - 1) / bx);
    gy = static_cast<unsigned int>((h + by - 1) / by);
}

bool CudaEngine::buildPyramid(
    uint64_t devRGBA, uint64_t devMatte, int matteComp, bool needMatte,
    int width, int height, PyramidSet& outPyr, void* stream)
{
    outPyr = PyramidSet{};
    outPyr.lw[0] = width;
    outPyr.lh[0] = height;
    int numLevels = 1;
    for (int i = 1; i < PyramidSet::kMaxLevels; ++i) {
        int nw = outPyr.lw[i - 1] / 2;
        int nh = outPyr.lh[i - 1] / 2;
        if (nw < 16 || nh < 16) break;
        outPyr.lw[i] = nw;
        outPyr.lh[i] = nh;
        ++numLevels;
    }
    outPyr.numLevels = numLevels;
    outPyr.hasMatte = needMatte && (devMatte != 0 && matteComp > 0);

    for (int i = 0; i < numLevels; ++i) {
        size_t sz = static_cast<size_t>(outPyr.lw[i]) * outPyr.lh[i] * sizeof(float);
        outPyr.luma[i] = acquireScratch(sz, stream);
        if (!outPyr.luma[i]) {
            releasePyramid(outPyr, stream);
            return false;
        }
        if (outPyr.hasMatte) {
            outPyr.matte[i] = acquireScratch(sz, stream);
            if (!outPyr.matte[i]) {
                releasePyramid(outPyr, stream);
                return false;
            }
        }
    }

    const unsigned int bx = 16, by = 16;
    unsigned int gx = 0, gy = 0;
    grid2D(width, height, bx, by, gx, gy);

    uint64_t outM0 = outPyr.hasMatte ? outPyr.matte[0] : 0;
    void* params0[] = {&width, &height, &devRGBA, &devMatte, &matteComp, &outPyr.luma[0], &outM0};
    g_cu.cuLaunchKernel(m_fnExtractLumaMatte, gx, gy, 1, bx, by, 1, 0, stream, params0, nullptr);

    for (int i = 1; i < numLevels; ++i) {
        unsigned int lgx = 0, lgy = 0;
        grid2D(outPyr.lw[i], outPyr.lh[i], bx, by, lgx, lgy);
        void* pL[] = {&outPyr.lw[i - 1], &outPyr.lh[i - 1], &outPyr.lw[i], &outPyr.lh[i],
                      &outPyr.luma[i - 1], &outPyr.luma[i]};
        g_cu.cuLaunchKernel(m_fnDownsample2x, lgx, lgy, 1, bx, by, 1, 0, stream, pL, nullptr);
        if (outPyr.hasMatte) {
            void* pM[] = {&outPyr.lw[i - 1], &outPyr.lh[i - 1], &outPyr.lw[i], &outPyr.lh[i],
                          &outPyr.matte[i - 1], &outPyr.matte[i]};
            g_cu.cuLaunchKernel(m_fnDownsample2x, lgx, lgy, 1, bx, by, 1, 0, stream, pM, nullptr);
        }
    }
    return true;
}

void CudaEngine::releasePyramid(PyramidSet& pyr, void* stream) {
    for (int i = 0; i < pyr.numLevels; ++i) {
        size_t sz = static_cast<size_t>(pyr.lw[i]) * pyr.lh[i] * sizeof(float);
        if (pyr.luma[i]) {
            releaseScratch(pyr.luma[i], sz, stream);
            pyr.luma[i] = 0;
        }
        if (pyr.matte[i]) {
            releaseScratch(pyr.matte[i], sz, stream);
            pyr.matte[i] = 0;
        }
    }
    pyr.numLevels = 0;
    pyr.hasMatte = false;
}

bool CudaEngine::launchFlowFromPyramids(
    const PyramidSet& pyrA, const PyramidSet& pyrB,
    int width, int height,
    FlowEngineType engineType, float flowDetail, MatteMode matteMode,
    uint64_t devOutFlow2, void* stream)
{
    const int numLevels = pyrA.numLevels;
    const unsigned int bx = 16, by = 16;
    unsigned int gx = 0, gy = 0;
    grid2D(width, height, bx, by, gx, gy);

    int hasMatte = (pyrA.hasMatte && matteMode == eMatteForegroundIsolation) ? 1 : 0;

    // Proxy level target for Neural mode: ~960px proxy width for 4K/HD, full-res for <1024px
    int finestLevel = 0;
    if (engineType == eFlowNeural && numLevels > 2) {
        finestLevel = (width >= 2560) ? 2 : ((width >= 1024) ? 1 : 0);
    }

    uint64_t devPrevFlow = 0;
    size_t prevFlowBytes = 0;
    float sigmaRange = 0.03f + (1.0f - flowDetail) * 0.12f;

    for (int lvl = numLevels - 1; lvl >= finestLevel; --lvl) {
        int w = pyrA.lw[lvl];
        int h = pyrA.lh[lvl];
        size_t flowBytes = static_cast<size_t>(w) * h * sizeof(Float2);
        uint64_t devFlowIn = acquireScratch(flowBytes, stream);
        uint64_t devFlowTmp = acquireScratch(flowBytes, stream);
        if (!devFlowIn || !devFlowTmp) {
            releaseScratch(devFlowIn, flowBytes, stream);
            releaseScratch(devFlowTmp, flowBytes, stream);
            releaseScratch(devPrevFlow, prevFlowBytes, stream);
            return false;
        }

        unsigned int lgx = 0, lgy = 0;
        grid2D(w, h, bx, by, lgx, lgy);

        if (lvl == numLevels - 1) {
            void* pZero[] = {&w, &h, &devFlowIn};
            g_cu.cuLaunchKernel(m_fnInitZeroFlow, lgx, lgy, 1, bx, by, 1, 0, stream, pZero, nullptr);
        } else {
            int pw = pyrA.lw[lvl + 1], ph = pyrA.lh[lvl + 1];
            void* pUp[] = {&pw, &ph, &w, &h, &devPrevFlow, &devFlowIn};
            g_cu.cuLaunchKernel(m_fnUpsampleFlow2x, lgx, lgy, 1, bx, by, 1, 0, stream, pUp, nullptr);
            releaseScratch(devPrevFlow, prevFlowBytes, stream);
            devPrevFlow = 0;
            prevFlowBytes = 0;
        }

        uint64_t lA = pyrA.luma[lvl];
        uint64_t mA = pyrA.matte[lvl];
        uint64_t lB = pyrB.luma[lvl];

        int searchRadius = 0;
        if (lvl == numLevels - 1) {
            searchRadius = (numLevels <= 4) ? 5 : 3;
        } else if (lvl >= 2) {
            searchRadius = 1;
        }
        int lkIters = (lvl <= 1) ? 5 : 4;
        void* pPMLK[] = {&w, &h, &lA, &mA, &lB, &devFlowIn, &devFlowTmp, &searchRadius, &lkIters, &hasMatte};
        g_cu.cuLaunchKernel(m_fnPatchMatchLK, lgx, lgy, 1, bx, by, 1, 0, stream, pPMLK, nullptr);

        // Neighbor propagation cleanup pass (eliminates isolated coarse outliers via spatial consensus)
        if (lvl >= 2) {
            int zeroRad = 0;
            int cleanIters = 2;
            void* pClean[] = {&w, &h, &lA, &mA, &lB, &devFlowTmp, &devFlowIn, &zeroRad, &cleanIters, &hasMatte};
            g_cu.cuLaunchKernel(m_fnPatchMatchLK, lgx, lgy, 1, bx, by, 1, 0, stream, pClean, nullptr);
            // Swap buffers cleanly: clean result in devFlowIn -> pJBF reads devFlowIn and writes devFlowTmp
            void* pJBF[] = {&w, &h, &devFlowIn, &lA, &mA, &sigmaRange, &hasMatte, &devFlowTmp};
            g_cu.cuLaunchKernel(m_fnJointBilateralFilter, lgx, lgy, 1, bx, by, 1, 0, stream, pJBF, nullptr);
            std::swap(devFlowIn, devFlowTmp);
        } else {
            void* pJBF[] = {&w, &h, &devFlowTmp, &lA, &mA, &sigmaRange, &hasMatte, &devFlowIn};
            g_cu.cuLaunchKernel(m_fnJointBilateralFilter, lgx, lgy, 1, bx, by, 1, 0, stream, pJBF, nullptr);
        }

        if (lvl == finestLevel && engineType == eFlowNeural) {
            const bool useExtOrt = NeuralFlowEngine::instance().initialize() &&
                                   NeuralFlowEngine::instance().isUsingExternalModel();
            size_t gradBytes = useExtOrt ? (static_cast<size_t>(w) * h * 2 * sizeof(float)) : 0;
            uint64_t devGradXY = useExtOrt ? acquireScratch(gradBytes, stream) : 0;
            void* pCorr[] = {&w, &h, &lA, &mA, &lB, &devFlowIn, &devFlowTmp, &devGradXY};
            g_cu.cuLaunchKernel(m_fnCorrelationFlow, lgx, lgy, 1, bx, by, 1, 0, stream, pCorr, nullptr);

            if (useExtOrt && devGradXY) {
                syncStream(stream);
                const size_t nPix = static_cast<size_t>(w) * h;
                {
                    std::lock_guard<std::mutex> lock(m_ortBufMutex);
                    m_ortFlowBuf.resize(nPix);
                    m_ortGradBuf.resize(nPix);
                    m_ortInNchw4.resize(4 * nPix);
                    m_ortOutNchw2.resize(2 * nPix);
                    downloadFromDevice(m_ortFlowBuf.data(), devFlowTmp, nPix * sizeof(Float2), stream);
                    downloadFromDevice(m_ortGradBuf.data(), devGradXY, nPix * sizeof(Float2), stream);
                    syncStream(stream);

                    for (size_t i = 0; i < nPix; ++i) {
                        m_ortInNchw4[0 * nPix + i] = m_ortFlowBuf[i].x;
                        m_ortInNchw4[1 * nPix + i] = m_ortFlowBuf[i].y;
                        m_ortInNchw4[2 * nPix + i] = m_ortGradBuf[i].x;
                        m_ortInNchw4[3 * nPix + i] = m_ortGradBuf[i].y;
                    }
                    if (NeuralFlowEngine::instance().refineFlow(w, h, m_ortInNchw4.data(), m_ortOutNchw2.data())) {
                        for (size_t i = 0; i < nPix; ++i) {
                            m_ortFlowBuf[i].x = m_ortOutNchw2[0 * nPix + i];
                            m_ortFlowBuf[i].y = m_ortOutNchw2[1 * nPix + i];
                        }
                        uploadToDevice(devFlowTmp, m_ortFlowBuf.data(), nPix * sizeof(Float2), stream);
                    }
                }
                releaseScratch(devGradXY, gradBytes, stream);

                void* pJBF2[] = {&w, &h, &devFlowTmp, &lA, &mA, &sigmaRange, &hasMatte, &devFlowIn};
                g_cu.cuLaunchKernel(m_fnJointBilateralFilter, lgx, lgy, 1, bx, by, 1, 0, stream, pJBF2, nullptr);
                releaseScratch(devFlowTmp, flowBytes, stream);
                devPrevFlow = devFlowIn;
                prevFlowBytes = flowBytes;
            } else {
                if (devGradXY) releaseScratch(devGradXY, gradBytes, stream);
                // GPU-native 3x3 Laplacian refinement matching embedded ONNX graph (zero PCIe roundtrip)
                void* pEmb[] = {&w, &h, &devFlowTmp, &devFlowIn};
                g_cu.cuLaunchKernel(m_fnEmbeddedResidualRefine, lgx, lgy, 1, bx, by, 1, 0, stream, pEmb, nullptr);

                void* pJBF2[] = {&w, &h, &devFlowIn, &lA, &mA, &sigmaRange, &hasMatte, &devFlowTmp};
                g_cu.cuLaunchKernel(m_fnJointBilateralFilter, lgx, lgy, 1, bx, by, 1, 0, stream, pJBF2, nullptr);
                releaseScratch(devFlowIn, flowBytes, stream);
                devPrevFlow = devFlowTmp;
                prevFlowBytes = flowBytes;
            }
        } else {
            releaseScratch(devFlowTmp, flowBytes, stream);
            devPrevFlow = devFlowIn;
            prevFlowBytes = flowBytes;
        }
    }

    if (finestLevel > 0) {
        int lwF = pyrA.lw[finestLevel], lhF = pyrA.lh[finestLevel];
        uint64_t lLow = pyrA.luma[finestLevel];
        uint64_t lHigh = pyrA.luma[0];
        uint64_t mHigh = pyrA.matte[0];
        void* pUpJBU[] = {&lwF, &lhF, &width, &height, &devPrevFlow, &lLow, &lHigh, &mHigh, &sigmaRange, &hasMatte, &devOutFlow2};
        g_cu.cuLaunchKernel(m_fnJointBilateralUpsample, gx, gy, 1, bx, by, 1, 0, stream, pUpJBU, nullptr);
        releaseScratch(devPrevFlow, prevFlowBytes, stream);
    } else {
        copyDeviceToDevice(devOutFlow2, devPrevFlow, static_cast<size_t>(width) * height * sizeof(Float2), stream);
        releaseScratch(devPrevFlow, prevFlowBytes, stream);
    }

    return true;
}

namespace {

struct TrailPairBatch {
    uint64_t ptrs[24];
    float scales[24];
    int count;
    int pad;
};

inline float halfToFloat(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t fBits = 0;
    if (exp == 0) {
        if (mant == 0) {
            fBits = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            fBits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        fBits = sign | 0x7F800000u | (mant << 13);
    } else {
        fBits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &fBits, sizeof(out));
    return out;
}

inline FlowCacheKey makeFlowCacheKey(
    const void* instanceId, int fLow, int fHigh, const RenderParams& params)
{
    return FlowCacheKey{
        instanceId,
        fLow,
        fHigh,
        params.width,
        params.height,
        static_cast<int>(params.flowEngine),
        static_cast<int>(std::lround(params.flowDetail * 20.0f)),
        static_cast<int>(params.matteMode),
        true
    };
}

inline TrailSplatCacheKey makeTrailSplatCacheKey(
    const void* instanceId, int frameA, int frameB, bool frameHasMatte, const RenderParams& params)
{
    const bool isLightPainting = (params.trailBlendMode == eTrailLightPainting);
    const bool isComet = (params.trailBlendMode == eTrailCometDecay);
    TrailSplatCacheKey key;
    key.instanceId = instanceId;
    key.frameA = frameA;
    key.frameB = frameB;
    key.width = params.width;
    key.height = params.height;
    key.flowEngine = static_cast<int>(params.flowEngine);
    key.detailQuantized = static_cast<int>(std::lround(params.flowDetail * 20.0f));
    key.matteMode = static_cast<int>(params.matteMode);
    key.hasMatte = (params.matteMode != eMatteIgnore) && frameHasMatte;
    key.blendMode = static_cast<int>(params.trailBlendMode);
    key.subFrameSteps = params.subFrameSteps;
    key.sensitivityQuantized = static_cast<int>(std::lround(params.motionSensitivity * 1000.0f));
    key.motionThresholdQuantized = isLightPainting ? static_cast<int>(std::lround(params.motionThreshold * 1000.0f)) : 0;
    key.lumaThresholdQuantized = isLightPainting ? static_cast<int>(std::lround(params.lumaThreshold * 1000.0f)) : 0;
    key.trailDecayQuantized = isComet ? static_cast<int>(std::lround(params.trailDecay * 1000.0f)) : 0;
    key.highlightBoostQuantized = !isLightPainting ? static_cast<int>(std::lround(params.highlightBoost * 1000.0f)) : 0;
    return key;
}

} // namespace

CudaEngine::FlowPairResult CudaEngine::getOrComputeFlowPairCanonical(
    const void* instanceId,
    int frameAIndex, uint64_t devFrameA, uint64_t devMatteA, int matteAComp,
    int frameBIndex, uint64_t devFrameB, uint64_t devMatteB, int matteBComp,
    const RenderParams& params,
    FlowCache& cache,
    void* stream,
    PyramidSet* callerPyrA,
    PyramidSet* callerPyrB,
    bool skipCacheLookup)
{
    const bool reversed = (frameAIndex > frameBIndex);
    int fLow = reversed ? frameBIndex : frameAIndex;
    int fHigh = reversed ? frameAIndex : frameBIndex;
    uint64_t devLowRGBA = reversed ? devFrameB : devFrameA;
    uint64_t devHighRGBA = reversed ? devFrameA : devFrameB;
    uint64_t devLowMatte = reversed ? devMatteB : devMatteA;
    uint64_t devHighMatte = reversed ? devMatteA : devMatteB;
    int lowMatteComp = reversed ? matteBComp : matteAComp;
    int highMatteComp = reversed ? matteAComp : matteBComp;

    FlowCacheKey key = makeFlowCacheKey(instanceId, fLow, fHigh, params);

    if (!skipCacheLookup) {
        if (auto cached = cache.get(key)) {
            return {cached, reversed};
        }
    }

    const size_t nPix = static_cast<size_t>(params.width) * params.height;
    const size_t flow2Bytes = nPix * sizeof(Float2);
    const size_t flow4Bytes = nPix * 2 * sizeof(uint32_t); // uint2 (4x fp16)
    const size_t conf2Bytes = nPix * 2;                    // uchar2
    uint64_t devFwd2 = acquireScratch(flow2Bytes, stream);
    uint64_t devBwd2 = acquireScratch(flow2Bytes, stream);

    auto entry = std::make_shared<FlowEntry>();
    entry->width = params.width;
    entry->height = params.height;
    entry->isGpu = true;
    entry->isHalfFlow = true;
    entry->devFlow4 = acquireScratch(flow4Bytes, stream);
    entry->devConf2 = acquireScratch(conf2Bytes, stream);
    entry->byteSize = flow4Bytes + conf2Bytes;
    const uint64_t fPtr = entry->devFlow4;
    entry->gpuFreeFn = [this, fPtr, flow4Bytes, conf2Bytes, stream](uint64_t ptr) {
        this->releaseScratch(ptr, (ptr == fPtr) ? flow4Bytes : conf2Bytes, stream);
    };

    if (!devFwd2 || !devBwd2 || !entry->devFlow4 || !entry->devConf2) {
        releaseScratch(devFwd2, flow2Bytes, stream);
        releaseScratch(devBwd2, flow2Bytes, stream);
        return {nullptr, reversed};
    }

    const bool needMatte = (params.matteMode == eMatteForegroundIsolation);
    PyramidSet* callerPyrLow = reversed ? callerPyrB : callerPyrA;
    PyramidSet* callerPyrHigh = reversed ? callerPyrA : callerPyrB;

    PyramidSet localPyrLow, localPyrHigh;
    PyramidSet* usePyrLow = callerPyrLow ? callerPyrLow : &localPyrLow;
    PyramidSet* usePyrHigh = callerPyrHigh ? callerPyrHigh : &localPyrHigh;

    if (usePyrLow->numLevels == 0) {
        if (!buildPyramid(devLowRGBA, devLowMatte, lowMatteComp, needMatte, params.width, params.height, *usePyrLow, stream)) {
            releasePyramid(localPyrLow, stream);
            releasePyramid(localPyrHigh, stream);
            releaseScratch(devFwd2, flow2Bytes, stream);
            releaseScratch(devBwd2, flow2Bytes, stream);
            return {nullptr, reversed};
        }
    }
    if (usePyrHigh->numLevels == 0) {
        if (!buildPyramid(devHighRGBA, devHighMatte, highMatteComp, needMatte, params.width, params.height, *usePyrHigh, stream)) {
            releasePyramid(localPyrLow, stream);
            releasePyramid(localPyrHigh, stream);
            releaseScratch(devFwd2, flow2Bytes, stream);
            releaseScratch(devBwd2, flow2Bytes, stream);
            return {nullptr, reversed};
        }
    }

    bool ok1 = launchFlowFromPyramids(*usePyrLow, *usePyrHigh, params.width, params.height,
                                       params.flowEngine, params.flowDetail, params.matteMode,
                                       devFwd2, stream);
    bool ok2 = ok1 && launchFlowFromPyramids(*usePyrHigh, *usePyrLow, params.width, params.height,
                                              params.flowEngine, params.flowDetail, params.matteMode,
                                              devBwd2, stream);

    releasePyramid(localPyrLow, stream);
    releasePyramid(localPyrHigh, stream);

    if (!ok1 || !ok2) {
        releaseScratch(devFwd2, flow2Bytes, stream);
        releaseScratch(devBwd2, flow2Bytes, stream);
        return {nullptr, reversed};
    }

    const unsigned int bx = 16, by = 16;
    unsigned int gx = 0, gy = 0;
    grid2D(params.width, params.height, bx, by, gx, gy);

    void* pPack[] = {const_cast<int*>(&params.width), const_cast<int*>(&params.height),
                     &devFwd2, &devBwd2, &entry->devFlow4, &entry->devConf2};
    g_cu.cuLaunchKernel(m_fnPackBidirectional, gx, gy, 1, bx, by, 1, 0, stream, pPack, nullptr);

    releaseScratch(devFwd2, flow2Bytes, stream);
    releaseScratch(devBwd2, flow2Bytes, stream);

    cache.put(key, entry);
    return {entry, reversed};
}

bool CudaEngine::downloadFlowToHost(const FlowEntry& entry, std::vector<Float4>& outHostFlow, void* stream) {
    if (!entry.isGpu) {
        outHostFlow = entry.hostFlow4;
        return !outHostFlow.empty();
    }
    if (!initialize() || !entry.devFlow4 || entry.width <= 0 || entry.height <= 0) return false;
    const size_t nPix = static_cast<size_t>(entry.width) * entry.height;
    outHostFlow.resize(nPix);
    if (entry.isHalfFlow) {
        std::vector<uint32_t> packed(nPix * 2);
        if (!downloadFromDevice(packed.data(), entry.devFlow4, nPix * 2 * sizeof(uint32_t), stream)) return false;
        if (!syncStream(stream)) return false;
        for (size_t i = 0; i < nPix; ++i) {
            uint32_t p0 = packed[i * 2 + 0];
            uint32_t p1 = packed[i * 2 + 1];
            outHostFlow[i].x = halfToFloat(static_cast<uint16_t>(p0 & 0xFFFFu));
            outHostFlow[i].y = halfToFloat(static_cast<uint16_t>(p0 >> 16));
            outHostFlow[i].z = halfToFloat(static_cast<uint16_t>(p1 & 0xFFFFu));
            outHostFlow[i].w = halfToFloat(static_cast<uint16_t>(p1 >> 16));
        }
    } else {
        if (!downloadFromDevice(outHostFlow.data(), entry.devFlow4, nPix * sizeof(Float4), stream)) return false;
        if (!syncStream(stream)) return false;
    }
    return true;
}

std::shared_ptr<FlowEntry> CudaEngine::getOrComputeFlowPair(
    const void* instanceId,
    int frameAIndex, uint64_t devFrameA, uint64_t devMatteA, int matteAComp,
    int frameBIndex, uint64_t devFrameB, uint64_t devMatteB, int matteBComp,
    const RenderParams& params,
    FlowCache& cache,
    void* stream)
{
    return getOrComputeFlowPairCanonical(
               instanceId,
               frameAIndex, devFrameA, devMatteA, matteAComp,
               frameBIndex, devFrameB, devMatteB, matteBComp,
               params, cache, stream).entry;
}

bool CudaEngine::renderFrame(
    const void* instanceId,
    const std::vector<FrameInput>& frames,
    int centerIdx,
    uint64_t devOutputRGBA,
    const RenderParams& params,
    FlowCache& cache,
    void* stream)
{
    if (!initialize() || centerIdx < 0 || centerIdx >= static_cast<int>(frames.size())) {
        return false;
    }

    const int w = params.width;
    const int h = params.height;
    const size_t nPix = static_cast<size_t>(w) * h;
    const FrameInput& curr = frames[centerIdx];

    const unsigned int bx = 16, by = 16;
    unsigned int gx = 0, gy = 0;
    grid2D(w, h, bx, by, gx, gy);

    // Extract single-channel matte only when needed by Diagnostic View or Shutter Blur
    uint64_t devCurrMatte = 0;
    const size_t floatBufBytes = nPix * sizeof(float);
    int hasMatte = (curr.deviceMatte != 0 && curr.matteComponents > 0 && params.matteMode != eMatteIgnore) ? 1 : 0;
    const bool needsCurrMatteTex = (hasMatte != 0) &&
                                   ((params.viewMode != eViewFinal) ||
                                    (params.mode == eModeShutterBlur) ||
                                    (params.mode == eModeCombined));
    if (needsCurrMatteTex) {
        devCurrMatte = acquireScratch(floatBufBytes, stream);
        uint64_t tmpLuma = acquireScratch(floatBufBytes, stream);
        int comp = curr.matteComponents;
        void* pEx[] = {const_cast<int*>(&w), const_cast<int*>(&h),
                       const_cast<uint64_t*>(&curr.deviceRGBA), const_cast<uint64_t*>(&curr.deviceMatte),
                       &comp, &tmpLuma, &devCurrMatte};
        g_cu.cuLaunchKernel(m_fnExtractLumaMatte, gx, gy, 1, bx, by, 1, 0, stream, pEx, nullptr);
        releaseScratch(tmpLuma, floatBufBytes, stream);
    }

    // Compute bidirectional flow around center frame t -> t+1 and t -> t-1 only when needed
    FlowPairResult flowFwdRes;
    FlowPairResult flowBwdRes;
    const bool needsCenterFlow = (params.viewMode != eViewFinal) ||
                                 (params.mode == eModeShutterBlur) ||
                                 (params.mode == eModeCombined);

    if (needsCenterFlow) {
        PyramidSet pyrCurr;
        if (centerIdx + 1 < static_cast<int>(frames.size())) {
            const auto& nextF = frames[centerIdx + 1];
            flowFwdRes = getOrComputeFlowPairCanonical(
                instanceId,
                curr.frameIndex, curr.deviceRGBA, curr.deviceMatte, curr.matteComponents,
                nextF.frameIndex, nextF.deviceRGBA, nextF.deviceMatte, nextF.matteComponents,
                params, cache, stream, &pyrCurr, nullptr);
        }
        if (centerIdx - 1 >= 0) {
            const auto& prevF = frames[centerIdx - 1];
            flowBwdRes = getOrComputeFlowPairCanonical(
                instanceId,
                curr.frameIndex, curr.deviceRGBA, curr.deviceMatte, curr.matteComponents,
                prevF.frameIndex, prevF.deviceRGBA, prevF.deviceMatte, prevF.matteComponents,
                params, cache, stream, &pyrCurr, nullptr);
        }
        releasePyramid(pyrCurr, stream);
    }

    // Diagnostic views
    if (params.viewMode != eViewFinal) {
        int vm = static_cast<int>(params.viewMode);
        const FlowPairResult& diagFlow = flowFwdRes.entry ? flowFwdRes : flowBwdRes;
        uint64_t f4 = diagFlow.entry ? diagFlow.entry->devFlow4 : 0;
        uint64_t c2 = diagFlow.entry ? diagFlow.entry->devConf2 : 0;
        int diagRev = diagFlow.reversed ? 1 : 0;
        float sens = params.motionSensitivity;
        float mt = params.motionThreshold;
        float lt = params.lumaThreshold;
        void* pDiag[] = {const_cast<int*>(&w), const_cast<int*>(&h), &vm,
                         const_cast<uint64_t*>(&curr.deviceRGBA), &f4, &c2, &diagRev,
                         &devCurrMatte, &hasMatte,
                         &sens, &mt, &lt, &devOutputRGBA};
        g_cu.cuLaunchKernel(m_fnRenderDiagnostic, gx, gy, 1, bx, by, 1, 0, stream, pDiag, nullptr);
        if (devCurrMatte) releaseScratch(devCurrMatte, floatBufBytes, stream);
        return true;
    }

    uint64_t intermediateShutterOut = 0;
    const size_t rgbaBytes = nPix * 4 * sizeof(float);
    uint64_t shutterSrcRGBA = curr.deviceRGBA;

    // Pass 1: Natural Shutter Blur (if eModeShutterBlur or eModeCombined)
    if (params.mode == eModeShutterBlur || params.mode == eModeCombined) {
        const int tileSize = 16;
        const int tileW = (w + tileSize - 1) / tileSize;
        const int tileH = (h + tileSize - 1) / tileSize;
        const size_t tileBytes = static_cast<size_t>(tileW) * tileH * sizeof(Float2);
        uint64_t devTileMax = acquireScratch(tileBytes, stream);
        uint64_t devNeighborMax = acquireScratch(tileBytes, stream);

        unsigned int tgx = 0, tgy = 0;
        grid2D(tileW, tileH, bx, by, tgx, tgy);

        uint64_t f4Fwd = flowFwdRes.entry ? flowFwdRes.entry->devFlow4 : 0;
        int fwdRev = flowFwdRes.reversed ? 1 : 0;
        uint64_t f4Bwd = flowBwdRes.entry ? flowBwdRes.entry->devFlow4 : 0;
        int bwdRev = flowBwdRes.reversed ? 1 : 0;
        uint64_t c2Ptr = flowFwdRes.entry ? flowFwdRes.entry->devConf2 : (flowBwdRes.entry ? flowBwdRes.entry->devConf2 : 0);
        float sens = params.motionSensitivity;
        float mt = params.motionThreshold;

        // 1 block of 16x16 threads per tile for coalesced parallel max reduction
        void* pTile[] = {const_cast<int*>(&w), const_cast<int*>(&h),
                         const_cast<int*>(&tileW), const_cast<int*>(&tileH), const_cast<int*>(&tileSize),
                         &f4Fwd, &fwdRev, &f4Bwd, &bwdRev, &sens, &mt, &devTileMax};
        g_cu.cuLaunchKernel(m_fnTileMaxVel, static_cast<unsigned int>(tileW), static_cast<unsigned int>(tileH), 1,
                            16, 16, 1, 0, stream, pTile, nullptr);

        void* pNeigh[] = {const_cast<int*>(&tileW), const_cast<int*>(&tileH), &devTileMax, &devNeighborMax};
        g_cu.cuLaunchKernel(m_fnNeighborMaxVel, tgx, tgy, 1, bx, by, 1, 0, stream, pNeigh, nullptr);

        uint64_t targetBuf = devOutputRGBA;
        if (params.mode == eModeCombined) {
            intermediateShutterOut = acquireScratch(rgbaBytes, stream);
            targetBuf = intermediateShutterOut;
        }

        int mMode = static_cast<int>(params.matteMode);
        float sAngle = params.shutterAngle;
        float sPhase = params.shutterPhase;
        int sSamples = params.shutterSamples;
        int autoSamples = params.autoShutterSamples ? 1 : 0;
        float hBoost = params.highlightBoost;

        void* pBlur[] = {const_cast<int*>(&w), const_cast<int*>(&h),
                         const_cast<int*>(&tileW), const_cast<int*>(&tileH), const_cast<int*>(&tileSize),
                         const_cast<uint64_t*>(&curr.deviceRGBA),
                         &f4Fwd, &fwdRev, &f4Bwd, &bwdRev, &c2Ptr, &devNeighborMax,
                         &devCurrMatte, &hasMatte, &mMode,
                         &sAngle, &sPhase, &sSamples, &autoSamples,
                         &sens, &mt, &hBoost,
                         &targetBuf};
        g_cu.cuLaunchKernel(m_fnShutterBlur, gx, gy, 1, bx, by, 1, 0, stream, pBlur, nullptr);

        releaseScratch(devTileMax, tileBytes, stream);
        releaseScratch(devNeighborMax, tileBytes, stream);

        if (params.mode == eModeShutterBlur) {
            if (devCurrMatte) releaseScratch(devCurrMatte, floatBufBytes, stream);
            return true;
        }
        shutterSrcRGBA = intermediateShutterOut;
    }

    // Pass 2: Long-Exposure Motion Trail (eModeLongExposureTrail or eModeCombined)
    const int maxLookback = std::min(centerIdx, std::max(1, params.trailFrames - 1));
    int bMode = static_cast<int>(params.trailBlendMode);
    int mMode = static_cast<int>(params.matteMode);
    int subSteps = params.subFrameSteps;
    float decay = params.trailDecay;
    float sens = params.motionSensitivity;
    float mt = params.motionThreshold;
    float lt = params.lumaThreshold;
    float hBoost = params.highlightBoost;
    const float decayLambda = 0.25f + 3.2f * decay;

    const int numPairs = maxLookback; // k = 0 .. numPairs - 1, plus oldest endpoint at centerIdx - maxLookback
    const size_t workingSetBytes = static_cast<size_t>(numPairs) * rgbaBytes;
    const bool canCacheSplats = (numPairs > 0) && (workingSetBytes <= cache.maxBytes());
    const bool needsPairFlow = (subSteps > 1) || (params.trailBlendMode == eTrailLightPainting && mt > 0.0f);

    std::vector<std::shared_ptr<TrailSplatEntry>> pairRefs(numPairs);
    std::vector<TrailSplatCacheKey> pairKeys(numPairs);
    std::vector<bool> needCompute(numPairs, false);
    std::vector<FlowPairResult> pairFlows(numPairs);
    std::vector<bool> needFlowCompute(numPairs, false);

    // 1. Lookup cached per-pair trail splats first so misses never evict active-window hits
    for (int k = 0; k < numPairs; ++k) {
        const FrameInput& fA = frames[centerIdx - k];
        const FrameInput& fB = frames[centerIdx - k - 1];
        bool pairHasMatte = (fA.deviceMatte != 0 && fA.matteComponents > 0);
        bool cacheable = canCacheSplats && (k < 24) && !(params.mode == eModeCombined && k == 0);
        if (cacheable) {
            pairKeys[k] = makeTrailSplatCacheKey(instanceId, fA.frameIndex, fB.frameIndex, pairHasMatte, params);
            pairRefs[k] = cache.getTrailSplat(pairKeys[k]);
            if (!pairRefs[k]) {
                needCompute[k] = true;
            }
        } else {
            needCompute[k] = true;
        }
    }

    // 2. Lookup cached flows for uncached pairs before computing missing flows
    if (needsPairFlow) {
        for (int k = 0; k < numPairs; ++k) {
            if (!needCompute[k]) continue;
            const FrameInput& fA = frames[centerIdx - k];
            const FrameInput& fB = frames[centerIdx - k - 1];
            const bool rev = (fA.frameIndex > fB.frameIndex);
            const int fLow = rev ? fB.frameIndex : fA.frameIndex;
            const int fHigh = rev ? fA.frameIndex : fB.frameIndex;
            if (auto cachedFlow = cache.get(makeFlowCacheKey(instanceId, fLow, fHigh, params))) {
                pairFlows[k] = {cachedFlow, rev};
            } else {
                needFlowCompute[k] = true;
            }
        }
    }

    const size_t tileActiveBytes = static_cast<size_t>(gx) * gy;
    uint64_t devTileActive = 0;
    uint64_t devPair0 = 0;
    uint64_t devUncachedAccum = 0;

    // 3. Compute missing flows & splat missing pairs oldest-to-newest (reusing boundary pyramids)
    PyramidSet pyrCarry;
    int carryIdx = -1;

    for (int k = numPairs - 1; k >= 0; --k) {
        if (!needCompute[k]) {
            if (carryIdx != -1) {
                releasePyramid(pyrCarry, stream);
                carryIdx = -1;
            }
            continue;
        }

        const int idxA = centerIdx - k;
        const int idxB = idxA - 1;
        const FrameInput& fA = frames[idxA];
        const FrameInput& fB = frames[idxB];

        if (needsPairFlow && needFlowCompute[k]) {
            PyramidSet pyrA, pyrB;
            PyramidSet* ptrPyrB = (carryIdx == idxB) ? &pyrCarry : &pyrB;
            if (carryIdx != idxB && carryIdx != -1) {
                releasePyramid(pyrCarry, stream);
                carryIdx = -1;
            }

            pairFlows[k] = getOrComputeFlowPairCanonical(
                instanceId,
                fA.frameIndex, fA.deviceRGBA, fA.deviceMatte, fA.matteComponents,
                fB.frameIndex, fB.deviceRGBA, fB.deviceMatte, fB.matteComponents,
                params, cache, stream, &pyrA, ptrPyrB, true);

            if (carryIdx == idxB) {
                releasePyramid(pyrCarry, stream);
                carryIdx = -1;
            } else {
                releasePyramid(pyrB, stream);
            }

            if (pyrA.numLevels > 0 && k > 0 && needCompute[k - 1] && needFlowCompute[k - 1]) {
                pyrCarry = pyrA;
                carryIdx = idxA;
            } else {
                releasePyramid(pyrA, stream);
            }
        } else if (carryIdx != -1) {
            releasePyramid(pyrCarry, stream);
            carryIdx = -1;
        }

        if (!devTileActive) {
            devTileActive = acquireScratch(tileActiveBytes, stream);
        }

        uint64_t rgbaA = (k == 0) ? shutterSrcRGBA : fA.deviceRGBA;
        uint64_t rgbaB = fB.deviceRGBA;
        uint64_t f4 = pairFlows[k].entry ? pairFlows[k].entry->devFlow4 : 0;
        uint64_t c2 = pairFlows[k].entry ? pairFlows[k].entry->devConf2 : 0;
        int flowRev = pairFlows[k].reversed ? 1 : 0;

        uint64_t matteA = (hasMatte && mMode == 2 && fA.deviceMatte != 0 && fA.matteComponents > 0) ? fA.deviceMatte : 0;
        int matteAComp = matteA ? fA.matteComponents : 0;
        int pairHasMatte = matteA ? 1 : 0;

        const bool isCombinedPair0 = (params.mode == eModeCombined && k == 0);
        const bool storeAsPairEntry = canCacheSplats && !isCombinedPair0 && (k < 24);

        uint64_t targetAccum = 0;
        int pairAge = 0;
        int addToExisting = 0;

        if (isCombinedPair0) {
            devPair0 = acquireScratch(rgbaBytes, stream);
            targetAccum = devPair0;
        } else if (storeAsPairEntry) {
            auto entry = std::make_shared<TrailSplatEntry>();
            entry->width = w;
            entry->height = h;
            entry->byteSize = rgbaBytes;
            uint64_t ptr = acquireScratch(rgbaBytes, stream);
            entry->devAccum4 = ptr;
            entry->gpuFreeFn = [this, rgbaBytes, stream](uint64_t p) {
                this->releaseScratch(p, rgbaBytes, stream);
            };
            if (ptr) {
                targetAccum = ptr;
                pairRefs[k] = entry;
            }
        }

        if (!targetAccum) {
            if (!devUncachedAccum) {
                devUncachedAccum = acquireScratch(rgbaBytes, stream);
                if (devUncachedAccum) {
                    memsetDevice(devUncachedAccum, 0, rgbaBytes, stream);
                }
            }
            targetAccum = devUncachedAccum;
            pairAge = k;
            addToExisting = 1;
        }

        if (targetAccum && devTileActive) {
            void* pInit[] = {
                const_cast<int*>(&w), const_cast<int*>(&h),
                &rgbaA, &f4, &flowRev,
                &matteA, &matteAComp, &pairHasMatte, &mMode,
                &pairAge, &subSteps,
                &bMode, &decay,
                &sens, &mt, &lt,
                &hBoost, &addToExisting,
                &targetAccum, &devTileActive
            };
            g_cu.cuLaunchKernel(m_fnTrailPairInitStatic, gx, gy, 1, bx, by, 1, 0, stream, pInit, nullptr);

            if (subSteps > 1 && f4) {
                void* pSplat[] = {
                    const_cast<int*>(&w), const_cast<int*>(&h),
                    &rgbaA, &rgbaB, &f4, &c2, &flowRev,
                    &matteA, &matteAComp, &pairHasMatte, &mMode,
                    &pairAge, &subSteps,
                    &bMode, &decay,
                    &sens, &mt, &lt,
                    &hBoost, &devTileActive, &targetAccum
                };
                g_cu.cuLaunchKernel(m_fnTrailPairSplatMoving, gx, gy, 1, bx, by, 1, 0, stream, pSplat, nullptr);
            }

            if (storeAsPairEntry && pairRefs[k]) {
                cache.putTrailSplat(pairKeys[k], pairRefs[k]);
            }
        }
    }

    if (carryIdx != -1) {
        releasePyramid(pyrCarry, stream);
        carryIdx = -1;
    }

    TrailPairBatch batch{};
    batch.count = 0;
    batch.pad = 0;
    for (int k = 0; k < numPairs && batch.count < 24; ++k) {
        if (params.mode == eModeCombined && k == 0 && devPair0) {
            batch.ptrs[batch.count] = devPair0;
            batch.scales[batch.count] = 1.0f;
            ++batch.count;
        } else if (pairRefs[k] && pairRefs[k]->devAccum4) {
            batch.ptrs[batch.count] = pairRefs[k]->devAccum4;
            batch.scales[batch.count] = (bMode == 2) ? std::exp(-decayLambda * static_cast<float>(k)) : 1.0f;
            ++batch.count;
        }
    }

    const FrameInput& fLast = frames[centerIdx - maxLookback];
    uint64_t lastRGBA = (maxLookback == 0) ? shutterSrcRGBA : fLast.deviceRGBA;
    uint64_t lastMatte = (hasMatte && mMode == 2 && fLast.deviceMatte != 0 && fLast.matteComponents > 0) ? fLast.deviceMatte : 0;
    int lastMatteComp = lastMatte ? fLast.matteComponents : 0;
    int hasLastFrame = 1;
    float lastTimeW = (bMode == 2 && maxLookback > 0) ? std::exp(-decayLambda * static_cast<float>(maxLookback)) : 1.0f;

    uint64_t currMattePtr = (hasMatte && mMode == 2 && curr.deviceMatte != 0 && curr.matteComponents > 0) ? curr.deviceMatte : 0;
    int currMatteComp = currMattePtr ? curr.matteComponents : 0;
    int hasUncached = devUncachedAccum ? 1 : 0;
    float crisp = params.rearCurtainCrispness;

    float seqDecay = std::clamp(1.0f - decay * 0.5f, 0.1f, 0.96f);
    uint64_t seqPrevLin = 0;
    uint64_t seqOutLin = 0;
    std::shared_ptr<SequentialTrailState> seq;
    if (params.sequentialCache) {
        seq = cache.getOrCreateSequentialState(instanceId);
        if (!seq->isGpu || seq->width != w || seq->height != h || seq->lastFrame != curr.frameIndex - 1) {
            seq->reset();
            seq->isGpu = true;
            seq->width = w;
            seq->height = h;
            seq->devAccumRGBA = allocDevice(rgbaBytes);
            seq->gpuFreeFn = [this](uint64_t ptr) { this->freeDevice(ptr); };
            seqPrevLin = 0;
            seqOutLin = seq->devAccumRGBA;
        } else if (seq->devAccumRGBA) {
            seqPrevLin = seq->devAccumRGBA;
            seqOutLin = seq->devAccumRGBA;
        }
        seq->lastFrame = curr.frameIndex;
    }

    void* pFin[] = {
        const_cast<int*>(&w), const_cast<int*>(&h),
        &shutterSrcRGBA, &currMattePtr, &currMatteComp,
        &lastRGBA, &lastMatte, &lastMatteComp,
        &hasLastFrame, &lastTimeW,
        &batch,
        &devUncachedAccum, &hasUncached,
        &hasMatte, &mMode, &bMode,
        &mt, &lt, &hBoost,
        &crisp,
        &seqDecay, &seqPrevLin, &seqOutLin,
        &devOutputRGBA
    };
    g_cu.cuLaunchKernel(m_fnTrailCompositeAndFinalize, gx, gy, 1, bx, by, 1, 0, stream, pFin, nullptr);

    if (devTileActive) releaseScratch(devTileActive, tileActiveBytes, stream);
    if (devPair0) releaseScratch(devPair0, rgbaBytes, stream);
    if (devUncachedAccum) releaseScratch(devUncachedAccum, rgbaBytes, stream);
    if (intermediateShutterOut) releaseScratch(intermediateShutterOut, rgbaBytes, stream);
    if (devCurrMatte) releaseScratch(devCurrMatte, floatBufBytes, stream);
    return true;
}

bool CudaEngine::renderHostFrames(
    const void* instanceId,
    const std::vector<FrameInput>& hostFrames,
    int centerIdx,
    float* hostOutputRGBA,
    const RenderParams& params,
    FlowCache& cache)
{
    if (!initialize() || !hostOutputRGBA) return false;
    const size_t nPix = static_cast<size_t>(params.width) * params.height;
    const size_t rgbaBytes = nPix * 4 * sizeof(float);

    std::vector<FrameInput> gpuFrames(hostFrames.size());
    bool allocOk = true;
    for (size_t i = 0; i < hostFrames.size(); ++i) {
        gpuFrames[i].frameIndex = hostFrames[i].frameIndex;
        gpuFrames[i].deviceRGBA = allocDevice(rgbaBytes);
        if (!gpuFrames[i].deviceRGBA) {
            allocOk = false;
            break;
        }
        uploadToDevice(gpuFrames[i].deviceRGBA, hostFrames[i].hostRGBA, rgbaBytes);

        if (hostFrames[i].hostMatte && hostFrames[i].matteComponents > 0) {
            size_t mBytes = nPix * hostFrames[i].matteComponents * sizeof(float);
            gpuFrames[i].deviceMatte = allocDevice(mBytes);
            if (!gpuFrames[i].deviceMatte) {
                allocOk = false;
                break;
            }
            gpuFrames[i].matteComponents = hostFrames[i].matteComponents;
            uploadToDevice(gpuFrames[i].deviceMatte, hostFrames[i].hostMatte, mBytes);
        }
    }

    uint64_t devOut = allocOk ? allocDevice(rgbaBytes) : 0;
    bool ok = (allocOk && devOut) ? renderFrame(instanceId, gpuFrames, centerIdx, devOut, params, cache, nullptr) : false;
    if (ok) {
        downloadFromDevice(hostOutputRGBA, devOut, rgbaBytes);
        syncStream(nullptr);
    }

    freeDevice(devOut);
    for (auto& gf : gpuFrames) {
        freeDevice(gf.deviceRGBA);
        freeDevice(gf.deviceMatte);
    }
    return ok;
}

void CudaEngine::shutdown() {
    if (isProcessShuttingDown()) {
        clearScratchPool();
        std::lock_guard<std::mutex> lock(m_mutex);
        m_module = nullptr;
        m_ownedCtx = nullptr;
        m_initialized = false;
        m_attemptedInit = false;
        return;
    }

    // Release any GPU allocations held by the global cache and scratch pool before tearing down module/context.
    FlowCache::global().clear();
    clearScratchPool();

    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_module && g_cu.cuModuleUnload) {
        g_cu.cuModuleUnload(m_module);
        m_module = nullptr;
    }
    if (m_ownedCtx && g_cu.cuCtxDestroy) {
        g_cu.cuCtxDestroy(m_ownedCtx);
        m_ownedCtx = nullptr;
    }
    m_initialized = false;
    m_attemptedInit = false;
}

} // namespace bmb
