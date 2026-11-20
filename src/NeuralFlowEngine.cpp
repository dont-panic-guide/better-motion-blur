#include "NeuralFlowEngine.h"
#include "OnnxModelGenerator.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

#include <cstring>
#include <filesystem>
#include <fstream>

namespace bmb {

namespace {

struct OrtApiBaseRaw {
    const void*(__stdcall* GetApi)(uint32_t version);
    const char*(__stdcall* GetVersionString)();
};

using FnGetApiBase = const OrtApiBaseRaw*(__stdcall*)();

using FnGetErrorMessage = const char*(__stdcall*)(const void* status);
using FnCreateEnv = void*(__stdcall*)(int logLevel, const char* logid, void** out);
using FnCreateSession = void*(__stdcall*)(const void* env, const wchar_t* modelPath, const void* opts, void** out);
using FnCreateSessionFromArray = void*(__stdcall*)(const void* env, const void* modelData, size_t modelLen, const void* opts, void** out);
using FnRun = void*(__stdcall*)(void* sess, const void* runOpts, const char* const* inputNames, const void* const* inputs, size_t inputLen,
                                  const char* const* outputNames, size_t outputNamesLen, void** outputs);
using FnCreateSessionOptions = void*(__stdcall*)(void** out);
using FnCreateTensorWithDataAsOrtValue = void*(__stdcall*)(const void* info, void* pData, size_t pDataLen, const int64_t* shape, size_t shapeLen,
                                                            int elemType, void** out);
using FnGetTensorMutableData = void*(__stdcall*)(void* value, void** out);
using FnCreateCpuMemoryInfo = void*(__stdcall*)(int allocType, int memType, void** out);
using FnReleasePtr = void(__stdcall*)(void* ptr);

constexpr int kSlotGetErrorMessage = 2;
constexpr int kSlotCreateEnv = 3;
constexpr int kSlotCreateSession = 7;
constexpr int kSlotCreateSessionFromArray = 8;
constexpr int kSlotRun = 9;
constexpr int kSlotCreateSessionOptions = 10;
constexpr int kSlotCreateTensorWithDataAsOrtValue = 49;
constexpr int kSlotGetTensorMutableData = 51;
constexpr int kSlotCreateCpuMemoryInfo = 69;
constexpr int kSlotReleaseEnv = 92;
constexpr int kSlotReleaseStatus = 93;
constexpr int kSlotReleaseMemoryInfo = 94;
constexpr int kSlotReleaseSession = 95;
constexpr int kSlotReleaseValue = 96;
constexpr int kSlotReleaseSessionOptions = 100;

constexpr int kOrtLoggingLevelError = 3;
constexpr int kOrtArenaAllocator = 1;
constexpr int kOrtMemTypeDefault = 0;
constexpr int kOnnxFloat = 1;

std::string consumeStatus(const void* const* slots, void* status) {
    if (!status) return {};
    auto getMsg = reinterpret_cast<FnGetErrorMessage>(slots[kSlotGetErrorMessage]);
    auto releaseSt = reinterpret_cast<FnReleasePtr>(slots[kSlotReleaseStatus]);
    std::string msg = getMsg ? getMsg(status) : "ONNX Runtime error";
    if (releaseSt) releaseSt(status);
    return msg;
}

std::string findBundleResourcesModelPath() {
    HMODULE hm = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(&findBundleResourcesModelPath), &hm) && hm) {
        char pathBuf[MAX_PATH] = {};
        if (GetModuleFileNameA(hm, pathBuf, MAX_PATH) > 0) {
            std::filesystem::path dllPath(pathBuf);
            // Typical OFX bundle: .../BetterMotionBlur.ofx.bundle/Contents/Win64/BetterMotionBlur.ofx
            auto resCandidate = dllPath.parent_path().parent_path() / "Resources" / "neuflow_v2.onnx";
            if (std::filesystem::exists(resCandidate)) {
                return resCandidate.string();
            }
            auto localCandidate = dllPath.parent_path() / "models" / "neuflow_v2.onnx";
            if (std::filesystem::exists(localCandidate)) {
                return localCandidate.string();
            }
        }
    }
    return "";
}

bool isProcessShuttingDown() {
    HMODULE ntdll = GetModuleHandleA("ntdll.dll");
    if (!ntdll) return false;
    using FnRtlShutdown = BOOLEAN(NTAPI*)();
    auto fn = reinterpret_cast<FnRtlShutdown>(GetProcAddress(ntdll, "RtlDllShutdownInProgress"));
    return fn && fn() != 0;
}

} // namespace

NeuralFlowEngine& NeuralFlowEngine::instance() {
    static NeuralFlowEngine inst;
    return inst;
}

NeuralFlowEngine::~NeuralFlowEngine() {
    shutdown();
}

bool NeuralFlowEngine::initialize(const std::string& modelPath) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_initialized && m_session) return true;
    if (m_attemptedInit && !m_initialized) return false;
    m_attemptedInit = true;

    HMODULE ortMod = GetModuleHandleA("onnxruntime.dll");
    if (!ortMod) {
        ortMod = LoadLibraryExA("C:\\Program Files\\Blackmagic Design\\DaVinci Resolve\\onnxruntime.dll",
                                nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    }
    if (!ortMod) {
        ortMod = LoadLibraryA("onnxruntime.dll");
    }
    if (!ortMod) {
        m_lastError = "onnxruntime.dll not found";
        return false;
    }
    m_dll = ortMod;

    auto getApiBase = reinterpret_cast<FnGetApiBase>(GetProcAddress(ortMod, "OrtGetApiBase"));
    if (!getApiBase) {
        m_lastError = "OrtGetApiBase export missing";
        return false;
    }

    const OrtApiBaseRaw* base = getApiBase();
    if (!base || !base->GetApi) {
        m_lastError = "OrtApiBase invalid";
        return false;
    }
    m_apiBase = base;
    if (base->GetVersionString) {
        m_versionString = base->GetVersionString();
    }

    const void* apiTable = nullptr;
    for (uint32_t ver : {13u, 12u, 11u, 10u, 8u, 1u}) {
        apiTable = base->GetApi(ver);
        if (apiTable) break;
    }
    if (!apiTable) {
        m_lastError = "Failed to acquire OrtApi table";
        return false;
    }
    m_ortSlots = reinterpret_cast<const void* const*>(apiTable);

    auto createEnv = reinterpret_cast<FnCreateEnv>(m_ortSlots[kSlotCreateEnv]);
    auto createSessOpts = reinterpret_cast<FnCreateSessionOptions>(m_ortSlots[kSlotCreateSessionOptions]);
    auto createCpuMem = reinterpret_cast<FnCreateCpuMemoryInfo>(m_ortSlots[kSlotCreateCpuMemoryInfo]);
    auto createSessFromArray = reinterpret_cast<FnCreateSessionFromArray>(m_ortSlots[kSlotCreateSessionFromArray]);
    auto createSess = reinterpret_cast<FnCreateSession>(m_ortSlots[kSlotCreateSession]);

    void* st = createEnv(kOrtLoggingLevelError, "BetterMotionBlur", &m_env);
    if (st) {
        m_lastError = consumeStatus(m_ortSlots, st);
        return false;
    }

    st = createSessOpts(&m_sessionOpts);
    if (st) {
        m_lastError = consumeStatus(m_ortSlots, st);
        return false;
    }

    st = createCpuMem(kOrtArenaAllocator, kOrtMemTypeDefault, &m_cpuMemInfo);
    if (st) {
        m_lastError = consumeStatus(m_ortSlots, st);
        return false;
    }

    std::string resolvedPath = modelPath;
    if (resolvedPath.empty()) {
        resolvedPath = findBundleResourcesModelPath();
    }

    m_usingExternalModel = false;
    if (!resolvedPath.empty() && std::filesystem::exists(resolvedPath)) {
        std::wstring wpath = std::filesystem::path(resolvedPath).wstring();
        st = createSess(m_env, wpath.c_str(), m_sessionOpts, &m_session);
        if (st) {
            m_lastError = consumeStatus(m_ortSlots, st);
            m_session = nullptr;
        } else if (m_session) {
            m_usingExternalModel = true;
        }
    }

    if (!m_session) {
        auto embeddedBytes = buildResidualFlowRefinementOnnx();
        st = createSessFromArray(m_env, embeddedBytes.data(), embeddedBytes.size(), m_sessionOpts, &m_session);
        if (st) {
            m_lastError = consumeStatus(m_ortSlots, st);
            return false;
        }
        m_usingExternalModel = false;
    }

    m_initialized = (m_session != nullptr);
    return m_initialized;
}

bool NeuralFlowEngine::refineFlow(int width, int height, const float* inputNchw4, float* outputNchw2) {
    if (!inputNchw4 || !outputNchw2 || width <= 0 || height <= 0) return false;
    if (!initialize()) return false;

    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_session || !m_ortSlots) return false;

    auto createTensor = reinterpret_cast<FnCreateTensorWithDataAsOrtValue>(m_ortSlots[kSlotCreateTensorWithDataAsOrtValue]);
    auto runFn = reinterpret_cast<FnRun>(m_ortSlots[kSlotRun]);
    auto getData = reinterpret_cast<FnGetTensorMutableData>(m_ortSlots[kSlotGetTensorMutableData]);
    auto releaseVal = reinterpret_cast<FnReleasePtr>(m_ortSlots[kSlotReleaseValue]);

    const int64_t inShape[4] = {1, 4, static_cast<int64_t>(height), static_cast<int64_t>(width)};
    const size_t inBytes = static_cast<size_t>(4) * width * height * sizeof(float);

    void* inTensor = nullptr;
    void* st = createTensor(m_cpuMemInfo, const_cast<float*>(inputNchw4), inBytes, inShape, 4, kOnnxFloat, &inTensor);
    if (st) {
        m_lastError = consumeStatus(m_ortSlots, st);
        return false;
    }

    const char* inputNames[1] = {"flow_feat"};
    const char* outputNames[1] = {"refined_flow"};
    const void* inputPtrs[1] = {inTensor};
    void* outputTensor = nullptr;

    st = runFn(m_session, nullptr, inputNames, inputPtrs, 1, outputNames, 1, &outputTensor);
    releaseVal(inTensor);
    if (st) {
        m_lastError = consumeStatus(m_ortSlots, st);
        if (outputTensor) releaseVal(outputTensor);
        return false;
    }

    void* rawOut = nullptr;
    st = getData(outputTensor, &rawOut);
    if (st || !rawOut) {
        m_lastError = st ? consumeStatus(m_ortSlots, st) : "Null output tensor pointer";
        releaseVal(outputTensor);
        return false;
    }

    const size_t outFloats = static_cast<size_t>(2) * width * height;
    std::memcpy(outputNchw2, rawOut, outFloats * sizeof(float));
    releaseVal(outputTensor);
    return true;
}

void NeuralFlowEngine::shutdown() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (isProcessShuttingDown()) {
        // Process is terminating; avoid joining ONNX Runtime worker threads under Loader Lock.
        m_session = nullptr;
        m_sessionOpts = nullptr;
        m_cpuMemInfo = nullptr;
        m_env = nullptr;
        m_initialized = false;
        m_attemptedInit = false;
        m_usingExternalModel = false;
        return;
    }

    if (m_ortSlots) {
        auto relSess = reinterpret_cast<FnReleasePtr>(m_ortSlots[kSlotReleaseSession]);
        auto relOpts = reinterpret_cast<FnReleasePtr>(m_ortSlots[kSlotReleaseSessionOptions]);
        auto relMem = reinterpret_cast<FnReleasePtr>(m_ortSlots[kSlotReleaseMemoryInfo]);
        auto relEnv = reinterpret_cast<FnReleasePtr>(m_ortSlots[kSlotReleaseEnv]);
        if (m_session && relSess) { relSess(m_session); m_session = nullptr; }
        if (m_sessionOpts && relOpts) { relOpts(m_sessionOpts); m_sessionOpts = nullptr; }
        if (m_cpuMemInfo && relMem) { relMem(m_cpuMemInfo); m_cpuMemInfo = nullptr; }
        if (m_env && relEnv) { relEnv(m_env); m_env = nullptr; }
    }
    m_initialized = false;
    m_attemptedInit = false;
    m_usingExternalModel = false;
}

} // namespace bmb
