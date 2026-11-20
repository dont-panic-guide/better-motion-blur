#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace bmb {

/// Dynamic ONNX Runtime loader/session manager for external or embedded proxy optical flow refinement.
class NeuralFlowEngine {
public:
    static NeuralFlowEngine& instance();

    bool initialize(const std::string& modelPath = "");
    bool isAvailable() const { return m_initialized && m_session != nullptr; }
    bool isUsingExternalModel() const { return m_initialized && m_usingExternalModel; }
    std::string runtimeVersion() const { return m_versionString; }
    std::string lastError() const { return m_lastError; }

    // Runs ONNX residual optical flow refinement on proxy resolution [1, 4, H, W] -> [1, 2, H, W].
    // inputNchw4 must have 4 * width * height floats: (vx, vy, gradX, gradY).
    // outputNchw2 receives 2 * width * height floats: (refined_vx, refined_vy).
    bool refineFlow(int width, int height, const float* inputNchw4, float* outputNchw2);

    void shutdown();

private:
    NeuralFlowEngine() = default;
    ~NeuralFlowEngine();
    NeuralFlowEngine(const NeuralFlowEngine&) = delete;
    NeuralFlowEngine& operator=(const NeuralFlowEngine&) = delete;

    std::mutex m_mutex;
    bool m_attemptedInit = false;
    bool m_initialized = false;
    bool m_usingExternalModel = false;

    void* m_dll = nullptr;
    const void* m_apiBase = nullptr;
    const void* const* m_ortSlots = nullptr;

    void* m_env = nullptr;
    void* m_sessionOpts = nullptr;
    void* m_session = nullptr;
    void* m_cpuMemInfo = nullptr;

    std::string m_versionString;
    std::string m_lastError;
};

} // namespace bmb
