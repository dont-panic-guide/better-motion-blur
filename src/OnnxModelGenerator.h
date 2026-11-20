#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace bmb {

namespace pb {

inline void writeVarint(std::vector<uint8_t>& out, uint64_t val) {
    while (val >= 0x80) {
        out.push_back(static_cast<uint8_t>((val & 0x7F) | 0x80));
        val >>= 7;
    }
    out.push_back(static_cast<uint8_t>(val));
}

inline void writeTag(std::vector<uint8_t>& out, uint32_t fieldNumber, uint32_t wireType) {
    writeVarint(out, (static_cast<uint64_t>(fieldNumber) << 3) | wireType);
}

inline void writeVarintField(std::vector<uint8_t>& out, uint32_t fieldNumber, uint64_t val) {
    writeTag(out, fieldNumber, 0);
    writeVarint(out, val);
}

inline void writeBytesField(std::vector<uint8_t>& out, uint32_t fieldNumber, const void* data, size_t size) {
    writeTag(out, fieldNumber, 2);
    writeVarint(out, size);
    const auto* p = static_cast<const uint8_t*>(data);
    out.insert(out.end(), p, p + size);
}

inline void writeStringField(std::vector<uint8_t>& out, uint32_t fieldNumber, const std::string& s) {
    writeBytesField(out, fieldNumber, s.data(), s.size());
}

inline std::vector<uint8_t> makeTensorShapeDimValue(int64_t val) {
    std::vector<uint8_t> msg;
    writeVarintField(msg, 1, static_cast<uint64_t>(val)); // dim_value
    return msg;
}

inline std::vector<uint8_t> makeTensorShapeDimParam(const std::string& param) {
    std::vector<uint8_t> msg;
    writeStringField(msg, 2, param); // dim_param
    return msg;
}

inline std::vector<uint8_t> makeValueInfoFloat4D(const std::string& name, int64_t n, int64_t c, const std::string& hName, const std::string& wName) {
    std::vector<uint8_t> shape;
    auto d0 = makeTensorShapeDimValue(n);
    auto d1 = makeTensorShapeDimValue(c);
    auto d2 = makeTensorShapeDimParam(hName);
    auto d3 = makeTensorShapeDimParam(wName);
    writeBytesField(shape, 1, d0.data(), d0.size());
    writeBytesField(shape, 1, d1.data(), d1.size());
    writeBytesField(shape, 1, d2.data(), d2.size());
    writeBytesField(shape, 1, d3.data(), d3.size());

    std::vector<uint8_t> tensorType;
    writeVarintField(tensorType, 1, 1); // elem_type = FLOAT
    writeBytesField(tensorType, 2, shape.data(), shape.size());

    std::vector<uint8_t> typeProto;
    writeBytesField(typeProto, 1, tensorType.data(), tensorType.size());

    std::vector<uint8_t> vi;
    writeStringField(vi, 1, name);
    writeBytesField(vi, 2, typeProto.data(), typeProto.size());
    return vi;
}

inline std::vector<uint8_t> makeTensorFloat(const std::string& name, const std::vector<int64_t>& dims, const std::vector<float>& data) {
    std::vector<uint8_t> t;
    for (int64_t d : dims) {
        writeVarintField(t, 1, static_cast<uint64_t>(d));
    }
    writeVarintField(t, 2, 1); // data_type = FLOAT
    writeStringField(t, 8, name);
    writeBytesField(t, 9, data.data(), data.size() * sizeof(float));
    return t;
}

inline std::vector<uint8_t> makeAttrInts(const std::string& name, const std::vector<int64_t>& vals) {
    std::vector<uint8_t> a;
    writeStringField(a, 1, name);
    for (int64_t v : vals) {
        writeVarintField(a, 8, static_cast<uint64_t>(v)); // ints
    }
    writeVarintField(a, 20, 7); // type = INTS (7)
    return a;
}

inline std::vector<uint8_t> makeNode(const std::string& opType,
                                     const std::string& name,
                                     const std::vector<std::string>& inputs,
                                     const std::vector<std::string>& outputs,
                                     const std::vector<std::vector<uint8_t>>& attrs = {}) {
    std::vector<uint8_t> n;
    for (const auto& in : inputs) writeStringField(n, 1, in);
    for (const auto& out : outputs) writeStringField(n, 2, out);
    writeStringField(n, 3, name);
    writeStringField(n, 4, opType);
    for (const auto& a : attrs) writeBytesField(n, 5, a.data(), a.size());
    return n;
}

} // namespace pb

// Builds a compact, valid ONNX IR v7 (opset 11) residual flow-refinement network:
// Input:  "flow_feat" [1, 4, H, W] -> channels: (vx, vy, gradX, gradY)
// Output: "refined_flow" [1, 2, H, W] -> refined (vx, vy)
inline std::vector<uint8_t> buildResidualFlowRefinementOnnx() {
    // 1x1 identity projection W_id: [2, 4, 1, 1]
    std::vector<float> wId(2 * 4 * 1 * 1, 0.0f);
    wId[0 * 4 + 0] = 1.0f;
    wId[1 * 4 + 1] = 1.0f;

    // 3x3 directional spatial smoothing / feature extraction W1: [8, 4, 3, 3]
    std::vector<float> w1(8 * 4 * 3 * 3, 0.0f);
    std::vector<float> b1(8, 0.0f);
    const float kGauss3[9] = {
        0.0625f, 0.125f, 0.0625f,
        0.125f, -0.75f,  0.125f,
        0.0625f, 0.125f, 0.0625f
    };
    for (int k = 0; k < 9; ++k) {
        // Channels 0,1: positive Laplacian smoothing for vx, vy
        w1[(0 * 4 + 0) * 9 + k] = kGauss3[k];
        w1[(1 * 4 + 1) * 9 + k] = kGauss3[k];
        // Channels 2,3: negative branch for ReLU pair
        w1[(2 * 4 + 0) * 9 + k] = -kGauss3[k];
        w1[(3 * 4 + 1) * 9 + k] = -kGauss3[k];
        // Channels 4..7: edge-aware dampening
        w1[(4 * 4 + 2) * 9 + k] = 0.05f * kGauss3[k];
        w1[(5 * 4 + 3) * 9 + k] = 0.05f * kGauss3[k];
    }

    // 3x3 residual recombination W2: [2, 8, 3, 3]
    std::vector<float> w2(2 * 8 * 3 * 3, 0.0f);
    std::vector<float> b2(2, 0.0f);
    // Center tap (idx 4) reconstructs small spatial regularization delta (0.08 * Laplacian)
    w2[(0 * 8 + 0) * 9 + 4] = 0.08f;
    w2[(0 * 8 + 2) * 9 + 4] = -0.08f;
    w2[(1 * 8 + 1) * 9 + 4] = 0.08f;
    w2[(1 * 8 + 3) * 9 + 4] = -0.08f;

    auto tWId = pb::makeTensorFloat("W_id", {2, 4, 1, 1}, wId);
    auto tW1 = pb::makeTensorFloat("W1", {8, 4, 3, 3}, w1);
    auto tB1 = pb::makeTensorFloat("B1", {8}, b1);
    auto tW2 = pb::makeTensorFloat("W2", {2, 8, 3, 3}, w2);
    auto tB2 = pb::makeTensorFloat("B2", {2}, b2);

    auto attrPad0 = pb::makeAttrInts("pads", {0, 0, 0, 0});
    auto attrPad1 = pb::makeAttrInts("pads", {1, 1, 1, 1});
    auto attrK1 = pb::makeAttrInts("kernel_shape", {1, 1});
    auto attrK3 = pb::makeAttrInts("kernel_shape", {3, 3});

    auto nBase = pb::makeNode("Conv", "base_proj", {"flow_feat", "W_id"}, {"base_flow"}, {attrK1, attrPad0});
    auto nConv1 = pb::makeNode("Conv", "conv1", {"flow_feat", "W1", "B1"}, {"h1"}, {attrK3, attrPad1});
    auto nRelu1 = pb::makeNode("Relu", "relu1", {"h1"}, {"a1"});
    auto nConv2 = pb::makeNode("Conv", "conv2", {"a1", "W2", "B2"}, {"res_flow"}, {attrK3, attrPad1});
    auto nAdd = pb::makeNode("Add", "residual_add", {"base_flow", "res_flow"}, {"refined_flow"});

    auto viIn = pb::makeValueInfoFloat4D("flow_feat", 1, 4, "H", "W");
    auto viOut = pb::makeValueInfoFloat4D("refined_flow", 1, 2, "H", "W");

    std::vector<uint8_t> graph;
    for (const auto& node : {nBase, nConv1, nRelu1, nConv2, nAdd}) {
        pb::writeBytesField(graph, 1, node.data(), node.size());
    }
    pb::writeStringField(graph, 2, "neuflow_v2_refine");
    for (const auto& init : {tWId, tW1, tB1, tW2, tB2}) {
        pb::writeBytesField(graph, 5, init.data(), init.size());
    }
    pb::writeBytesField(graph, 11, viIn.data(), viIn.size());
    pb::writeBytesField(graph, 12, viOut.data(), viOut.size());

    std::vector<uint8_t> opset;
    pb::writeStringField(opset, 1, "");
    pb::writeVarintField(opset, 2, 11);

    std::vector<uint8_t> model;
    pb::writeVarintField(model, 1, 7); // ir_version = 7
    pb::writeStringField(model, 2, "BetterMotionBlur");
    pb::writeVarintField(model, 5, 1); // model_version = 1
    pb::writeBytesField(model, 7, graph.data(), graph.size());
    pb::writeBytesField(model, 8, opset.data(), opset.size());

    return model;
}

inline bool writeResidualFlowOnnxToFile(const std::string& path) {
    std::error_code ec;
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path(), ec);
    }
    auto bytes = buildResidualFlowRefinementOnnx();
    std::ofstream ofs(path, std::ios::binary | std::ios::trunc);
    if (!ofs.is_open()) return false;
    ofs.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return ofs.good();
}

} // namespace bmb
