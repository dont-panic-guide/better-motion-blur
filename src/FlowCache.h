#pragma once

#include "CommonTypes.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace bmb {

struct FlowCacheKey {
    const void* instanceId = nullptr;
    int frameA = 0;
    int frameB = 0;
    int width = 0;
    int height = 0;
    int engineType = 0;
    int detailQuantized = 0;
    int matteMode = 0;
    bool isGpu = true;

    bool operator==(const FlowCacheKey& other) const noexcept {
        return instanceId == other.instanceId &&
               frameA == other.frameA &&
               frameB == other.frameB &&
               width == other.width &&
               height == other.height &&
               engineType == other.engineType &&
               detailQuantized == other.detailQuantized &&
               matteMode == other.matteMode &&
               isGpu == other.isGpu;
    }
};

struct FlowCacheKeyHash {
    std::size_t operator()(const FlowCacheKey& k) const noexcept {
        std::size_t h = std::hash<const void*>{}(k.instanceId);
        auto combine = [&h](std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        combine(std::hash<int>{}(k.frameA));
        combine(std::hash<int>{}(k.frameB));
        combine(std::hash<int>{}(k.width));
        combine(std::hash<int>{}(k.height));
        combine(std::hash<int>{}(k.engineType));
        combine(std::hash<int>{}(k.detailQuantized));
        combine(std::hash<int>{}(k.matteMode));
        combine(std::hash<bool>{}(k.isGpu));
        return h;
    }
};

struct TrailSplatCacheKey {
    const void* instanceId = nullptr;
    int frameA = 0;
    int frameB = 0;
    int width = 0;
    int height = 0;
    int flowEngine = 0;
    int detailQuantized = 0;
    int matteMode = 0;
    bool hasMatte = false;
    int blendMode = 0;
    int subFrameSteps = 0;
    int sensitivityQuantized = 0;
    // Mode-canonicalized parameters (0 when unused by active blendMode)
    int motionThresholdQuantized = 0;
    int lumaThresholdQuantized = 0;
    int trailDecayQuantized = 0;
    int highlightBoostQuantized = 0;

    bool operator==(const TrailSplatCacheKey& other) const noexcept {
        return instanceId == other.instanceId &&
               frameA == other.frameA &&
               frameB == other.frameB &&
               width == other.width &&
               height == other.height &&
               flowEngine == other.flowEngine &&
               detailQuantized == other.detailQuantized &&
               matteMode == other.matteMode &&
               hasMatte == other.hasMatte &&
               blendMode == other.blendMode &&
               subFrameSteps == other.subFrameSteps &&
               sensitivityQuantized == other.sensitivityQuantized &&
               motionThresholdQuantized == other.motionThresholdQuantized &&
               lumaThresholdQuantized == other.lumaThresholdQuantized &&
               trailDecayQuantized == other.trailDecayQuantized &&
               highlightBoostQuantized == other.highlightBoostQuantized;
    }
};

struct TrailSplatCacheKeyHash {
    std::size_t operator()(const TrailSplatCacheKey& k) const noexcept {
        std::size_t h = std::hash<const void*>{}(k.instanceId);
        auto combine = [&h](std::size_t v) {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        combine(std::hash<int>{}(k.frameA));
        combine(std::hash<int>{}(k.frameB));
        combine(std::hash<int>{}(k.width));
        combine(std::hash<int>{}(k.height));
        combine(std::hash<int>{}(k.flowEngine));
        combine(std::hash<int>{}(k.detailQuantized));
        combine(std::hash<int>{}(k.matteMode));
        combine(std::hash<bool>{}(k.hasMatte));
        combine(std::hash<int>{}(k.blendMode));
        combine(std::hash<int>{}(k.subFrameSteps));
        combine(std::hash<int>{}(k.sensitivityQuantized));
        combine(std::hash<int>{}(k.motionThresholdQuantized));
        combine(std::hash<int>{}(k.lumaThresholdQuantized));
        combine(std::hash<int>{}(k.trailDecayQuantized));
        combine(std::hash<int>{}(k.highlightBoostQuantized));
        return h;
    }
};

struct FlowEntry {
    int width = 0;
    int height = 0;
    bool isGpu = true;
    bool isHalfFlow = false;

    // GPU resident buffers (CUdeviceptr cast to uint64_t).
    // When isHalfFlow == true: devFlow4 holds uint2 (4x fp16), devConf2 holds uchar2 (2x unorm8).
    uint64_t devFlow4 = 0;
    uint64_t devConf2 = 0;
    std::function<void(uint64_t)> gpuFreeFn;

    // CPU fallback buffers
    std::vector<Float4> hostFlow4;
    std::vector<Float2> hostConf2;

    std::size_t byteSize = 0;

    ~FlowEntry();
    FlowEntry() = default;
    FlowEntry(const FlowEntry&) = delete;
    FlowEntry& operator=(const FlowEntry&) = delete;
};

struct TrailSplatEntry {
    int width = 0;
    int height = 0;
    uint64_t devAccum4 = 0; // float4: (R, G, B, W) at pairAgeIndex = 0
    std::function<void(uint64_t)> gpuFreeFn;
    std::size_t byteSize = 0;

    ~TrailSplatEntry();
    TrailSplatEntry() = default;
    TrailSplatEntry(const TrailSplatEntry&) = delete;
    TrailSplatEntry& operator=(const TrailSplatEntry&) = delete;
};

struct SequentialTrailState {
    int width = 0;
    int height = 0;
    int lastFrame = -999999;
    bool isGpu = true;
    uint64_t devAccumRGBA = 0; // On GPU: stores scene-linear DWG float4 directly
    std::function<void(uint64_t)> gpuFreeFn;
    std::vector<float> hostAccumRGBA;

    ~SequentialTrailState();
    void reset();
};

class FlowCache {
public:
    explicit FlowCache(std::size_t maxBytes = 1024ULL * 1024ULL * 1024ULL);
    ~FlowCache();

    static FlowCache& global();

    void setMaxBytes(std::size_t maxBytes);
    std::size_t maxBytes() const;
    std::size_t currentBytes() const;

    // Returns nullptr on miss; increments hit/miss counters accordingly.
    std::shared_ptr<FlowEntry> get(const FlowCacheKey& key);

    // Inserts or updates a flow entry, evicting LRU items if over budget.
    void put(const FlowCacheKey& key, const std::shared_ptr<FlowEntry>& entry);

    // Per-pair Long-Exposure trail splat cache (pairAgeIndex = 0).
    // Hit increments hits(); miss returns nullptr without double-counting misses.
    std::shared_ptr<TrailSplatEntry> getTrailSplat(const TrailSplatCacheKey& key);
    void putTrailSplat(const TrailSplatCacheKey& key, const std::shared_ptr<TrailSplatEntry>& entry);

    // Sequential trail feedback buffer management per plugin instance.
    std::shared_ptr<SequentialTrailState> getOrCreateSequentialState(const void* instanceId);
    void invalidateInstance(const void* instanceId);
    void clear();

    uint64_t hits() const { return m_hits.load(std::memory_order_relaxed); }
    uint64_t misses() const { return m_misses.load(std::memory_order_relaxed); }
    void resetStats();

private:
    struct LruItem {
        bool isSplat = false;
        FlowCacheKey flowKey;
        TrailSplatCacheKey splatKey;
        std::shared_ptr<FlowEntry> flow;
        std::shared_ptr<TrailSplatEntry> splat;
    };

    void evictIfNeeded();

    mutable std::mutex m_mutex;
    std::size_t m_maxBytes;
    std::size_t m_currentBytes = 0;

    using LruList = std::list<LruItem>;
    LruList m_lru;
    std::unordered_map<FlowCacheKey, LruList::iterator, FlowCacheKeyHash> m_flowMap;
    std::unordered_map<TrailSplatCacheKey, LruList::iterator, TrailSplatCacheKeyHash> m_splatMap;
    std::unordered_map<const void*, std::shared_ptr<SequentialTrailState>> m_seqStates;

    std::atomic<uint64_t> m_hits{0};
    std::atomic<uint64_t> m_misses{0};
};

} // namespace bmb
