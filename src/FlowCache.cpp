#include "FlowCache.h"

namespace bmb {

FlowEntry::~FlowEntry() {
    if (isGpu && gpuFreeFn) {
        if (devFlow4) {
            gpuFreeFn(devFlow4);
            devFlow4 = 0;
        }
        if (devConf2) {
            gpuFreeFn(devConf2);
            devConf2 = 0;
        }
    }
}

TrailSplatEntry::~TrailSplatEntry() {
    if (devAccum4 && gpuFreeFn) {
        gpuFreeFn(devAccum4);
        devAccum4 = 0;
    }
}

SequentialTrailState::~SequentialTrailState() {
    reset();
}

void SequentialTrailState::reset() {
    if (devAccumRGBA && gpuFreeFn) {
        gpuFreeFn(devAccumRGBA);
    }
    devAccumRGBA = 0;
    hostAccumRGBA.clear();
    width = 0;
    height = 0;
    lastFrame = -999999;
}

FlowCache::FlowCache(std::size_t maxBytes)
    : m_maxBytes(maxBytes) {}

FlowCache::~FlowCache() {
    clear();
}

FlowCache& FlowCache::global() {
    static FlowCache instance(1536ULL * 1024ULL * 1024ULL); // Dynamically sized up to 5 GB at CUDA init
    return instance;
}

void FlowCache::setMaxBytes(std::size_t maxBytes) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_maxBytes = maxBytes;
    evictIfNeeded();
}

std::size_t FlowCache::maxBytes() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_maxBytes;
}

std::size_t FlowCache::currentBytes() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_currentBytes;
}

std::shared_ptr<FlowEntry> FlowCache::get(const FlowCacheKey& key) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_flowMap.find(key);
    if (it == m_flowMap.end()) {
        m_misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    m_lru.splice(m_lru.begin(), m_lru, it->second);
    m_hits.fetch_add(1, std::memory_order_relaxed);
    return it->second->flow;
}

void FlowCache::put(const FlowCacheKey& key, const std::shared_ptr<FlowEntry>& entry) {
    if (!entry) return;
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_flowMap.find(key);
    if (it != m_flowMap.end()) {
        m_currentBytes -= it->second->flow->byteSize;
        it->second->flow = entry;
        m_currentBytes += entry->byteSize;
        m_lru.splice(m_lru.begin(), m_lru, it->second);
    } else {
        LruItem item;
        item.isSplat = false;
        item.flowKey = key;
        item.flow = entry;
        m_lru.emplace_front(std::move(item));
        m_flowMap.emplace(key, m_lru.begin());
        m_currentBytes += entry->byteSize;
    }
    evictIfNeeded();
}

std::shared_ptr<TrailSplatEntry> FlowCache::getTrailSplat(const TrailSplatCacheKey& key) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_splatMap.find(key);
    if (it == m_splatMap.end()) {
        return nullptr;
    }
    m_lru.splice(m_lru.begin(), m_lru, it->second);
    m_hits.fetch_add(1, std::memory_order_relaxed);
    return it->second->splat;
}

void FlowCache::putTrailSplat(const TrailSplatCacheKey& key, const std::shared_ptr<TrailSplatEntry>& entry) {
    if (!entry) return;
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_splatMap.find(key);
    if (it != m_splatMap.end()) {
        m_currentBytes -= it->second->splat->byteSize;
        it->second->splat = entry;
        m_currentBytes += entry->byteSize;
        m_lru.splice(m_lru.begin(), m_lru, it->second);
    } else {
        LruItem item;
        item.isSplat = true;
        item.splatKey = key;
        item.splat = entry;
        m_lru.emplace_front(std::move(item));
        m_splatMap.emplace(key, m_lru.begin());
        m_currentBytes += entry->byteSize;
    }
    evictIfNeeded();
}

void FlowCache::evictIfNeeded() {
    while (m_currentBytes > m_maxBytes && !m_lru.empty()) {
        auto last = std::prev(m_lru.end());
        if (last->isSplat) {
            m_currentBytes -= last->splat->byteSize;
            m_splatMap.erase(last->splatKey);
        } else {
            m_currentBytes -= last->flow->byteSize;
            m_flowMap.erase(last->flowKey);
        }
        m_lru.pop_back();
    }
}

std::shared_ptr<SequentialTrailState> FlowCache::getOrCreateSequentialState(const void* instanceId) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_seqStates.find(instanceId);
    if (it != m_seqStates.end()) {
        return it->second;
    }
    auto state = std::make_shared<SequentialTrailState>();
    m_seqStates.emplace(instanceId, state);
    return state;
}

void FlowCache::invalidateInstance(const void* instanceId) {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (auto it = m_lru.begin(); it != m_lru.end();) {
        const void* owner = it->isSplat ? it->splatKey.instanceId : it->flowKey.instanceId;
        if (owner == instanceId) {
            if (it->isSplat) {
                m_currentBytes -= it->splat->byteSize;
                m_splatMap.erase(it->splatKey);
            } else {
                m_currentBytes -= it->flow->byteSize;
                m_flowMap.erase(it->flowKey);
            }
            it = m_lru.erase(it);
        } else {
            ++it;
        }
    }
    m_seqStates.erase(instanceId);
}

void FlowCache::clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_flowMap.clear();
    m_splatMap.clear();
    m_lru.clear();
    m_seqStates.clear();
    m_currentBytes = 0;
}

void FlowCache::resetStats() {
    m_hits.store(0, std::memory_order_relaxed);
    m_misses.store(0, std::memory_order_relaxed);
}

} // namespace bmb
