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
    static FlowCache instance(1536ULL * 1024ULL * 1024ULL); // 1.5 GB LRU budget on RTX 4090
    return instance;
}

void FlowCache::setMaxBytes(std::size_t maxBytes) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_maxBytes = maxBytes;
    evictIfNeeded();
}

std::size_t FlowCache::currentBytes() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_currentBytes;
}

std::shared_ptr<FlowEntry> FlowCache::get(const FlowCacheKey& key) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_map.find(key);
    if (it == m_map.end()) {
        m_misses.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    // Move to front of LRU list
    m_lru.splice(m_lru.begin(), m_lru, it->second);
    m_hits.fetch_add(1, std::memory_order_relaxed);
    return it->second->second;
}

void FlowCache::put(const FlowCacheKey& key, const std::shared_ptr<FlowEntry>& entry) {
    if (!entry) return;
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_map.find(key);
    if (it != m_map.end()) {
        m_currentBytes -= it->second->second->byteSize;
        it->second->second = entry;
        m_currentBytes += entry->byteSize;
        m_lru.splice(m_lru.begin(), m_lru, it->second);
    } else {
        m_lru.emplace_front(key, entry);
        m_map.emplace(key, m_lru.begin());
        m_currentBytes += entry->byteSize;
    }
    evictIfNeeded();
}

void FlowCache::evictIfNeeded() {
    while (m_currentBytes > m_maxBytes && !m_lru.empty()) {
        auto last = std::prev(m_lru.end());
        m_currentBytes -= last->second->byteSize;
        m_map.erase(last->first);
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
        if (it->first.instanceId == instanceId) {
            m_currentBytes -= it->second->byteSize;
            m_map.erase(it->first);
            it = m_lru.erase(it);
        } else {
            ++it;
        }
    }
    m_seqStates.erase(instanceId);
}

void FlowCache::clear() {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_map.clear();
    m_lru.clear();
    m_seqStates.clear();
    m_currentBytes = 0;
}

void FlowCache::resetStats() {
    m_hits.store(0, std::memory_order_relaxed);
    m_misses.store(0, std::memory_order_relaxed);
}

} // namespace bmb
