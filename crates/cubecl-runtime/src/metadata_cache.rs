//! Cache of per-launch **metadata info buffers** (kernel shapes, strides and
//! scalars) keyed by the exact info bytes they were built from (see
//! [`InfoCacheKey`]).
//!
//! Info depends only on shapes and scalar arguments — not on the tensor data
//! pointers, which are separate kernel arguments — so two launches with identical
//! info, even of different kernels, can share one read-only device buffer. Caching those
//! buffers removes a fresh allocation and a host→device copy from every launch,
//! and it is what makes hardware graph capture clean: a stable-shape decode's
//! launches all hit warm buffers, so nothing is allocated or copied inside the
//! capture window (a device allocation mid-capture is illegal).
//!
//! [`MetadataCachePolicy`] owns every decision: whether a given metadata info is
//! worth caching at all (small enough, and the cache enabled) and how many
//! entries to keep before the least-recently-used one is evicted. Its two knobs,
//! `max_entries` and `max_cached_size`, are tuned by the backend. The current
//! [`CacheMode`] feeds into those same decisions — during graph capture the
//! policy caches everything and evicts nothing — so the runtime never has to
//! special-case capture: it just asks the policy and follows the answer.
//!
//! A captured graph records the device pointer of every info buffer its launches
//! touch, so those buffers must outlive the graph. While a capture is recording,
//! every entry the launch path touches — a fresh miss *or* a hit on a buffer
//! built earlier in normal operation — is **pinned** to the graph being built.
//! Pinned entries are never evicted, so the pointer a graph recorded stays valid
//! for its whole life even if the cache would otherwise reclaim it.
//! [`capture_commit`](MetadataInfoCache::capture_commit) seals those pins under
//! the graph's id at `end_capture`, and
//! [`graph_release`](MetadataInfoCache::graph_release) drops them when the graph
//! is destroyed, freeing any buffer no other live graph still pins. Pins are
//! refcounted, so an info buffer shared by two graphs survives until both are
//! gone.
//!
//! The runtime drives it in three steps, so a value is only ever cloned into the
//! cache when it will actually be kept:
//!
//! 1. ask [`MetadataInfoCache::should_cache`] — if `false`, create the value and
//!    return it directly, never touching the cache;
//! 2. otherwise [`get`](MetadataInfoCache::get) — a hit returns the cached value;
//! 3. on a miss, create the value and [`insert`](MetadataInfoCache::insert) it.

use alloc::vec::Vec;
use cubecl_environment::collections::{HashMap, HashSet};

use crate::id::GraphId;

/// Identifies a cached info buffer: the exact metadata words (shapes, strides,
/// scalars) the buffer was built from — nothing else.
///
/// The kernel is deliberately **not** part of the key. An info buffer is
/// read-only metadata whose bytes are fully determined by these words, so two
/// launches (even of different kernels) with identical info want a byte-identical
/// buffer and can safely share one. Keying on the words alone means a hit needs
/// only the caller's borrowed slice — no owned key to clone on the hot path — and
/// buffers are reused across kernels, not just within one.
pub type InfoCacheKey = Vec<u64>;

/// How the cache should behave for the current launch. Fed into the
/// [`MetadataCachePolicy`], so it shapes every decision, not just a setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    /// Normal operation: cache only info small enough to be worth it, and evict
    /// the least-recently-used entry once the cache is full.
    Normal,
    /// Graph-capture warmup/recording: cache every info buffer regardless of
    /// size and never evict, so the capture window finds every buffer warm and
    /// no entry a recorded launch depends on is dropped mid-capture.
    Capture,
}

/// Every caching decision, in one place. Given an info's size and the current
/// [`CacheMode`], the policy decides whether that info is cached at all
/// ([`should_cache`](Self::should_cache)) and, through its
/// [`capacity`](Self::capacity), when the cache must evict to stay bounded. How
/// those decisions are carried out (lookups, eviction) is the cache's job and
/// differs per runtime; *what* to do lives here.
///
/// `max_entries` and `max_cached_size` are the backend-tunable knobs.
#[derive(Debug, Clone, Copy)]
pub struct MetadataCachePolicy {
    max_entries: usize,
    max_cached_size: usize,
    mode: CacheMode,
}

impl Default for MetadataCachePolicy {
    fn default() -> Self {
        // 4096 entries, ~256 metadata words (2048 bytes): a handful of tensors'
        // shapes and strides. Larger infos (many tensors / high rank) are
        // cheaper to rebuild than to hash and look up.
        Self::new(4096, 2048)
    }
}

impl MetadataCachePolicy {
    /// Build a policy in [`CacheMode::Normal`]. `max_entries` caps the number of
    /// cached entries; `max_cached_size` (bytes) is the largest info still worth
    /// caching in normal operation.
    pub fn new(max_entries: usize, max_cached_size: usize) -> Self {
        Self {
            max_entries,
            max_cached_size,
            mode: CacheMode::Normal,
        }
    }

    /// Switch the mode driving the policy's decisions (see [`CacheMode`]).
    pub fn mode(&mut self, mode: CacheMode) {
        self.mode = mode;
    }

    /// Whether an info of `size` bytes should go through the cache at all. In
    /// [`CacheMode::Normal`] a too-large info (or a disabled cache) is skipped:
    /// the runtime should create it directly, without a lookup or a store. In
    /// [`CacheMode::Capture`] everything is cached so the capture window stays
    /// warm.
    pub fn should_cache(&self, size: usize) -> bool {
        match self.mode {
            CacheMode::Capture => true,
            CacheMode::Normal => self.max_entries > 0 && size <= self.max_cached_size,
        }
    }

    /// The entry bound the cache must hold to, or `None` when unbounded. In
    /// [`CacheMode::Capture`] the cache is unbounded and never evicts (dropping
    /// an entry could free a buffer a recorded launch still needs); in
    /// [`CacheMode::Normal`] it is capped at `max_entries`.
    pub fn capacity(&self) -> Option<usize> {
        match self.mode {
            CacheMode::Capture => None,
            CacheMode::Normal => Some(self.max_entries),
        }
    }

    /// Whether entries touched right now must be pinned to the graph being
    /// captured. True in [`CacheMode::Capture`]: a graph records the device
    /// pointer of every info buffer its launches touch, so those entries must
    /// not be evicted for the graph's lifetime — even after the cache returns to
    /// [`CacheMode::Normal`], and even if the buffer lives in the regular
    /// (non-persistent) pool and so is not otherwise retained by the graph.
    pub fn pins_entries(&self) -> bool {
        matches!(self.mode, CacheMode::Capture)
    }
}

#[derive(Debug)]
struct Entry<V> {
    value: V,
    /// Clock tick of this entry's most recent use (insert or hit); the smallest
    /// marks the least-recently-used entry, evicted first.
    last_used: u64,
    /// Number of live captured graphs that pinned this entry. While `> 0` the
    /// entry is never evicted, so every graph that recorded this info buffer
    /// keeps replaying against the exact buffer it captured. Dropped back toward
    /// zero as those graphs are destroyed (see [`MetadataInfoCache::graph_release`]).
    locks: u32,
}

/// A cache of metadata info buffers keyed by [`InfoCacheKey`], generic over the
/// value type `V` (a device buffer handle for a compute backend). Evicting an
/// entry drops its `V`; when `V` is a memory handle that returns the buffer to
/// the pool, so the cache never pins more device memory than its live entries.
///
/// All policy is delegated to [`MetadataCachePolicy`]; this type only carries
/// out its decisions. See the [module docs](self) for the intended
/// `should_cache` → `get` → `insert` flow.
#[derive(Debug)]
pub struct MetadataInfoCache<V> {
    entries: HashMap<InfoCacheKey, Entry<V>>,
    policy: MetadataCachePolicy,
    /// Monotonic logical clock; advanced once per [`get`](Self::get).
    clock: u64,
    /// Keys touched during the in-progress capture, each pinned exactly once,
    /// awaiting association with a [`GraphId`] at
    /// [`capture_commit`](Self::capture_commit) (or release at
    /// [`capture_discard`](Self::capture_discard) if the capture is abandoned).
    pending: HashSet<InfoCacheKey>,
    /// The keys each live captured graph pinned, so
    /// [`graph_release`](Self::graph_release) can drop that graph's locks when it
    /// is destroyed.
    graph_locks: HashMap<GraphId, Vec<InfoCacheKey>>,
}

impl<V> MetadataInfoCache<V> {
    /// Create a cache governed by `policy`.
    pub fn new(policy: MetadataCachePolicy) -> Self {
        Self {
            entries: HashMap::new(),
            policy,
            clock: 0,
            pending: HashSet::new(),
            graph_locks: HashMap::new(),
        }
    }

    /// Switch the [`CacheMode`] driving the policy (see
    /// [`MetadataCachePolicy::mode`]).
    pub fn mode(&mut self, mode: CacheMode) {
        self.policy.mode(mode);
    }

    /// Whether an info of `size` bytes should be cached — see
    /// [`MetadataCachePolicy::should_cache`]. When `false`, skip the cache
    /// entirely and create the value directly.
    pub fn should_cache(&self, size: usize) -> bool {
        self.policy.should_cache(size)
    }

    /// Number of entries currently cached.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Drop every cached entry (releasing every held `V`).
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl<V: Clone> MetadataInfoCache<V> {
    /// Look up `key`, advancing the logical clock by one tick. On a hit the
    /// entry is marked used — its recency resets — and the value is cloned out.
    /// During a capture ([`pins_entries`](MetadataCachePolicy::pins_entries)) a
    /// hit also pins the entry to the graph being recorded, once per capture.
    ///
    /// Call only when [`should_cache`](Self::should_cache) is `true`; on a miss
    /// create the value and hand it to [`insert`](Self::insert).
    pub fn get(&mut self, key: &[u64]) -> Option<V> {
        self.clock += 1;
        let clock = self.clock;
        // Pin once per capture. Check membership first (borrowed, no alloc); only
        // when this is a fresh pin do we materialize an owned key for `pending`.
        // `pending`/`entries` are disjoint fields, so both borrows coexist.
        let pin = self.policy.pins_entries() && !self.pending.contains(key);
        let entry = self.entries.get_mut(key)?;
        entry.last_used = clock;
        if pin {
            self.pending.insert(key.to_vec());
            entry.locks += 1;
        }
        Some(entry.value.clone())
    }

    /// Store `value` under `key` after a [`get`](Self::get) miss. During a
    /// capture the new entry is pinned to the graph being recorded; otherwise it
    /// evicts the least-recently-used *unpinned* entry first if the cache is at
    /// [capacity](MetadataCachePolicy::capacity) (a capture is unbounded and
    /// never evicts).
    ///
    /// Because the runtime only reaches here on a
    /// [`should_cache`](Self::should_cache) miss, the value it clones in is
    /// always kept — no wasted clones.
    pub fn insert(&mut self, key: InfoCacheKey, value: V) {
        if let Some(capacity) = self.policy.capacity() {
            if capacity == 0 {
                return;
            }
            if self.entries.len() >= capacity {
                self.evict_least_recently_used();
            }
        }
        // A miss during capture pins the fresh entry to the graph being recorded.
        let locks = if self.policy.pins_entries() {
            self.pending.insert(key.clone());
            1
        } else {
            0
        };
        self.entries.insert(
            key,
            Entry {
                value,
                last_used: self.clock,
                locks,
            },
        );
    }

    /// Seal the entries pinned during the just-finished capture under `graph`,
    /// so [`graph_release`](Self::graph_release) can drop their locks when the
    /// graph is destroyed. Call once from `end_capture` after the graph is built.
    pub fn capture_commit(&mut self, graph: GraphId) {
        if !self.pending.is_empty() {
            let keys = self.pending.drain().collect();
            self.graph_locks.insert(graph, keys);
        }
    }

    /// Drop the locks taken during a capture that was abandoned (never turned
    /// into a graph), so the touched entries become evictable again. The entries
    /// themselves stay as ordinary cached values.
    pub fn capture_discard(&mut self) {
        let keys: Vec<_> = self.pending.drain().collect();
        for key in keys {
            if let Some(entry) = self.entries.get_mut(&key) {
                entry.locks = entry.locks.saturating_sub(1);
            }
        }
    }

    /// Release the entries a destroyed `graph` pinned. An entry no other live
    /// graph still pins is removed, freeing its buffer — this is how the cache
    /// is cleaned up when a graph is destroyed.
    pub fn graph_release(&mut self, graph: GraphId) {
        let Some(keys) = self.graph_locks.remove(&graph) else {
            return;
        };
        for key in keys {
            let drop_entry = match self.entries.get_mut(&key) {
                Some(entry) => {
                    entry.locks = entry.locks.saturating_sub(1);
                    entry.locks == 0
                }
                None => false,
            };
            if drop_entry {
                self.entries.remove(&key);
            }
        }
    }

    /// Drop the entry whose last use is oldest (largest "time since last use"),
    /// skipping entries pinned to a live graph.
    fn evict_least_recently_used(&mut self) {
        let victim = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.locks == 0)
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone());
        if let Some(key) = victim {
            self.entries.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u64) -> InfoCacheKey {
        alloc::vec![n]
    }

    fn cache(max_entries: usize) -> MetadataInfoCache<u32> {
        MetadataInfoCache::new(MetadataCachePolicy::new(max_entries, 64))
    }

    #[test]
    fn normal_mode_gates_on_size() {
        let cache = cache(8);
        assert!(cache.should_cache(64), "within max_cached_size");
        assert!(!cache.should_cache(65), "over max_cached_size");
    }

    #[test]
    fn capture_mode_caches_any_size() {
        let mut cache = cache(8);
        cache.mode(CacheMode::Capture);
        assert!(cache.should_cache(10_000));
    }

    #[test]
    fn hit_returns_value_and_records_use() {
        let mut cache = cache(8);
        let k = key(1);
        assert!(cache.get(&k).is_none());
        cache.insert(k.clone(), 42);
        assert_eq!(cache.get(&k), Some(42));
    }

    #[test]
    fn normal_mode_evicts_least_recently_used() {
        // Capacity 2: fill it, keep entry 0 hot so entry 1 is the LRU, then
        // insert a third entry and expect entry 1 evicted, entry 0 kept.
        let mut cache = cache(2);
        cache.insert(key(0), 0);
        cache.insert(key(1), 1);

        // Touch entry 0 so entry 1 becomes least-recently-used.
        assert_eq!(cache.get(&key(0)), Some(0));

        cache.insert(key(2), 2);
        assert_eq!(cache.len(), 2, "stayed at capacity");
        assert_eq!(cache.get(&key(0)), Some(0), "recently used entry kept");
        assert!(cache.get(&key(1)).is_none(), "least-recently-used evicted");
        assert_eq!(cache.get(&key(2)), Some(2), "new entry cached");
    }

    #[test]
    fn capture_mode_is_unbounded() {
        let mut cache = cache(2);
        cache.mode(CacheMode::Capture);
        for i in 0..10 {
            cache.insert(key(i), i as u32);
        }
        assert_eq!(cache.len(), 10, "capture must cache every buffer");
    }

    #[test]
    fn zero_capacity_disables_caching() {
        let mut cache = cache(0);
        assert!(!cache.should_cache(8), "disabled cache never caches");
        cache.insert(key(0), 0);
        assert!(cache.is_empty());
    }

    /// Capture one entry into a graph, then thrash the cache well past capacity
    /// in normal mode: the pinned entry must survive (its buffer is still
    /// replayed against), and only after the graph is released can it be evicted.
    #[test]
    fn pinned_entry_survives_eviction_until_graph_released() {
        let mut cache = cache(2);
        let graph = GraphId::new();

        // Capture pins entry 0.
        cache.mode(CacheMode::Capture);
        cache.insert(key(0), 0);
        cache.capture_commit(graph);

        // Back to normal: flood the cache far past capacity (get-then-insert,
        // as the launch path does, so recency advances per entry).
        cache.mode(CacheMode::Normal);
        for i in 1..20 {
            cache.get(&key(i));
            cache.insert(key(i), i as u32);
        }
        assert_eq!(cache.get(&key(0)), Some(0), "pinned entry never evicted");

        // Destroying the graph releases the pin and drops the entry.
        cache.graph_release(graph);
        assert!(cache.get(&key(0)).is_none(), "released entry cleaned up");
    }

    /// A cache hit during capture (on a buffer built earlier in normal mode)
    /// must pin that entry — otherwise later eviction would free a buffer the
    /// graph still replays against. This is the finding-#1 regression guard.
    #[test]
    fn capture_hit_on_normal_entry_pins_it() {
        let mut cache = cache(2);
        let graph = GraphId::new();

        // Built in normal mode (e.g. regular pool), cached.
        cache.insert(key(0), 0);

        // Captured: the launch hits the existing entry, which must pin it.
        cache.mode(CacheMode::Capture);
        assert_eq!(cache.get(&key(0)), Some(0));
        cache.capture_commit(graph);

        cache.mode(CacheMode::Normal);
        for i in 1..20 {
            cache.get(&key(i));
            cache.insert(key(i), i as u32);
        }
        assert_eq!(cache.get(&key(0)), Some(0), "hit-pinned entry survives");
    }

    /// An entry shared by two graphs stays until both are destroyed (refcount).
    #[test]
    fn pin_is_refcounted_across_graphs() {
        let mut cache = cache(8);
        let (g1, g2) = (GraphId::new(), GraphId::new());

        cache.mode(CacheMode::Capture);
        cache.insert(key(0), 0);
        cache.capture_commit(g1);

        // Second capture hits the same entry.
        assert_eq!(cache.get(&key(0)), Some(0));
        cache.capture_commit(g2);

        cache.mode(CacheMode::Normal);
        cache.graph_release(g1);
        assert_eq!(cache.get(&key(0)), Some(0), "still pinned by g2");
        cache.graph_release(g2);
        assert!(cache.get(&key(0)).is_none(), "gone once both released");
    }

    /// An abandoned capture releases its pins but keeps the entries as ordinary
    /// cached values (they become evictable again).
    #[test]
    fn capture_discard_unpins_without_removing() {
        let mut cache = cache(8);
        cache.mode(CacheMode::Capture);
        cache.insert(key(0), 0);
        cache.capture_discard();

        // Entry still present, but now evictable: flood past capacity in normal.
        cache.mode(CacheMode::Normal);
        assert_eq!(cache.get(&key(0)), Some(0), "kept as a normal entry");
        for i in 1..20 {
            cache.get(&key(i));
            cache.insert(key(i), i as u32);
        }
        assert!(cache.get(&key(0)).is_none(), "no longer pinned, evicted");
    }
}
