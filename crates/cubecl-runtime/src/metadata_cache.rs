//! Cache of per-launch **metadata info buffers** (kernel shapes, strides and
//! scalars) keyed by the kernel and the exact info bytes it was built from.
//!
//! A kernel's info depends only on its shapes and scalar arguments — not on the
//! tensor data pointers, which are separate kernel arguments — so two launches
//! with identical info can share one read-only device buffer. Caching those
//! buffers removes a fresh allocation and a host→device copy from every launch,
//! and it is what makes hardware graph capture clean: a stable-shape decode's
//! launches all hit warm buffers, so nothing is allocated or copied inside the
//! capture window (a device allocation mid-capture is illegal).
//!
//! Two knobs govern the cache, split by scope:
//!
//! * [`CacheSettings`] — cache-wide **admission**: how many entries to keep and
//!   how large an info buffer is still worth caching (past some size the key
//!   hash/compare costs more than just re-allocating the buffer).
//! * [`CachePolicy`] — per-entry **invalidation**: which entries are stale and
//!   may be evicted to make room. [`TimeSinceLastUse`] is the default strategy.
//!
//! During graph capture both are suspended (see [`CacheMode::Capture`]): every
//! info buffer is cached regardless of size and nothing is ever invalidated, so
//! the capture window finds every buffer warm and drops none out from under a
//! recorded launch.

use alloc::boxed::Box;
use alloc::vec::Vec;
use hashbrown::HashMap;

use crate::id::KernelId;

/// Identifies a cached info buffer: the kernel plus the exact metadata words
/// (shapes, strides, scalars) the buffer was built from.
pub type InfoCacheKey = (KernelId, Vec<u64>);

/// How the cache should treat the current launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheMode {
    /// Normal operation: honor the [size gate](CacheSettings::max_cached_size)
    /// and let the [policy](CachePolicy) invalidate stale entries to make room.
    Normal,
    /// Graph-capture warmup/recording: cache every info buffer regardless of
    /// size and never invalidate, so the capture window finds every buffer warm
    /// and no entry a recorded launch depends on is dropped mid-capture.
    Capture,
}

/// Cache-wide admission settings — the "global" knobs that decide *what* is
/// worth caching, independent of any per-entry [invalidation policy](CachePolicy).
#[derive(Debug, Clone, Copy)]
pub struct CacheSettings {
    /// Hard cap on the number of cached entries in [`CacheMode::Normal`]. When
    /// the cache is full a fresh entry first tries to reclaim room by
    /// invalidating stale entries; if none are stale the entry is left uncached
    /// (correct, just not reused). Bypassed in [`CacheMode::Capture`].
    pub max_entries: usize,
    /// Largest info buffer, in bytes, still worth caching in
    /// [`CacheMode::Normal`]. Above this the cost of hashing and comparing the
    /// key can exceed the cost of just re-allocating the buffer, so the launch
    /// falls back to a fresh per-launch buffer. Bypassed in
    /// [`CacheMode::Capture`], where every buffer must be cached.
    pub max_cached_size: usize,
}

impl Default for CacheSettings {
    fn default() -> Self {
        Self {
            // Matches the previous fixed `INFO_CACHE_MAX`.
            max_entries: 4096,
            // ~256 metadata words: a handful of tensors' shapes and strides.
            // Larger infos (many tensors / high rank) are cheaper to rebuild
            // than to look up.
            max_cached_size: 2048,
        }
    }
}

/// Per-entry statistics handed to a [`CachePolicy`] so it can decide whether an
/// entry is stale. Deliberately richer than any single policy needs, so new
/// strategies (least-frequently-used, size-weighted, …) can be added without
/// touching the cache.
#[derive(Debug, Clone, Copy)]
pub struct EntryStats {
    /// Logical ticks elapsed since this entry was last used. One tick passes
    /// per cache lookup (i.e. per launch that consults the cache), so this is
    /// "how many launches ago was this info last needed".
    pub idle_ticks: u64,
    /// Number of times this entry has been reused since it was inserted.
    pub hits: u64,
    /// Size in bytes of the cached info buffer.
    pub size: usize,
}

/// Decides which cache entries are stale and may be evicted. Implementations
/// must be cheap: [`should_invalidate`](CachePolicy::should_invalidate) is
/// called once per entry whenever the cache needs to reclaim room.
pub trait CachePolicy: core::fmt::Debug {
    /// Whether an entry with these [stats](EntryStats) should be invalidated.
    fn should_invalidate(&self, stats: &EntryStats) -> bool;
}

/// Evict an entry once it has gone unused for more than [`max_idle_ticks`]
/// lookups — a time-since-last-use (recency) strategy. Hot entries keep
/// resetting their idle count on every hit and survive; entries whose shape has
/// dropped out of the working set age out and free their buffers.
///
/// [`max_idle_ticks`]: TimeSinceLastUse::max_idle_ticks
#[derive(Debug, Clone, Copy)]
pub struct TimeSinceLastUse {
    /// Idle-tick threshold past which an entry is considered stale. One tick
    /// passes per cache lookup.
    pub max_idle_ticks: u64,
}

impl Default for TimeSinceLastUse {
    fn default() -> Self {
        Self {
            max_idle_ticks: 2048,
        }
    }
}

impl CachePolicy for TimeSinceLastUse {
    fn should_invalidate(&self, stats: &EntryStats) -> bool {
        stats.idle_ticks > self.max_idle_ticks
    }
}

#[derive(Debug)]
struct Entry<V> {
    value: V,
    /// Clock tick of this entry's most recent use (insert or hit).
    last_used: u64,
    hits: u64,
    size: usize,
}

/// A cache of metadata info buffers keyed by [`InfoCacheKey`], generic over the
/// value type `V` (a device buffer handle for a compute backend). Evicting an
/// entry drops its `V`; when `V` is a memory handle that returns the buffer to
/// the pool, so the cache never pins more device memory than its live entries.
///
/// Admission is governed by [`CacheSettings`] and invalidation by a
/// [`CachePolicy`], both suspended during graph capture (see [`CacheMode`]).
#[derive(Debug)]
pub struct MetadataInfoCache<V> {
    entries: HashMap<InfoCacheKey, Entry<V>>,
    settings: CacheSettings,
    policy: Box<dyn CachePolicy>,
    /// Monotonic logical clock; advanced once per [`get`](Self::get).
    clock: u64,
}

impl<V> MetadataInfoCache<V> {
    /// Create a cache with the given admission `settings` and invalidation
    /// `policy`.
    pub fn new(settings: CacheSettings, policy: Box<dyn CachePolicy>) -> Self {
        Self {
            entries: HashMap::new(),
            settings,
            policy,
            clock: 0,
        }
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
    /// entry is marked used — its recency resets and its hit count grows — and
    /// the value is cloned out; on a miss the caller creates the value and
    /// offers it back via [`insert`](Self::insert).
    pub fn get(&mut self, key: &InfoCacheKey) -> Option<V> {
        self.clock += 1;
        let entry = self.entries.get_mut(key)?;
        entry.last_used = self.clock;
        entry.hits += 1;
        Some(entry.value.clone())
    }

    /// Offer a freshly created value for a key that just [missed](Self::get).
    ///
    /// In [`Normal`](CacheMode::Normal) mode this honors admission: an info
    /// larger than [`max_cached_size`](CacheSettings::max_cached_size) is not
    /// cached, and when the cache is at
    /// [`max_entries`](CacheSettings::max_entries) it first evicts stale entries
    /// per the [policy](CachePolicy) — if none are stale the value is left
    /// uncached (the caller still uses its own copy, just without reuse). In
    /// [`Capture`](CacheMode::Capture) mode the size gate, capacity bound and
    /// invalidation are all bypassed: every buffer is cached so the capture
    /// window hits only warm entries.
    pub fn insert(&mut self, key: InfoCacheKey, value: V, size: usize, mode: CacheMode) {
        if mode == CacheMode::Normal {
            if size > self.settings.max_cached_size {
                return;
            }
            if self.entries.len() >= self.settings.max_entries {
                // Only scan for stale entries when we actually need room, so a
                // steady stream of new shapes costs at most one O(n) sweep per
                // time the cache fills rather than one per miss.
                self.invalidate_stale();
                if self.entries.len() >= self.settings.max_entries {
                    return;
                }
            }
        }

        self.entries.insert(
            key,
            Entry {
                value,
                last_used: self.clock,
                hits: 0,
                size,
            },
        );
    }

    /// Evict every entry the [policy](CachePolicy) considers stale. Never run in
    /// [`CacheMode::Capture`].
    fn invalidate_stale(&mut self) {
        let clock = self.clock;
        let policy = &self.policy;
        self.entries.retain(|_, entry| {
            let stats = EntryStats {
                idle_ticks: clock.saturating_sub(entry.last_used),
                hits: entry.hits,
                size: entry.size,
            };
            !policy.should_invalidate(&stats)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(n: u64) -> InfoCacheKey {
        (KernelId::new::<()>(), alloc::vec![n])
    }

    fn cache(max_entries: usize, max_idle_ticks: u64) -> MetadataInfoCache<u32> {
        MetadataInfoCache::new(
            CacheSettings {
                max_entries,
                max_cached_size: 64,
            },
            Box::new(TimeSinceLastUse { max_idle_ticks }),
        )
    }

    #[test]
    fn hit_returns_value_and_records_use() {
        let mut cache = cache(8, 1000);
        let k = key(1);
        assert!(cache.get(&k).is_none());
        cache.insert(k.clone(), 42, 8, CacheMode::Normal);
        assert_eq!(cache.get(&k), Some(42));
    }

    #[test]
    fn normal_mode_rejects_oversized_info() {
        let mut cache = cache(8, 1000);
        let k = key(1);
        cache.insert(k.clone(), 42, 65, CacheMode::Normal); // over max_cached_size
        assert!(cache.get(&k).is_none());
    }

    #[test]
    fn capture_mode_caches_oversized_info() {
        let mut cache = cache(8, 1000);
        let k = key(1);
        cache.insert(k.clone(), 42, 10_000, CacheMode::Capture);
        assert_eq!(cache.get(&k), Some(42));
    }

    #[test]
    fn capture_mode_bypasses_capacity() {
        let mut cache = cache(2, 1000);
        for i in 0..10 {
            cache.insert(key(i), i as u32, 8, CacheMode::Capture);
        }
        assert_eq!(cache.len(), 10, "capture must cache every buffer");
    }

    #[test]
    fn time_since_last_use_evicts_stale_not_hot() {
        // Capacity 2: fill it, keep entry 0 hot, then insert a third entry and
        // expect the idle entry (1) to be evicted while the hot one (0) stays.
        let mut cache = cache(2, 3);
        cache.insert(key(0), 0, 8, CacheMode::Normal);
        cache.insert(key(1), 1, 8, CacheMode::Normal);

        // Keep entry 0 hot across several ticks so entry 1 goes idle.
        for _ in 0..5 {
            assert_eq!(cache.get(&key(0)), Some(0));
        }

        // Cache is full; inserting a new key must reclaim room by evicting the
        // stale entry 1, keeping the hot entry 0.
        cache.insert(key(2), 2, 8, CacheMode::Normal);
        assert_eq!(cache.get(&key(0)), Some(0), "hot entry survived");
        assert!(cache.get(&key(1)).is_none(), "idle entry evicted");
        assert_eq!(cache.get(&key(2)), Some(2), "new entry cached");
    }

    #[test]
    fn full_of_hot_entries_leaves_new_uncached() {
        // Capacity 2, both entries kept hot: a third insert finds nothing stale
        // and leaves the newcomer uncached (correct fallback).
        let mut cache = cache(2, 1000);
        cache.insert(key(0), 0, 8, CacheMode::Normal);
        cache.insert(key(1), 1, 8, CacheMode::Normal);
        cache.insert(key(2), 2, 8, CacheMode::Normal);
        assert_eq!(cache.len(), 2);
        assert!(cache.get(&key(2)).is_none());
    }
}
