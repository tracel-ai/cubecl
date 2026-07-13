use super::{
    MemoryConfiguration, MemoryPoolOptions, MemoryUsage, PoolType,
    memory_pool::{ExclusiveMemoryPool, MemoryPool, PersistentPool, SlicedPool},
};
use crate::{
    config::{
        CubeClRuntimeConfig, RuntimeConfig,
        memory::{
            MemoryLogLevel, MemoryPoolConfig, MemoryPoolsConfig, MemoryPoolsPreset,
            PersistentMemory,
        },
    },
    logging::ServerLogger,
    memory_management::{BytesFormat, memory_pool::Slice},
    server::IoError,
    storage::{ComputeStorage, StorageHandle},
};

use alloc::format;
use alloc::string::{String, ToString};
#[cfg(not(exclusive_memory_only))]
use alloc::vec;
use alloc::vec::Vec;
use cubecl_common::{backtrace::BackTrace, stub::Arc};
use cubecl_ir::MemoryDeviceProperties;
use hashbrown::HashSet;

pub use super::memory_pool::{ManagedMemoryBinding, handle::*};

// These are 288 bytes vs 64 bytes. Adding boxing isn't really worth
// saving the 200 bytes.
#[allow(clippy::large_enum_variant)]
enum DynamicPool {
    Sliced(SlicedPool),
    Exclusive(ExclusiveMemoryPool),
}

impl MemoryPool for DynamicPool {
    fn accept(&self, size: u64) -> bool {
        match self {
            DynamicPool::Sliced(pool) => pool.accept(size),
            DynamicPool::Exclusive(pool) => pool.accept(size),
        }
    }

    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        match self {
            DynamicPool::Sliced(m) => m.find(binding),
            DynamicPool::Exclusive(m) => m.find(binding),
        }
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn try_reserve(&mut self, size: u64) -> Option<ManagedMemoryHandle> {
        match self {
            DynamicPool::Sliced(m) => m.try_reserve(size),
            DynamicPool::Exclusive(m) => m.try_reserve(size),
        }
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, storage))
    )]
    fn alloc<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        size: u64,
    ) -> Result<ManagedMemoryHandle, IoError> {
        match self {
            DynamicPool::Sliced(m) => m.alloc(storage, size),
            DynamicPool::Exclusive(m) => m.alloc(storage, size),
        }
    }

    fn get_memory_usage(&self) -> MemoryUsage {
        match self {
            DynamicPool::Sliced(m) => m.get_memory_usage(),
            DynamicPool::Exclusive(m) => m.get_memory_usage(),
        }
    }

    fn cleanup<Storage: ComputeStorage>(
        &mut self,
        storage: &mut Storage,
        alloc_nr: u64,
        explicit: bool,
    ) {
        match self {
            DynamicPool::Sliced(m) => m.cleanup(storage, alloc_nr, explicit),
            DynamicPool::Exclusive(m) => m.cleanup(storage, alloc_nr, explicit),
        };
        storage.flush();
    }

    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        match self {
            DynamicPool::Sliced(m) => m.bind(reserved, assigned, cursor),
            DynamicPool::Exclusive(m) => m.bind(reserved, assigned, cursor),
        }
    }
}

#[derive(Default, Clone, Copy, Debug)]
/// The mode of allocation used.
pub enum MemoryAllocationMode {
    /// Use the automatic memory management strategy for allocation.
    #[default]
    Auto,
    /// Use a persistent memory management strategy, meaning that all allocations are for data that is
    /// likely never going to be freed.
    Persistent,
}

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct MemoryManagement<Storage> {
    name: String,
    persistent: PersistentPool,
    pools: Vec<DynamicPool>,
    storage: Storage,
    alloc_reserve_count: u64,
    mode: MemoryAllocationMode,
    config: PersistentMemory,
    logger: Arc<ServerLogger>,
    /// State of the active graph capture, if any.
    capture: Option<CaptureState>,
}

/// While a graph capture is active, allocations are forced into the persistent
/// pool; slices there stay freely reusable during the window (warmup populates
/// them, the capture run reuses them), and `capture_end` hands the graph exactly
/// the slices the window touched.
struct CaptureState {
    /// The mode to restore at `capture_end`. Mid-capture [`mode`] changes land
    /// here instead of taking effect, so they can't reroute capture allocations
    /// away from the persistent pool.
    restore_mode: MemoryAllocationMode,
    /// Ids of every persistent slice handed out (reserved or freshly allocated)
    /// while the window was open — exactly the slices the graph's recorded
    /// kernels may replay against. `capture_end` retains these and nothing else,
    /// so a slice the window never touched is not over-retained, and a
    /// pre-existing slice freed and reused mid-window is still pinned.
    touched: HashSet<ManagedMemoryId>,
}

fn generate_bucket_sizes(
    start_size: u64,
    end_size: u64,
    max_buckets: usize,
    alignment: u64,
) -> Vec<u64> {
    let mut buckets = Vec::with_capacity(max_buckets);
    let log_min = (start_size as f64).ln();
    let log_max = (end_size as f64).ln();
    let log_range = log_max - log_min;

    // Pure exponential performed best, but let's try slightly denser in lower-mid range
    for i in 0..max_buckets {
        let p = i as f64 / (max_buckets - 1) as f64;
        // Slight bias toward lower-mid range with less aggressive curve than sigmoid
        let log_size = log_min + log_range * p;
        let size = log_size.exp() as u64;
        let aligned_size = size.next_multiple_of(alignment);
        buckets.push(aligned_size);
    }

    buckets.dedup();
    buckets
}

const DEALLOC_SCALE_MB: u64 = 1024 * 1024 * 1024;
const BASE_DEALLOC_PERIOD: u64 = 5000;

/// The options for creating a new [`MemoryManagement`] instance.
#[derive(Debug)]
pub struct MemoryManagementOptions {
    /// The name of the memory management.
    name: String,
    /// The [`MemoryAllocationOption`] used by this instance.
    memory: MemoryAllocationOption,
}

impl MemoryManagementOptions {
    /// Creates a new [`MemoryManagementOptions`].
    pub fn new<S: Into<String>>(name: S) -> Self {
        Self {
            name: name.into(),
            memory: MemoryAllocationOption::FromConfig,
        }
    }

    /// Forces the [`MemoryAllocationMode`] during execution to always be the provided one.
    pub fn mode(mut self, mode: MemoryAllocationMode) -> Self {
        self.memory = MemoryAllocationOption::Provided(mode);
        self
    }
}

#[derive(Default, Debug)]
/// Determines which [`MemoryAllocationMode`] is used during allocations.
enum MemoryAllocationOption {
    #[default]
    /// Uses the [`GlobalConfig`] to determine the mode of allocation.
    FromConfig,
    /// Use the provided [`MemoryAllocationMode`].
    Provided(MemoryAllocationMode),
}

/// Why a `memory.pools` config could not be turned into a pool layout.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PoolConfigError {
    /// `memory.pools` was an empty list.
    EmptyPoolList,
    /// A size field that must be non-zero was zero.
    ZeroSize {
        /// The offending field.
        field: &'static str,
    },
    /// `max_slice_size` exceeds `page_size` (a slice can never span pages).
    SliceLargerThanPage {
        /// The page size in bytes (after alignment).
        page_size: u64,
        /// The maximum slice size in bytes (after alignment).
        max_slice_size: u64,
    },
    /// `max_pool_size` is smaller than `page_size` (the cap can't fit one page).
    CapSmallerThanPage {
        /// The page size in bytes (after alignment).
        page_size: u64,
        /// The pool capacity in bytes.
        max_pool_size: u64,
    },
    /// `max_pool_size` spans more pages of `page_size` than a pool can hold.
    TooManyPages {
        /// The number of pages the configuration asks for.
        pages: u64,
    },
    /// The pool list has more entries than the pool routing can address.
    TooManyPools {
        /// The number of entries in the configuration.
        count: usize,
    },
    /// The preset is not available in this build.
    PresetUnavailable {
        /// The preset name.
        preset: &'static str,
    },
    /// Sliced pools are not available in this build (`exclusive_memory_only`).
    SlicedPoolsUnavailable,
}

impl core::fmt::Display for PoolConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PoolConfigError::EmptyPoolList => write!(f, "the pool list is empty"),
            PoolConfigError::ZeroSize { field } => write!(f, "`{field}` must be non-zero"),
            PoolConfigError::SliceLargerThanPage {
                page_size,
                max_slice_size,
            } => write!(
                f,
                "`max_slice_size` ({max_slice_size}) exceeds `page_size` ({page_size}); a slice can never span pages"
            ),
            PoolConfigError::CapSmallerThanPage {
                page_size,
                max_pool_size,
            } => write!(
                f,
                "`max_pool_size` ({max_pool_size}) is smaller than `page_size` ({page_size}); the cap can't fit a single page"
            ),
            PoolConfigError::TooManyPages { pages } => write!(
                f,
                "`max_pool_size` spans {pages} pages of `page_size`, exceeding the maximum of {}; increase `page_size` or lower the cap",
                u16::MAX
            ),
            PoolConfigError::TooManyPools { count } => write!(
                f,
                "the pool list has {count} entries, exceeding the maximum of {} dynamic pools",
                PERSISTENT_POOL_POS - 1
            ),
            PoolConfigError::PresetUnavailable { preset } => {
                write!(f, "the `{preset}` preset is not available in this build")
            }
            PoolConfigError::SlicedPoolsUnavailable => {
                write!(
                    f,
                    "sliced pools are not available in this build (exclusive memory only)"
                )
            }
        }
    }
}

impl MemoryConfiguration {
    /// Resolve a programmatic [`MemoryPoolsConfig`] override against the
    /// runtime-chosen configuration for the **main GPU** pool.
    ///
    /// When `pools` is `None`, the runtime's own `self` is kept unchanged;
    /// when present, it wins. There is deliberately no config-file pathway for
    /// pool layouts — they are dynamic (set per model just before a load) and
    /// must not freeze at startup; the override reaches the server through
    /// [`configure_memory_pools`](crate::client::ComputeClient::configure_memory_pools).
    ///
    /// `page_size` is deliberately not validated against
    /// [`MemoryDeviceProperties::max_page_size`]: that value is a sizing
    /// heuristic for the default layouts (CUDA/HIP report a quarter of the
    /// device memory), not an allocation limit, and a large arena is exactly
    /// what an explicit pool override is for. An unallocatable page fails at
    /// allocation time.
    pub fn resolve(
        self,
        pools: Option<&MemoryPoolsConfig>,
        properties: &MemoryDeviceProperties,
    ) -> Result<Self, PoolConfigError> {
        let Some(pools) = pools else {
            return Ok(self);
        };

        match pools {
            MemoryPoolsConfig::Preset(MemoryPoolsPreset::SubSlices) => {
                #[cfg(exclusive_memory_only)]
                {
                    Err(PoolConfigError::PresetUnavailable {
                        preset: "sub-slices",
                    })
                }
                #[cfg(not(exclusive_memory_only))]
                {
                    Ok(MemoryConfiguration::SubSlices)
                }
            }
            MemoryPoolsConfig::Preset(MemoryPoolsPreset::ExclusivePages) => {
                Ok(MemoryConfiguration::ExclusivePages)
            }
            MemoryPoolsConfig::Explicit(entries) => {
                if entries.is_empty() {
                    return Err(PoolConfigError::EmptyPoolList);
                }
                // Slices route through their pool's position, and the
                // persistent pool owns the sentinel position, so the list must
                // stay addressable below it — checked here so the caller gets
                // the error instead of a panic on the device thread.
                if entries.len() >= PERSISTENT_POOL_POS as usize {
                    return Err(PoolConfigError::TooManyPools {
                        count: entries.len(),
                    });
                }
                let pool_options = entries
                    .iter()
                    .map(|entry| pool_options_from_entry(entry, properties))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(MemoryConfiguration::Custom { pool_options })
            }
        }
    }
}

/// Convert one config entry into runtime pool options, aligning sizes up to
/// the device alignment (a device constraint, not a user error).
fn pool_options_from_entry(
    entry: &MemoryPoolConfig,
    properties: &MemoryDeviceProperties,
) -> Result<MemoryPoolOptions, PoolConfigError> {
    let alignment = properties.alignment.max(1);
    match entry {
        MemoryPoolConfig::Exclusive {
            max_alloc_size,
            dealloc_period,
        } => {
            // 0 stays 0: a pool dedicated to zero-sized allocations, as used by
            // the `SubSlices` preset.
            let max_alloc_size = max_alloc_size.bytes().next_multiple_of(alignment);
            Ok(MemoryPoolOptions {
                pool_type: PoolType::ExclusivePages { max_alloc_size },
                dealloc_period: *dealloc_period,
            })
        }
        // Sliced pools break the invariant `exclusive_memory_only` builds rely
        // on (e.g. wgpu on wasm assumes a buffer is never shared between
        // slices), so an explicit list must be rejected just like the
        // `sub-slices` preset is.
        #[cfg(exclusive_memory_only)]
        MemoryPoolConfig::Sliced { .. } => Err(PoolConfigError::SlicedPoolsUnavailable),
        #[cfg(not(exclusive_memory_only))]
        MemoryPoolConfig::Sliced {
            page_size,
            max_slice_size,
            max_pool_size,
            dealloc_period,
        } => {
            if page_size.bytes() == 0 {
                return Err(PoolConfigError::ZeroSize { field: "page_size" });
            }

            let page_size = page_size.bytes().next_multiple_of(alignment);
            let max_slice_size = match max_slice_size {
                Some(size) if size.bytes() == 0 => {
                    return Err(PoolConfigError::ZeroSize {
                        field: "max_slice_size",
                    });
                }
                Some(size) => size.bytes().next_multiple_of(alignment),
                None => page_size,
            };
            if max_slice_size > page_size {
                return Err(PoolConfigError::SliceLargerThanPage {
                    page_size,
                    max_slice_size,
                });
            }
            if let Some(cap) = max_pool_size {
                let cap = cap.bytes();
                if cap == 0 {
                    return Err(PoolConfigError::ZeroSize {
                        field: "max_pool_size",
                    });
                }
                if cap < page_size {
                    return Err(PoolConfigError::CapSmallerThanPage {
                        page_size,
                        max_pool_size: cap,
                    });
                }
                let pages = cap / page_size;
                if pages > u16::MAX as u64 {
                    return Err(PoolConfigError::TooManyPages { pages });
                }
            }

            Ok(MemoryPoolOptions {
                pool_type: PoolType::SlicedPages {
                    page_size,
                    max_slice_size,
                    max_pool_size: max_pool_size.map(|size| size.bytes()),
                },
                dealloc_period: *dealloc_period,
            })
        }
    }
}

/// The pool position stamped on persistent-pool slices, routing their binds
/// and lookups to the persistent pool. A fixed sentinel (rather than "one past
/// the dynamic pools") so live persistent slices stay routable when
/// [`MemoryManagement::configure`] rebuilds the dynamic pools with a
/// different count.
const PERSISTENT_POOL_POS: u8 = u8::MAX;

/// Build the dynamic pools for `config` — the shared core of
/// [`MemoryManagement::from_configuration`] and
/// [`MemoryManagement::configure`].
fn build_pools(
    properties: &MemoryDeviceProperties,
    config: MemoryConfiguration,
    logger: &Arc<ServerLogger>,
    name: &str,
) -> Vec<DynamicPool> {
    let pool_options = match config {
            #[cfg(not(exclusive_memory_only))]
            MemoryConfiguration::SubSlices => {
                // Round chunk size to be aligned.
                let memory_alignment = properties.alignment;
                let max_page = properties.max_page_size;
                let mut pools = Vec::new();

                const MB: u64 = 1024 * 1024;

                // Add in a pool for allocations that are smaller than the min alignment,
                // as they can't use offsets at all (on wgpu at least).
                pools.push(MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages { max_alloc_size: 0 },
                    dealloc_period: None,
                });

                let mut current = max_page;
                let mut max_sizes = vec![];
                let mut page_sizes = vec![];
                let mut base = pools.len() as u32;

                while current >= 32 * MB {
                    current /= 4;

                    // Make sure every pool has an aligned size.
                    current = current.next_multiple_of(memory_alignment);

                    max_sizes.push(current / 2u64.pow(base));
                    page_sizes.push(current);
                    base += 1;
                }

                max_sizes.reverse();
                page_sizes.reverse();

                for i in 0..max_sizes.len() {
                    let max = max_sizes[i];
                    let page_size = page_sizes[i];

                    pools.push(MemoryPoolOptions {
                        // Creating max slices lower than the chunk size reduces fragmentation.
                        pool_type: PoolType::SlicedPages {
                            page_size,
                            max_slice_size: max,
                            max_pool_size: None,
                        },
                        dealloc_period: None,
                    });
                }

                // Add pools from big to small.
                pools.push(MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size: max_page / memory_alignment * memory_alignment,
                        max_slice_size: max_page / memory_alignment * memory_alignment,
                        max_pool_size: None,
                    },
                    dealloc_period: None,
                });
                pools
            }
            MemoryConfiguration::ExclusivePages => {
                // Add all bin sizes. Nb: because of alignment some buckets
                // end up as the same size, so only want unique ones,
                // but also keep the order, so a BTree will do.
                const MIN_BUCKET_SIZE: u64 = 1024 * 32;
                const NUM_POOLS: usize = 24;

                let sizes = generate_bucket_sizes(
                    MIN_BUCKET_SIZE,
                    properties.max_page_size,
                    NUM_POOLS,
                    properties.alignment,
                );

                sizes
                    .iter()
                    .map(|&size| {
                        let dealloc_period = (BASE_DEALLOC_PERIOD as f64
                            * (1.0 + size as f64 / (DEALLOC_SCALE_MB as f64)).round())
                            as u64;

                        MemoryPoolOptions {
                            pool_type: PoolType::ExclusivePages {
                                max_alloc_size: size,
                            },
                            dealloc_period: Some(dealloc_period),
                        }
                    })
                    .collect()
            }
            MemoryConfiguration::Custom { pool_options } => pool_options,
        };

    logger.log_memory(
        |level| !matches!(level, MemoryLogLevel::Disabled),
        || {
            let mut msg = String::new();
            for pool in pool_options.iter() {
                msg += &format!("[{name}] Using memory pool: \n {pool:?}\n");
            }
            msg
        },
    );

    assert!(
        pool_options.len() < PERSISTENT_POOL_POS as usize,
        "at most {} dynamic pools are supported",
        PERSISTENT_POOL_POS
    );

    pool_options
        .iter()
        .enumerate()
        .map(|(pool_pos, pool)| {
            let pool_pos = pool_pos as u8;

            match pool.pool_type {
                PoolType::SlicedPages {
                    page_size,
                    max_slice_size,
                    max_pool_size,
                } => DynamicPool::Sliced(SlicedPool::new(
                    page_size,
                    max_slice_size,
                    properties.alignment,
                    pool_pos,
                    max_pool_size,
                )),
                PoolType::ExclusivePages { max_alloc_size } => {
                    DynamicPool::Exclusive(ExclusiveMemoryPool::new(
                        max_alloc_size,
                        properties.alignment,
                        pool.dealloc_period.unwrap_or(u64::MAX),
                        pool_pos,
                    ))
                }
            }
        })
        .collect()
}

impl<Storage: ComputeStorage> MemoryManagement<Storage> {
    /// Creates the options from device limits.
    pub fn from_configuration(
        storage: Storage,
        properties: &MemoryDeviceProperties,
        config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
        options: MemoryManagementOptions,
    ) -> Self {
        let pools = build_pools(properties, config, &logger, &options.name);

        let config = CubeClRuntimeConfig::get().memory.persistent_memory.clone();

        let mode = match options.memory {
            MemoryAllocationOption::Provided(mode) => mode,
            MemoryAllocationOption::FromConfig => match config {
                PersistentMemory::Enabled => MemoryAllocationMode::Auto,
                PersistentMemory::Disabled => MemoryAllocationMode::Auto,
                PersistentMemory::Enforced => MemoryAllocationMode::Persistent,
            },
        };

        Self {
            name: options.name,
            persistent: PersistentPool::new(
                properties.max_page_size,
                properties.alignment,
                PERSISTENT_POOL_POS,
            ),
            pools,
            storage,
            alloc_reserve_count: 0,
            mode,
            config,
            logger,
            capture: None,
        }
    }

    /// Rebuild the dynamic pools with a new layout, in place.
    ///
    /// The old pools are cleaned up first (every currently-free page returned
    /// to the driver). Rebuilding only happens when no live allocation remains
    /// in them — a live slice carries its pool position, so swapping the pool
    /// list under it would corrupt routing. When something is still alive, the
    /// old layout is kept and `false` is returned; the caller reconfigures at a
    /// quiescent point (e.g. right after unloading a model) so this is the
    /// exceptional path, not the normal one.
    ///
    /// The persistent pool is untouched: its slices route through a fixed
    /// sentinel position and its layout is model-agnostic.
    pub fn configure(
        &mut self,
        config: MemoryConfiguration,
        properties: &MemoryDeviceProperties,
    ) -> bool {
        self.cleanup(true);

        // Only the dynamic pools are rebuilt, so only their live slices block
        // (persistent usage — weights of another workload — doesn't).
        let dynamic_in_use: u64 = self
            .pools
            .iter()
            .map(|pool| match pool {
                DynamicPool::Sliced(p) => p.get_memory_usage().bytes_in_use,
                DynamicPool::Exclusive(p) => p.get_memory_usage().bytes_in_use,
            })
            .sum();
        if dynamic_in_use > 0 {
            self.logger.log_memory(
                |level| !matches!(level, MemoryLogLevel::Disabled),
                || {
                    format!(
                        "[{}] Keeping the current pool layout: {dynamic_in_use} bytes \
                         are still live in the dynamic pools",
                        self.name
                    )
                },
            );
            return false;
        }

        self.pools = build_pools(properties, config, &self.logger, &self.name);
        true
    }

    /// Begin a graph capture: force every allocation into the persistent pool
    /// — exact-fit slices with no bucket padding, which is what a graph's
    /// static shapes want — and start recording which slices the window hands
    /// out (see [`reserve`](Self::reserve)). Every slice the window touches
    /// belongs to the graph at [`capture_end`](Self::capture_end); anything it
    /// never touches (pre-existing live buffers, idle free slices) does not.
    /// Slices stay reusable *within* the window — warmup populates the pool, then
    /// the capture run reuses those slices without a fresh device allocation
    /// (illegal mid-capture). Sets the mode directly, overriding the config gate
    /// that [`mode`](Self::mode) honors. If a capture is already active, only the
    /// mode is re-forced — the original capture keeps its touched set and restore
    /// state.
    pub fn capture_begin(&mut self) {
        if self.capture.is_none() {
            self.capture = Some(CaptureState {
                restore_mode: self.mode,
                touched: HashSet::new(),
            });
        }
        self.mode = MemoryAllocationMode::Persistent;
    }

    /// End a graph capture: restore the previous allocation mode and return a
    /// retained handle to every persistent slice the window touched — exactly
    /// the memory the graph's recorded kernels replay against. The caller pins
    /// these on the graph so the pool never reuses graph memory (which a replay
    /// would corrupt); dropping the graph drops the handles and releases the
    /// slices. Slices the window never touched are left alone, so a pre-existing
    /// live buffer keeps its reuse and in-place (`can_mut`) semantics. Empty if
    /// no capture was active.
    pub fn capture_end(&mut self) -> Vec<ManagedMemoryHandle> {
        match self.capture.take() {
            Some(capture) => {
                self.mode = capture.restore_mode;
                self.persistent.retain_touched(&capture.touched)
            }
            None => Vec::new(),
        }
    }

    /// Change the mode of allocation.
    pub fn mode(&mut self, mode: MemoryAllocationMode) {
        // We override the mode based on the cubecl config.
        let mode = match self.config {
            PersistentMemory::Enabled => mode,
            PersistentMemory::Disabled | PersistentMemory::Enforced => return,
        };

        self.logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled),
            || {
                format!(
                    "[{}] Setting memory allocation mode: from {:?} => {mode:?}",
                    self.name, self.mode
                )
            },
        );

        // A capture owns the effective mode until it ends: changing it now
        // would route capture allocations away from the persistent pool. Defer
        // the change to `capture_end`.
        match &mut self.capture {
            Some(capture) => capture.restore_mode = mode,
            None => self.mode = mode,
        }
    }

    /// Cleanup allocations in pools that are deemed unnecessary.
    pub fn cleanup(&mut self, explicit: bool) {
        self.logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled) && explicit,
            || "Manual memory cleanup ...".to_string(),
        );

        // Nothing may be freed during a capture. The persistent window's free
        // slices are exactly what the capture run reuses (deallocating one
        // forces a fresh device allocation mid-capture, which faults), and
        // the storage frees behind the dynamic pools can synchronize the
        // device (e.g. `hipFree`), which invalidates the capture. Everything
        // stays queued until the capture ends.
        if self.capture.is_some() {
            return;
        }

        self.persistent
            .cleanup(&mut self.storage, self.alloc_reserve_count, explicit);

        for pool in self.pools.iter_mut() {
            pool.cleanup(&mut self.storage, self.alloc_reserve_count, explicit);
        }

        // The pools only queue their page deallocations in the storage; an
        // explicit cleanup means "release the memory now", so push them to the
        // driver instead of leaving them pending.
        if explicit {
            self.storage.flush();
        }
    }

    /// Returns the storage from the specified binding
    pub fn get_cursor(&self, binding: ManagedMemoryBinding) -> Result<u64, IoError> {
        let slice = self.find(binding)?;
        Ok(slice.cursor)
    }

    /// Returns the storage from the specified binding
    fn find(&self, binding: ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let id = binding.descriptor();

        if id.location().init == 0 {
            return Err(IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: "Memory location was never initialized".into(),
            });
        }

        let slice = if id.location().pool == PERSISTENT_POOL_POS {
            self.persistent.find(&binding)?
        } else {
            let pool =
                self.pools
                    .get(id.location().pool as usize)
                    .ok_or_else(|| IoError::NotFound {
                        backtrace: BackTrace::capture(),
                        reason: format!("Pool {} doesn't exist", id.location().pool).into(),
                    })?;

            pool.find(&binding)?
        };

        // A stale location (e.g. a page that was deallocated and whose index a
        // later cleanup reassigned) must surface as `NotFound`, never as another
        // allocation's slice.
        if slice.handle.descriptor() != binding.descriptor() {
            return Err(IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: "Memory location points to a different allocation".into(),
            });
        }

        Ok(slice)
    }

    /// Returns the storage from the specified binding
    pub fn get_storage(&mut self, binding: ManagedMemoryBinding) -> Result<StorageHandle, IoError> {
        let slice = self.find(binding)?;
        Ok(slice.storage.clone())
    }

    /// Returns the resource from the storage at the specified handle
    pub fn get_resource(
        &mut self,
        binding: ManagedMemoryBinding,
        offset_start: Option<u64>,
        offset_end: Option<u64>,
    ) -> Result<Storage::Resource, IoError> {
        let handle = self.get_storage(binding)?;

        let handle = match offset_start {
            Some(offset) => handle.offset_start(offset),
            None => handle,
        };
        let handle = match offset_end {
            Some(offset) => handle.offset_end(offset),
            None => handle,
        };
        Ok(self.storage().get(&handle))
    }

    /// Record a persistent slice as touched by the active capture window, so
    /// [`capture_end`](Self::capture_end) retains exactly the slices the window
    /// handed out. A no-op outside a capture.
    ///
    /// Called with a slice's **final** identity: from [`reserve`](Self::reserve)
    /// for a handle used as-is (e.g. pinned staging), and from [`bind`](Self::bind)
    /// for a buffer whose reserved handle is replaced by an assigned one. A
    /// reserved id later superseded by `bind` also lands here but harmlessly —
    /// ids are unique, so it matches no live slice at `capture_end`.
    fn capture_touch(&mut self, handle: &ManagedMemoryHandle) {
        if let Some(capture) = &mut self.capture {
            capture.touched.insert(handle.descriptor().id);
        }
    }

    /// Finds a spot in memory for a resource with the given size in bytes, and returns a handle to it
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        // If this happens every nanosecond, counts overflows after 585 years, so not worth thinking too
        // hard about overflow here.
        self.alloc_reserve_count += 1;

        if let Some(val) = self.persistent.try_reserve(size) {
            self.logger.log_memory(
                |level| matches!(level, MemoryLogLevel::Full),
                || {
                    format!(
                        "[{}] Reserved memory {size} using persistent memory",
                        self.name
                    )
                },
            );
            self.capture_touch(&val);
            return Ok(val);
        }

        if matches!(self.mode, MemoryAllocationMode::Persistent) || self.persistent.has_size(size) {
            let allocated = self.persistent.alloc(&mut self.storage, size);

            self.logger.log_memory(
                |level| !matches!(level, MemoryLogLevel::Disabled),
                || {
                    format!(
                        "[{}] Allocated a new memory page using persistent memory, \n{}",
                        self.name, self,
                    )
                },
            );
            if let Ok(handle) = &allocated {
                self.capture_touch(handle);
            }
            return allocated;
        }

        self.logger.log_memory(
            |level| matches!(level, MemoryLogLevel::Full),
            || {
                format!(
                    "[{}] Reserved memory {} using dynamic pool",
                    self.name,
                    BytesFormat::new(size)
                )
            },
        );

        // Find first pool that fits this allocation
        let pool = self
            .pools
            .iter_mut()
            .find(|p| p.accept(size))
            .ok_or(IoError::BufferTooBig {
                size,
                backtrace: BackTrace::capture(),
            })?;

        if let Some(slice) = pool.try_reserve(size) {
            return Ok(slice);
        }

        let allocated = pool.alloc(&mut self.storage, size);

        self.logger.log_memory(
            |level| matches!(level, MemoryLogLevel::Full),
            || {
                format!(
                    "[{}], Allocated a new memory page, current usage: \n{}",
                    self.name, self
                )
            },
        );

        allocated
    }

    /// Fetch the storage used by the memory manager.
    ///
    /// # Notes
    ///
    /// The storage should probably not be used for allocations since the handles won't be
    /// compatible with the ones provided by the current trait. Prefer using the
    /// [alloc](ComputeStorage::alloc) and [dealloc](ComputeStorage::dealloc) functions.
    ///
    /// This is useful if you need to time the deallocations based on async computation, or to
    /// change the mode of storage for different reasons.
    pub fn storage(&mut self) -> &mut Storage {
        &mut self.storage
    }

    /// Get the current memory usage.
    pub fn memory_usage(&self) -> MemoryUsage {
        let memory_usage = self.pools.iter().map(|x| x.get_memory_usage()).fold(
            MemoryUsage {
                number_allocs: 0,
                bytes_in_use: 0,
                bytes_padding: 0,
                bytes_reserved: 0,
            },
            |m1, m2| m1.combine(m2),
        );
        memory_usage.combine(self.persistent.get_memory_usage())
    }

    /// Print out a report of the current memory usage.
    pub fn print_memory_usage(&self) {
        #[cfg(feature = "std")]
        log::info!("{}", self.memory_usage());
    }

    /// Binds the given [handle](HandleId) to a [`MemorySlot`].
    pub fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
    ) -> Result<(), IoError> {
        let descriptor = reserved.descriptor();

        if descriptor.location().init == 0 {
            return Err(IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: "Reserved memory isn't initialized".into(),
            });
        }

        let pool_index = descriptor.location().pool as usize;
        if pool_index == PERSISTENT_POOL_POS as usize {
            // `bind` sets the slice's final identity to `assigned` (replacing the
            // throwaway reserved handle), so this — not the earlier `reserve` — is
            // the id a capture must track for a bound persistent buffer.
            self.capture_touch(&assigned);
            return self.persistent.bind(reserved, assigned, cursor);
        }

        self.pools
            .get_mut(pool_index)
            .map(|p| p.bind(reserved, assigned, cursor))
            .ok_or_else(|| IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: format!("Memory pool {} doesn't exist", pool_index).into(),
            })?
    }
}

impl<Storage: ComputeStorage> core::fmt::Display for MemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("\n# MemoryManagement\n\n")?;
        f.write_fmt(format_args!(" - name: {:?}\n", self.name))?;
        f.write_fmt(format_args!("\n## Persistent\n\n{}", self.persistent))?;
        f.write_str("\n## Dynamic\n\n")?;

        for pool in self.pools.iter() {
            match pool {
                DynamicPool::Sliced(pool) => f.write_fmt(format_args!("{pool}\n"))?,
                DynamicPool::Exclusive(pool) => f.write_fmt(format_args!("{pool}\n"))?,
            }
        }
        let memory_usage = self.memory_usage();
        f.write_fmt(format_args!("\n## Summary\n\n{memory_usage}"))?;

        Ok(())
    }
}

impl<Storage> core::fmt::Debug for MemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(
            alloc::format!(
                "DynamicMemoryManagement {:?}",
                core::any::type_name::<Storage>(),
            )
            .as_str(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{memory_management::MemoryManagement, storage::BytesStorage};
    use alloc::vec;

    const DUMMY_MEM_PROPS: MemoryDeviceProperties = MemoryDeviceProperties {
        max_page_size: 128 * 1024 * 1024,
        alignment: 32,
    };

    fn options() -> MemoryManagementOptions {
        MemoryManagementOptions {
            name: "test".into(),
            memory: MemoryAllocationOption::FromConfig,
        }
    }

    // Test pools with slices.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn test_handle_mutability() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::SubSlices,
            Arc::new(ServerLogger::default()),
            options(),
        );
        let handle = memory_management.reserve(10).unwrap();
        let other_ref = handle.clone();
        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    // Test pools with slices.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn test_memory_usage() {
        let max_page_size = 512;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: max_page_size,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        let handle = memory_management.reserve(100);
        let usage = memory_management.memory_usage();

        assert_eq!(usage.bytes_in_use, 100);
        assert!(usage.bytes_reserved >= 100 && usage.bytes_reserved <= max_page_size);

        // Drop and re-alloc.
        drop(handle);
        let _handle = memory_management.reserve(100);
        let usage_new = memory_management.memory_usage();
        assert_eq!(usage, usage_new);
    }

    #[test_log::test]
    fn find_uninit_binding_returns_not_found() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size: 2048,
                        max_slice_size: 2048,
                        max_pool_size: None,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        // Even with a live page at index 0, a never-initialized descriptor must
        // not resolve to it.
        let _live = memory_management.reserve(512).unwrap();

        let binding = ManagedMemoryHandle::new().binding();
        assert!(matches!(
            memory_management.get_cursor(binding),
            Err(IoError::NotFound { .. })
        ));
    }

    #[test_log::test]
    fn find_stale_descriptor_returns_not_found() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size: 2048,
                        max_slice_size: 2048,
                        max_pool_size: None,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let reserved = memory_management.reserve(512).unwrap();
        let stale = reserved.clone();
        let assigned = ManagedMemoryHandle::new();
        let assigned_binding = assigned.clone().binding();

        memory_management.bind(reserved, assigned, 0).unwrap();

        // The slice's identity is now `assigned`; the stale reserved descriptor
        // must surface as `NotFound`, not as the new allocation's data.
        assert!(matches!(
            memory_management.get_cursor(stale.binding()),
            Err(IoError::NotFound { .. })
        ));
        assert!(memory_management.get_cursor(assigned_binding).is_ok());
    }

    #[test_log::test]
    fn held_binding_survives_explicit_cleanup_renumber() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 1024,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let handle_a = memory_management.reserve(1024).unwrap();
        let handle_b = memory_management.reserve(1024).unwrap();
        let handle_c = memory_management.reserve(1024).unwrap();

        let binding_b = handle_b.binding();
        drop(handle_a);
        drop(handle_c);

        // Deallocates the two free pages and renumbers the surviving one.
        memory_management.cleanup(true);

        assert!(memory_management.get_cursor(binding_b.clone()).is_ok());
        assert!(memory_management.get_storage(binding_b).is_ok());
        assert_eq!(memory_management.memory_usage().bytes_reserved, 1024);
    }

    fn capped_sliced_config(page_size: u64, max_pool_size: Option<u64>) -> MemoryConfiguration {
        MemoryConfiguration::Custom {
            pool_options: vec![MemoryPoolOptions {
                pool_type: PoolType::SlicedPages {
                    page_size,
                    max_slice_size: page_size,
                    max_pool_size,
                },
                dealloc_period: None,
            }],
        }
    }

    #[test_log::test]
    fn capped_sliced_pool_errors_instead_of_growing() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            capped_sliced_config(1024, Some(2048)),
            Arc::new(ServerLogger::default()),
            options(),
        );

        let _a = memory_management.reserve(1024).unwrap();
        let _b = memory_management.reserve(1024).unwrap();

        let result = memory_management.reserve(1024);
        assert!(matches!(
            result,
            Err(IoError::PoolCapacityExceeded {
                capacity: 2048,
                ..
            })
        ));
        assert_eq!(
            memory_management.memory_usage().bytes_reserved,
            2048,
            "a failed reservation must not grow the pool"
        );
    }

    #[test_log::test]
    fn capped_sliced_pool_reuses_freed_memory() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            capped_sliced_config(1024, Some(2048)),
            Arc::new(ServerLogger::default()),
            options(),
        );

        let handle_a = memory_management.reserve(1024).unwrap();
        let _b = memory_management.reserve(1024).unwrap();
        drop(handle_a);

        // The capacity error is transient: freeing makes the reservation fit
        // again without growing the pool.
        let _c = memory_management.reserve(1024).unwrap();
        assert_eq!(memory_management.memory_usage().bytes_reserved, 2048);
    }

    #[test_log::test]
    fn capped_lazy_pool_cleanup_still_frees() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            capped_sliced_config(1024, Some(2048)),
            Arc::new(ServerLogger::default()),
            options(),
        );

        let handle_a = memory_management.reserve(1024).unwrap();
        let handle_b = memory_management.reserve(1024).unwrap();
        drop(handle_a);
        drop(handle_b);
        memory_management.cleanup(true);
        assert_eq!(memory_management.memory_usage().bytes_reserved, 0);

        // The cap is still enforced after the pool shrank and regrew.
        let _a = memory_management.reserve(1024).unwrap();
        let _b = memory_management.reserve(1024).unwrap();
        assert!(matches!(
            memory_management.reserve(1024),
            Err(IoError::PoolCapacityExceeded { .. })
        ));
    }

    #[test_log::test]
    fn max_pool_size_smaller_than_page_shrinks_page() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            capped_sliced_config(2048, Some(512)),
            Arc::new(ServerLogger::default()),
            options(),
        );

        let _small = memory_management.reserve(256).unwrap();
        assert!(memory_management.memory_usage().bytes_reserved <= 512);

        // Larger than the (shrunk) page: rejected without growing the footprint.
        assert!(memory_management.reserve(1024).is_err());
        assert!(memory_management.memory_usage().bytes_reserved <= 512);
    }

    #[test_log::test]
    fn max_pool_size_below_alignment_never_overshoots() {
        // A cap below the device alignment (32 in `DUMMY_MEM_PROPS`) can't fit
        // even the smallest page, so every reservation must error rather than
        // exceed the budget.
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            capped_sliced_config(1024, Some(16)),
            Arc::new(ServerLogger::default()),
            options(),
        );

        assert!(matches!(
            memory_management.reserve(8),
            Err(IoError::PoolCapacityExceeded { .. })
        ));
        assert_eq!(memory_management.memory_usage().bytes_reserved, 0);
    }

    #[test_log::test]
    fn capacity_error_does_not_fall_through_to_later_pool() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![
                    MemoryPoolOptions {
                        pool_type: PoolType::SlicedPages {
                            page_size: 1024,
                            max_slice_size: 1024,
                            max_pool_size: Some(1024),
                        },
                        dealloc_period: None,
                    },
                    MemoryPoolOptions {
                        pool_type: PoolType::SlicedPages {
                            page_size: 1024,
                            max_slice_size: 1024,
                            max_pool_size: None,
                        },
                        dealloc_period: None,
                    },
                ],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let _fill = memory_management.reserve(1024).unwrap();
        assert!(matches!(
            memory_management.reserve(1024),
            Err(IoError::PoolCapacityExceeded { .. })
        ));
        assert_eq!(
            memory_management.memory_usage().bytes_reserved,
            1024,
            "the overflow must not silently land in the later pool"
        );
    }

    #[test_log::test]
    fn alloc_two_chunks_on_one_page() {
        let page_size = 2048;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                        max_pool_size: None,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size);
    }

    #[test_log::test]
    fn alloc_reuses_storage() {
        // If no storage is re-used, this will allocate two pages.
        let page_size = 512;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                        max_pool_size: None,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        drop(_handle);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 1);
        assert_eq!(usage.bytes_in_use, alloc_size);
        assert_eq!(usage.bytes_reserved, page_size);
    }

    #[test_log::test]
    fn alloc_allocs_new_storage() {
        let page_size = 1024;

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                        max_pool_size: None,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 768;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size * 2);
    }

    #[test_log::test]
    fn alloc_respects_alignment_size() {
        let page_size = 500;
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: page_size,
                alignment: 50,
            },
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::SlicedPages {
                        page_size,
                        max_slice_size: page_size,
                        max_pool_size: None,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        let alloc_size = 40;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        // Each slice should be aligned to 50 bytes, so 20 padding bytes.
        assert_eq!(usage.bytes_padding, 10 * 2);
    }

    #[test_log::test]
    fn allocs_on_correct_page() {
        let sizes = [100, 200, 300, 400];

        let pools = sizes
            .iter()
            .map(|size| MemoryPoolOptions {
                pool_type: PoolType::SlicedPages {
                    page_size: *size,
                    max_slice_size: *size,
                    max_pool_size: None,
                },
                dealloc_period: None,
            })
            .collect();
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 10,
            },
            MemoryConfiguration::Custom {
                pool_options: pools,
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate one thing on each page.
        let alloc_sizes = [50, 150, 250, 350];
        let _handles = alloc_sizes.map(|s| memory_management.reserve(s));

        let usage = memory_management.memory_usage();

        // Total memory should be size of all pages, and no more.
        assert_eq!(usage.bytes_in_use, alloc_sizes.iter().sum::<u64>());
        assert!(usage.bytes_reserved >= sizes.iter().sum::<u64>());
    }

    #[test_log::test]
    fn resolve_absent_pools_config_keeps_runtime_choice() {
        #[cfg(not(exclusive_memory_only))]
        assert!(matches!(
            MemoryConfiguration::SubSlices
                .resolve(None, &DUMMY_MEM_PROPS)
                .unwrap(),
            MemoryConfiguration::SubSlices
        ));
        assert!(matches!(
            MemoryConfiguration::ExclusivePages
                .resolve(None, &DUMMY_MEM_PROPS)
                .unwrap(),
            MemoryConfiguration::ExclusivePages
        ));
    }

    #[test_log::test]
    fn resolve_preset_overrides_runtime_choice() {
        let preset = MemoryPoolsConfig::Preset(MemoryPoolsPreset::ExclusivePages);
        #[cfg(not(exclusive_memory_only))]
        let base = MemoryConfiguration::SubSlices;
        #[cfg(exclusive_memory_only)]
        let base = MemoryConfiguration::ExclusivePages;

        assert!(matches!(
            base.resolve(Some(&preset), &DUMMY_MEM_PROPS).unwrap(),
            MemoryConfiguration::ExclusivePages
        ));
    }

    #[test_log::test]
    #[cfg(exclusive_memory_only)]
    fn resolve_rejects_sliced_pools_when_exclusive_only() {
        use crate::config::size::MemorySize;

        let pools = MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
            page_size: MemorySize(1024),
            max_slice_size: None,
            max_pool_size: None,
            dealloc_period: None,
        }]);
        assert_eq!(
            MemoryConfiguration::default()
                .resolve(Some(&pools), &DUMMY_MEM_PROPS)
                .unwrap_err(),
            PoolConfigError::SlicedPoolsUnavailable
        );
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn resolve_explicit_list_aligns_and_defaults() {
        use crate::config::size::MemorySize;

        let pools = MemoryPoolsConfig::Explicit(vec![
            MemoryPoolConfig::Exclusive {
                max_alloc_size: MemorySize(8 * 1024),
                dealloc_period: Some(10000),
            },
            MemoryPoolConfig::Sliced {
                // Rounded up to the 32-byte alignment.
                page_size: MemorySize(1000),
                max_slice_size: None,
                max_pool_size: Some(MemorySize(4096)),
                dealloc_period: None,
            },
        ]);

        let resolved = MemoryConfiguration::default()
            .resolve(Some(&pools), &DUMMY_MEM_PROPS)
            .unwrap();
        let MemoryConfiguration::Custom { pool_options } = resolved else {
            panic!("expected a custom configuration");
        };

        assert_eq!(pool_options.len(), 2);
        assert!(matches!(
            pool_options[0].pool_type,
            PoolType::ExclusivePages {
                max_alloc_size: 8192
            }
        ));
        assert_eq!(pool_options[0].dealloc_period, Some(10000));
        assert!(matches!(
            pool_options[1].pool_type,
            PoolType::SlicedPages {
                page_size: 1024,
                // Defaults to the aligned page size.
                max_slice_size: 1024,
                max_pool_size: Some(4096),
            }
        ));
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn resolve_invalid_pool_configs_fail() {
        use crate::config::size::MemorySize;

        let cases = [
            (MemoryPoolsConfig::Explicit(vec![]), PoolConfigError::EmptyPoolList),
            (
                MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
                    page_size: MemorySize(0),
                    max_slice_size: None,
                    max_pool_size: None,
                    dealloc_period: None,
                }]),
                PoolConfigError::ZeroSize { field: "page_size" },
            ),
            (
                MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
                    page_size: MemorySize(1024),
                    max_slice_size: Some(MemorySize(2048)),
                    max_pool_size: None,
                    dealloc_period: None,
                }]),
                PoolConfigError::SliceLargerThanPage {
                    page_size: 1024,
                    max_slice_size: 2048,
                },
            ),
            (
                MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
                    page_size: MemorySize(2048),
                    max_slice_size: None,
                    max_pool_size: Some(MemorySize(1024)),
                    dealloc_period: None,
                }]),
                PoolConfigError::CapSmallerThanPage {
                    page_size: 2048,
                    max_pool_size: 1024,
                },
            ),
            (
                MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
                    page_size: MemorySize(1024),
                    max_slice_size: None,
                    // 2^26 pages of 1 KiB: far beyond the u16 page index.
                    max_pool_size: Some(MemorySize(64 * 1024 * 1024 * 1024)),
                    dealloc_period: None,
                }]),
                PoolConfigError::TooManyPages { pages: 64 * 1024 * 1024 },
            ),
            (
                MemoryPoolsConfig::Explicit(vec![
                    MemoryPoolConfig::Exclusive {
                        max_alloc_size: MemorySize(1024),
                        dealloc_period: None,
                    };
                    PERSISTENT_POOL_POS as usize
                ]),
                PoolConfigError::TooManyPools {
                    count: PERSISTENT_POOL_POS as usize,
                },
            ),
        ];

        for (pools, expected) in cases {
            let result = MemoryConfiguration::default().resolve(Some(&pools), &DUMMY_MEM_PROPS);
            assert_eq!(result.unwrap_err(), expected);
        }
    }

    // The motivating use case: allocations from different "sequence-length
    // ranges" reuse the same arena instead of each landing in its own
    // size-bucketed pool that keeps a separate reservation.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn resolved_single_arena_reuses_across_sizes() {
        use crate::config::size::MemorySize;

        let page = 1024 * 1024; // 1 MiB arena.
        let pools = MemoryPoolsConfig::Explicit(vec![MemoryPoolConfig::Sliced {
            page_size: MemorySize(page),
            max_slice_size: None,
            max_pool_size: None,
            dealloc_period: None,
        }]);
        let config = MemoryConfiguration::default()
            .resolve(Some(&pools), &DUMMY_MEM_PROPS)
            .unwrap();

        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            config,
            Arc::new(ServerLogger::default()),
            options(),
        );

        // A "small seq" allocation, then freed.
        let small = memory_management.reserve(4 * 1024).unwrap();
        drop(small);
        // A "large seq" allocation must reuse the same arena page.
        let large = memory_management.reserve(512 * 1024).unwrap();

        let usage = memory_management.memory_usage();
        assert_eq!(
            usage.bytes_reserved, page,
            "both sizes must share a single arena page"
        );
        assert_eq!(usage.number_allocs, 1);
        drop(large);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn allocate_deallocate_reallocate() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            },
            MemoryConfiguration::SubSlices,
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_after = memory_management.memory_usage();
        assert_eq!(usage_before.number_allocs, usage_after.number_allocs);
        assert_eq!(usage_before.bytes_in_use, usage_after.bytes_in_use);
        // Usage after can actually be _less_ because of defragging.
        assert!(usage_before.bytes_reserved >= usage_after.bytes_reserved);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn test_fragmentation_resistance() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            },
            MemoryConfiguration::SubSlices,
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate a mix of small and large chunks
        let sizes = [50, 1000, 100, 5000, 200, 10000, 300];
        let handles: Vec<_> = sizes
            .iter()
            .map(|&size| memory_management.reserve(size).unwrap())
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate every other allocation
        for i in (0..handles.len()).step_by(2) {
            drop(handles[i].clone());
        }
        // Reallocate similar sizes
        for &size in &sizes[0..sizes.len() / 2] {
            memory_management.reserve(size).unwrap();
        }
        let usage_after = memory_management.memory_usage();
        // Check that we haven't increased our memory usage significantly
        assert!(usage_after.bytes_reserved <= (usage_before.bytes_reserved as f64 * 1.1) as u64);
    }

    // Test pools without slices. More or less same as tests above.
    #[test_log::test]
    fn noslice_test_handle_mutability() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &(MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            }),
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );
        let handle = memory_management.reserve(10).unwrap();
        let other_ref = handle.clone();
        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    #[test_log::test]
    fn noslice_alloc_two_chunk() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 1024,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert!(usage.bytes_reserved >= alloc_size * 2);
    }

    #[test_log::test]
    fn noslice_alloc_reuses_storage() {
        // If no storage is re-used, this will allocate two pages.
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 1024,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size);
        drop(_handle);
        let _new_handle = memory_management.reserve(alloc_size);

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 1);
        assert_eq!(usage.bytes_in_use, alloc_size);
        assert!(usage.bytes_reserved >= alloc_size);
    }

    #[test_log::test]
    fn noslice_alloc_allocs_new_storage() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 1024,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 768;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert!(usage.bytes_reserved >= alloc_size * 2);
    }

    #[test_log::test]
    fn noslice_alloc_respects_alignment_size() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: DUMMY_MEM_PROPS.max_page_size,
                alignment: 50,
            },
            MemoryConfiguration::Custom {
                pool_options: vec![MemoryPoolOptions {
                    pool_type: PoolType::ExclusivePages {
                        max_alloc_size: 50 * 20,
                    },
                    dealloc_period: None,
                }],
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        let alloc_size = 40;
        let _handle = memory_management.reserve(alloc_size);
        let _new_handle = memory_management.reserve(alloc_size);
        let usage = memory_management.memory_usage();
        // Each slice should be aligned to 60 bytes, so 20 padding bytes.
        assert_eq!(usage.bytes_padding, 10 * 2);
    }

    #[test_log::test]
    fn noslice_allocs_on_correct_page() {
        let pools = [100, 200, 300, 400]
            .iter()
            .map(|&size| MemoryPoolOptions {
                pool_type: PoolType::SlicedPages {
                    page_size: size,
                    max_slice_size: size,
                    max_pool_size: None,
                },
                dealloc_period: None,
            })
            .collect();
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: DUMMY_MEM_PROPS.max_page_size,
                alignment: 10,
            },
            MemoryConfiguration::Custom {
                pool_options: pools,
            },
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate one thing on each page.
        let alloc_sizes = [50, 150, 250, 350];
        let _handles = alloc_sizes.map(|s| memory_management.reserve(s));
        let usage = memory_management.memory_usage();
        // Total memory should be size of all pages, and no more.
        assert_eq!(usage.bytes_in_use, alloc_sizes.iter().sum::<u64>());
    }

    #[test_log::test]
    fn capture_pins_reused_persistent_slice() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        // First capture allocates a persistent slice, then everything is freed.
        memory_management.capture_begin();
        let first = memory_management.reserve(1024).unwrap();
        drop(first);
        drop(memory_management.capture_end());

        // A second capture reuses that now-free slice: the reuse must be pinned
        // even though the slice predates the capture.
        memory_management.capture_begin();
        let second = memory_management.reserve(1024).unwrap();
        drop(second);
        let pins = memory_management.capture_end();
        assert_eq!(pins.len(), 1, "the reused slice must be retained");

        // While pinned, the pool must not hand the slice to a later allocation.
        let before = memory_management.memory_usage();
        let _other = memory_management.reserve(1024).unwrap();
        let after = memory_management.memory_usage();
        assert!(
            after.bytes_reserved > before.bytes_reserved,
            "a pinned slice was handed to a later allocation"
        );
    }

    #[test_log::test]
    fn capture_pins_preexisting_slice_freed_and_reused_midwindow() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        // A persistent slice that is live (in use) when the next window opens.
        memory_management.capture_begin();
        let live = memory_management.reserve(1024).unwrap();
        drop(memory_management.capture_end()); // release the pin; `live` still holds the slice.

        // The window opens with `live`'s slice in use, then frees it mid-window
        // and reuses that exact slice for a window allocation the graph records
        // against. The old snapshot-of-in-use heuristic excluded it (it was in
        // use at begin); reservation-tracking pins it because the window touched
        // it — the whole point of the redesign.
        memory_management.capture_begin();
        drop(live);
        let reused = memory_management.reserve(1024).unwrap();
        drop(reused);
        let pins = memory_management.capture_end();
        assert_eq!(
            pins.len(),
            1,
            "a pre-existing slice freed and reused mid-window must be pinned"
        );
    }

    #[test_log::test]
    fn capture_does_not_retain_untouched_free_slices() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        // Leave an idle free slice in the pool from an earlier capture.
        memory_management.capture_begin();
        let earlier = memory_management.reserve(1024).unwrap();
        drop(earlier);
        drop(memory_management.capture_end());

        // A new capture that only ever touches a different size must not retain
        // that leftover idle slice — reservation-tracking pins exactly what the
        // window used, so no free-slice cleanup at `capture_begin` is needed.
        memory_management.capture_begin();
        let window = memory_management.reserve(2048).unwrap();
        drop(window);
        let pins = memory_management.capture_end();
        assert_eq!(
            pins.len(),
            1,
            "only the touched slice is retained, not the idle leftover"
        );
    }

    #[test_log::test]
    fn capture_survives_explicit_cleanup() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        memory_management.capture_begin();
        let handle = memory_management.reserve(1024).unwrap();
        drop(handle);
        // An explicit cleanup mid-capture compacts the persistent pool; the
        // capture must keep its pins through the rebuild.
        memory_management.cleanup(true);
        let pins = memory_management.capture_end();
        assert_eq!(pins.len(), 1, "pin lost across an explicit cleanup");
    }

    #[test_log::test]
    fn capture_begin_is_reentrant() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        memory_management.capture_begin();
        let first = memory_management.reserve(1024).unwrap();
        // A second begin (defensive: callers arm a capture exactly once) must
        // not discard the pins or the saved mode of the capture already in flight.
        memory_management.capture_begin();
        let second = memory_management.reserve(2048).unwrap();
        drop(first);
        drop(second);
        let pins = memory_management.capture_end();
        assert_eq!(pins.len(), 2, "pins from before the re-entrant begin lost");
        assert!(
            memory_management.capture_end().is_empty(),
            "capture must be fully disarmed"
        );
    }

    #[test_log::test]
    fn capture_leaves_preexisting_buffers_alone() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        // A persistent buffer that predates the capture and stays alive
        // through it (weights, a graph input created earlier).
        memory_management.capture_begin();
        let preexisting = memory_management.reserve(1024).unwrap();
        drop(memory_management.capture_end());

        memory_management.capture_begin();
        let window = memory_management.reserve(2048).unwrap();
        drop(window);
        let pins = memory_management.capture_end();

        // Only the window's slice is claimed; the pre-existing buffer keeps a
        // single user reference, so in-place ops on it keep working.
        assert_eq!(
            pins.len(),
            1,
            "only the window's slice belongs to the graph"
        );
        assert!(
            preexisting.can_mut(),
            "a capture must not claim pre-existing live buffers"
        );
    }

    #[test_log::test]
    fn noslice_allocate_deallocate_reallocate() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties {
                max_page_size: 128 * 1024 * 1024,
                alignment: 32,
            },
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1)))
            .collect();
        let usage_after = memory_management.memory_usage();
        assert_eq!(usage_before.number_allocs, usage_after.number_allocs);
        assert_eq!(usage_before.bytes_in_use, usage_after.bytes_in_use);
        assert_eq!(usage_before.bytes_reserved, usage_after.bytes_reserved);
    }
}
