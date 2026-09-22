use super::{
    DEDICATED_POOL_POS, ManagedMemoryBinding, ManagedMemoryHandle, ManagedMemoryId,
    MemoryAllocationMode, MemoryConfiguration, MemoryReport, MemoryUsage, PERSISTENT_POOL_POS,
    memory_pool::{DirectPool, MemoryPool, PageMapping, PersistentPool},
};
use crate::{
    config::{
        CubeClRuntimeConfig, RuntimeConfig,
        memory::{MemoryLogLevel, PersistentMemory},
    },
    logging::ServerLogger,
    memory_management::{BytesFormat, DynamicMemory, ErrorGraph, FailureId, memory_pool::Slice},
    server::IoError,
    storage::{ComputeStorage, StorageHandle},
};

use crate::memory_management::relocation::{Landed, PlannerId, Relocation};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::ops::Range;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashSet;
use cubecl_environment::sync::Arc;
use cubecl_ir::MemoryDeviceProperties;

/// Which pool a slice's location routes to: the two fixed sentinels, or a
/// dynamic pool by index.
#[derive(Clone, Copy)]
enum PoolPosition {
    Persistent,
    Dedicated,
    Dynamic(usize),
}

impl PoolPosition {
    fn new(pool: u8) -> Self {
        match pool {
            PERSISTENT_POOL_POS => PoolPosition::Persistent,
            DEDICATED_POOL_POS => PoolPosition::Dedicated,
            index => PoolPosition::Dynamic(index as usize),
        }
    }

    /// The error for a location naming a dynamic pool the layout does not have.
    fn missing(index: usize) -> IoError {
        IoError::NotFound {
            backtrace: BackTrace::capture(),
            reason: format!("Memory pool {index} doesn't exist").into(),
        }
    }
}

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct MemoryManagement<Storage> {
    name: String,
    persistent: PersistentPool,
    /// Allocations made under [`MemoryAllocationMode::Dedicated`]: each its own
    /// device allocation, returned to the driver on the tick after it is freed.
    dedicated: DirectPool,
    pools: DynamicMemory,
    /// What a relocation this one planned is stamped with, so no other
    /// commits it.
    planner: PlannerId,
    storage: Storage,
    alloc_reserve_count: u64,
    mode: MemoryAllocationMode,
    /// The mode no window overrides: the configured or provided one.
    base_mode: MemoryAllocationMode,
    /// Open allocation windows (see [`mode`](Self::mode)), innermost last,
    /// each holding the mode it put in force.
    windows: Vec<MemoryAllocationMode>,
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
    /// Whether the warmup (priming) phase is still running, i.e. the capture window has not opened
    /// yet. While set, every slice handed out is retained in `primed` instead of being recycled.
    priming: bool,
    /// Slices retained during priming, released by
    /// [`capture_priming_end`](MemoryManagement::capture_priming_end).
    ///
    /// Warmup exists to leave the pool able to serve the recorded run without allocating — an
    /// allocation inside the window is recorded as a memory node, and CUDA refuses to relaunch a
    /// graph holding one. Letting warmup recycle its own slices defeats that: the pool only ever
    /// grows to a warmup pass's transient *peak*, which depends on how far the host runs ahead of
    /// the device and can land below what the recorded run asks for. Holding every slice instead
    /// forces the pool up to the pass's full distinct working set — an upper bound on any peak —
    /// so once these are released the recorded run cannot ask for a slice the pool lacks.
    primed: Vec<ManagedMemoryHandle>,
}

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

impl<Storage: ComputeStorage> MemoryManagement<Storage> {
    /// Creates the options from device limits.
    pub fn from_configuration(
        storage: Storage,
        properties: &MemoryDeviceProperties,
        config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
        options: MemoryManagementOptions,
    ) -> Self {
        let pools = DynamicMemory::new(properties, config, logger.clone(), options.name.clone());

        let config = CubeClRuntimeConfig::get().memory.persistent_memory.clone();

        let mode = match options.memory {
            MemoryAllocationOption::Provided(mode) => mode,
            MemoryAllocationOption::FromConfig => match config {
                PersistentMemory::Enabled | PersistentMemory::SizeMatch => {
                    MemoryAllocationMode::Auto
                }
                PersistentMemory::Disabled => MemoryAllocationMode::Auto,
                PersistentMemory::Enforced => MemoryAllocationMode::Persistent,
            },
        };

        Self {
            name: options.name,
            planner: PlannerId::new(),
            persistent: PersistentPool::new(
                properties.max_page_size,
                properties.alignment,
                PERSISTENT_POOL_POS,
            ),
            // A watermark of zero: every free slice goes back on the next tick.
            dedicated: DirectPool::new(properties.alignment, DEDICATED_POOL_POS, Some(0)),
            pools,
            storage,
            alloc_reserve_count: 0,
            mode,
            base_mode: mode,
            windows: Vec::new(),
            config,
            logger,
            capture: None,
        }
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
                priming: true,
                primed: Vec::new(),
            });
        }
        self.mode = MemoryAllocationMode::Persistent;
    }

    /// End the priming phase and release the slices warmup retained, returning them to the pool as
    /// free. Call immediately before the capture window opens.
    ///
    /// After this the pool holds every slice a warmup pass touched, all of them free, so the
    /// recorded run reuses them instead of growing the pool (see [`CaptureState::primed`]). No-op
    /// when no capture is active or priming already ended.
    pub fn capture_priming_end(&mut self) {
        if let Some(capture) = &mut self.capture {
            capture.priming = false;
            // Dropping the handles makes the slices free again; the slices themselves stay in the
            // pool, which is the point.
            capture.primed.clear();
        }
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
    ///
    /// Windows **nest**: a `Persistent` or `Dedicated` call opens one, an
    /// `Auto` call closes the innermost, and the innermost open window decides
    /// the effective mode. Callers routinely nest without knowing it — a
    /// module load opens a persistent window around the whole load while the
    /// parameter machinery underneath opens one per parameter — and without
    /// the stack, the first inner window's exit would flip the rest of the
    /// outer window back to `Auto`: weights landing in the dynamic pools,
    /// where every page they sit on is held for the model's whole life.
    ///
    /// The persistent-memory config decides what a `Persistent` window puts in
    /// force (`Disabled` and `Enforced` keep the configured mode); a
    /// `Dedicated` window is honored whatever the config, since it exists to
    /// keep a buffer out of every pool.
    pub fn mode(&mut self, mode: MemoryAllocationMode) {
        match mode {
            MemoryAllocationMode::Auto => {
                self.windows.pop();
            }
            MemoryAllocationMode::Persistent => self.windows.push(match self.config {
                PersistentMemory::Enabled | PersistentMemory::SizeMatch => mode,
                PersistentMemory::Disabled | PersistentMemory::Enforced => self.base_mode,
            }),
            MemoryAllocationMode::Dedicated => self.windows.push(mode),
        }
        let mode = self.windows.last().copied().unwrap_or(self.base_mode);

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
    pub fn cleanup(&mut self, explicit: bool, failures: &mut ErrorGraph) {
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

        self.persistent.cleanup(
            &mut self.storage,
            self.alloc_reserve_count,
            explicit,
            failures,
        );

        // Dedicated buffers never wait for an explicit cleanup: freed is done.
        if self.dedicated.reclaim(&mut self.storage, failures) {
            self.storage.flush();
        }

        self.pools.cleanup(
            &mut self.storage,
            self.alloc_reserve_count,
            explicit,
            failures,
        );

        // The pools only queue their page deallocations in the storage; an
        // explicit cleanup means "release the memory now", so push them to the
        // driver instead of leaving them pending.
        if explicit {
            self.storage.flush();
        }
    }

    /// Returns the storage from the specified binding
    pub fn get_cursor(&self, binding: ManagedMemoryBinding) -> Result<u64, IoError> {
        let slice = self.find(&binding)?;
        Ok(slice.cursor)
    }

    /// Returns the storage from the specified binding
    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let id = binding.descriptor();

        if !id.is_allocated() {
            return Err(IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: "Memory location was never initialized".into(),
            });
        }

        let slice = match PoolPosition::new(id.location().pool) {
            PoolPosition::Persistent => self.persistent.find(binding)?,
            PoolPosition::Dedicated => self.dedicated.find(binding)?,
            PoolPosition::Dynamic(index) => self
                .pools
                .get(index)
                .ok_or_else(|| PoolPosition::missing(index))?
                .find(binding)?,
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

    /// [`find`](Self::find), mutably — the path [`taint`](Self::taint) and
    /// [`written`](Self::written) take to reach the slice.
    fn find_mut(&mut self, binding: &ManagedMemoryBinding) -> Result<&mut Slice, IoError> {
        let id = binding.descriptor();

        if !id.is_allocated() {
            return Err(IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: "Memory location was never initialized".into(),
            });
        }

        let slice = match PoolPosition::new(id.location().pool) {
            PoolPosition::Persistent => self.persistent.find_mut(binding)?,
            PoolPosition::Dedicated => self.dedicated.find_mut(binding)?,
            PoolPosition::Dynamic(index) => self
                .pools
                .get_mut(index)
                .ok_or_else(|| PoolPosition::missing(index))?
                .find_mut(binding)?,
        };

        // The same stale-location rule as `find`.
        if slice.handle.descriptor() != binding.descriptor() {
            return Err(IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: "Memory location points to a different allocation".into(),
            });
        }

        Ok(slice)
    }

    /// Returns the storage from the specified binding.
    ///
    /// This is the funnel every buffer dereference passes through
    /// ([`get_resource`](Self::get_resource) delegates here), so it is where
    /// a lazily-carved allocation gets its real device backing: the handle
    /// returned always refers to mapped memory.
    pub fn get_storage(&mut self, binding: ManagedMemoryBinding) -> Result<StorageHandle, IoError> {
        self.materialize(&binding)?;
        let slice = self.find(&binding)?;
        Ok(slice.storage.clone())
    }

    /// Keep the allocation behind `binding` where it is until it is freed: a
    /// graph being recorded resolved it, and replays against the address it
    /// resolved. Marked by whoever knows a recording is open — the capturing
    /// stream need not be the one that owns the allocation.
    pub fn mark_captured(&mut self, binding: &ManagedMemoryBinding) {
        if let Ok(slice) = self.find_mut(binding) {
            slice.captured = true;
        }
    }

    /// Plan moving what is still live on the sliced pools' outdated pages onto
    /// pages of the current size, reserving a target slice for each.
    ///
    /// Nothing moves until the returned relocation's copies have landed and it
    /// is [committed](Self::commit_relocation); dropping it abandons it with
    /// nothing lost.
    ///
    /// Empty during a capture, where nothing may move or be freed.
    pub fn plan_relocation(&mut self, failures: &mut ErrorGraph) -> Relocation {
        if self.capture.is_some() {
            return Relocation::new(self.planner, Vec::new());
        }
        Relocation::new(
            self.planner,
            self.pools.plan_relocation(&mut self.storage, failures),
        )
    }

    /// Hand every allocation of a landed relocation over to the slice that
    /// now holds its bytes, then return the pages that left empty to the
    /// driver.
    pub fn commit_relocation(&mut self, relocation: Landed, failures: &mut ErrorGraph) {
        for relocated in relocation.into_moves(self.planner) {
            self.pools.commit_relocation(relocated, failures);
        }
        self.cleanup(true, failures);
    }

    /// Install real backing behind `binding` when its allocation was carved
    /// lazily under a dry run. Lookup errors are left for
    /// [`find`](Self::find) to report with its usual diagnostics.
    fn materialize(&mut self, binding: &ManagedMemoryBinding) -> Result<(), IoError> {
        let location = binding.descriptor().location();
        if location.init == 0 {
            return Ok(());
        }
        match PoolPosition::new(location.pool) {
            PoolPosition::Persistent => self.persistent.materialize(&mut self.storage, binding),
            PoolPosition::Dedicated => self.dedicated.materialize(&mut self.storage, binding),
            PoolPosition::Dynamic(index) => match self.pools.get_mut(index) {
                Some(pool) => pool.materialize(&mut self.storage, binding),
                None => Ok(()),
            },
        }
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
        self.storage().get(&handle)
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
            if capture.priming {
                // Retain it so warmup cannot recycle this slice, forcing the pool to grow to the
                // pass's full working set rather than its transient peak.
                capture.primed.push(handle.clone());
            }
        }
    }

    /// Finds a spot in memory for a resource with the given size in bytes, and returns a handle to it
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip_all))]
    pub fn reserve(
        &mut self,
        size: u64,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        // If this happens every nanosecond, counts overflows after 585 years, so not worth thinking too
        // hard about overflow here.
        self.alloc_reserve_count += 1;

        // Drive the pools' periodic deallocation. Each pool gates itself on
        // its own `dealloc_period` (pools without one no-op), so this is a few
        // comparisons per reservation — without it, pages freed long ago are
        // never returned to the driver until an explicit cleanup, which on
        // long-running processes lets every stream's pools grow monotonically.
        self.cleanup(false, failures);

        let mapping = PageMapping::current();

        if matches!(self.mode, MemoryAllocationMode::Dedicated) {
            return self
                .dedicated
                .alloc(&mut self.storage, size, mapping, failures);
        }

        // In an explicit persistent window the pool always serves the
        // allocation (reusing a freed same-size slice when one exists).
        // Outside a window, the pool participates only under the `size-match`
        // config: recurring weight-shaped allocations reuse the buckets — a
        // training-friendly heuristic that would otherwise pull inference
        // activations into exact-sized persistent slices.
        let persistent_mode = matches!(self.mode, MemoryAllocationMode::Persistent);
        let size_match = matches!(self.config, PersistentMemory::SizeMatch);

        if (persistent_mode || size_match)
            && let Some(val) = self.persistent.try_reserve(size, failures)
        {
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

        if persistent_mode || (size_match && self.persistent.has_size(size)) {
            let allocated = self
                .persistent
                .alloc(&mut self.storage, size, mapping, failures);

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

        // Serve from the first pool that accepts this size and has capacity. A
        // hard-capped pool that is full falls through to the next accepting
        // pool instead of failing outright, so a growable tail pool can act as
        // an escape hatch behind a measured arena. Deliberate: a cap is a plan,
        // and a plan that turns out to be short should cost memory, not kill
        // the workload. Where the cap is a hard budget rather than a plan,
        // configure no pool behind it — then a full pool still errors, which is
        // what keeps the budget-vs-device-OOM distinction schedulers rely on.
        let reserved = self
            .pools
            .reserve(&mut self.storage, size, mapping, failures)?;

        self.logger.log_memory(
            |level| matches!(level, MemoryLogLevel::Full),
            || {
                format!(
                    "[{}], Allocated a new memory page, current usage: \n{}",
                    self.name, self
                )
            },
        );

        Ok(reserved)
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
        let memory_usage = core::iter::once(self.dedicated.get_memory_usage())
            .chain(core::iter::once(self.pools.memory_usage()))
            .fold(
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

    /// A structured per-pool report: each pool's shape, usage, and high-water
    /// marks, in allocation-routing order.
    ///
    /// The read side of a measured memory plan — the cycle, and what the
    /// marks cover, is on [`MemoryReport`].
    pub fn memory_report(&self) -> MemoryReport {
        MemoryReport {
            dynamic: self.pools.report(),
            persistent: self.persistent.report(),
        }
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
        failures: &mut ErrorGraph,
    ) -> Result<(), IoError> {
        let descriptor = reserved.descriptor();

        if !descriptor.is_allocated() {
            return Err(IoError::NotFound {
                backtrace: BackTrace::capture(),
                reason: "Reserved memory isn't initialized".into(),
            });
        }

        match PoolPosition::new(descriptor.location().pool) {
            PoolPosition::Persistent => {
                // `bind` sets the slice's final identity to `assigned` (replacing
                // the throwaway reserved handle), so this — not the earlier
                // `reserve` — is the id a capture must track for a bound
                // persistent buffer.
                self.capture_touch(&assigned);
                self.persistent.bind(reserved, assigned, cursor, failures)
            }
            // A capture forces every allocation persistent, so a dedicated one
            // is never in a window to touch.
            PoolPosition::Dedicated => self.dedicated.bind(reserved, assigned, cursor, failures),
            PoolPosition::Dynamic(index) => self
                .pools
                .get_mut(index)
                .ok_or_else(|| PoolPosition::missing(index))?
                .bind(reserved, assigned, cursor, failures),
        }
    }

    /// The failure claiming any byte of `range` in the allocation behind
    /// `binding`, if one does.
    ///
    /// This is the read path's whole check: a field on a slice the resolution
    /// walks anyway. A binding this manager does not hold answers `None` —
    /// lookup errors are [`find`](Self::find)'s to report, and a failed lookup
    /// carries no failure to name.
    pub fn failure(&self, binding: &ManagedMemoryBinding, range: Range<u64>) -> Option<FailureId> {
        self.find(binding)
            .ok()
            .and_then(|slice| slice.tainted.failure(&range))
    }

    /// Point `range` of the allocation behind `binding` at `failure`.
    ///
    /// A binding this manager does not hold is left alone, for the same
    /// reason [`failure`](Self::failure) answers `None` for one.
    pub fn taint(
        &mut self,
        binding: &ManagedMemoryBinding,
        range: Range<u64>,
        failure: FailureId,
        failures: &mut ErrorGraph,
    ) {
        if let Ok(slice) = self.find_mut(binding) {
            slice.tainted.taint(range, failure, failures);
        }
    }

    /// `range` of the allocation behind `binding` has a writer again: release
    /// every claim on those bytes — and only those bytes, since a partial
    /// write says nothing about the rest of the buffer.
    pub fn written(
        &mut self,
        binding: &ManagedMemoryBinding,
        range: Range<u64>,
        failures: &mut ErrorGraph,
    ) {
        if let Ok(slice) = self.find_mut(binding) {
            slice.tainted.written(range, failures);
        }
    }
}

impl<Storage: ComputeStorage> core::fmt::Display for MemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("\n# MemoryManagement\n\n")?;
        f.write_fmt(format_args!(" - name: {:?}\n", self.name))?;
        f.write_fmt(format_args!("\n## Persistent\n\n{}", self.persistent))?;
        f.write_str("\n## Dynamic\n\n")?;

        f.write_fmt(format_args!("{}", self.pools))?;
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
    #[cfg(not(exclusive_memory_only))]
    use crate::memory_management::MemoryPoolKind;
    use crate::{memory_management::MemoryManagement, storage::BytesStorage};

    const MIB: u64 = 1024 * 1024;
    const DUMMY_MEM_PROPS: MemoryDeviceProperties = MemoryDeviceProperties::new(128 * MIB, 32);
    /// The page the `Adaptive` preset's metadata pool carves.
    #[cfg(not(exclusive_memory_only))]
    const METADATA_PAGE: u64 = 8 * MIB;

    fn options() -> MemoryManagementOptions {
        MemoryManagementOptions {
            name: "test".into(),
            memory: MemoryAllocationOption::FromConfig,
        }
    }

    /// The adaptive preset's small pool is for metadata: an allocation close
    /// to its page size still goes to the adaptive pool, which sizes pages to
    /// what it serves, rather than onto the metadata pages.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn adaptive_preset_routes_near_page_sizes_to_the_adaptive_pool() {
        let mut memory = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );
        let _near = memory.reserve(7 * MIB, &mut ErrorGraph::default()).unwrap();

        let served = memory
            .memory_report()
            .dynamic
            .into_iter()
            .find(|pool| pool.largest_alloc == 7 * MIB)
            .expect("some pool served the allocation");
        assert!(
            matches!(served.kind, MemoryPoolKind::Adaptive { .. }),
            "served by {:?}",
            served.kind
        );
    }

    // Test pools with slices.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn test_handle_mutability() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );
        let handle = memory_management
            .reserve(10, &mut ErrorGraph::default())
            .unwrap();
        let other_ref = handle.clone();
        assert!(!handle.can_mut(), "Handle can't be mut when multiple ref.");
        drop(other_ref);
        assert!(handle.can_mut(), "Handle should be mut when only one ref.");
    }

    // Test pools with slices.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn test_memory_usage() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );
        let handle = memory_management.reserve(100, &mut ErrorGraph::default());
        let usage = memory_management.memory_usage();

        assert_eq!(usage.bytes_in_use, 100);
        // A metadata-sized allocation is carved from one metadata page.
        assert_eq!(usage.bytes_reserved, METADATA_PAGE);

        // Drop and re-alloc.
        drop(handle);
        let _handle = memory_management.reserve(100, &mut ErrorGraph::default());
        let usage_new = memory_management.memory_usage();
        assert_eq!(usage, usage_new);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn find_uninit_binding_returns_not_found() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );

        // Even with a live page at index 0, a never-initialized descriptor must
        // not resolve to it.
        let _live = memory_management
            .reserve(512, &mut ErrorGraph::default())
            .unwrap();

        let binding = ManagedMemoryHandle::new().binding();
        assert!(matches!(
            memory_management.get_cursor(binding),
            Err(IoError::NotFound { .. })
        ));
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn find_stale_descriptor_returns_not_found() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );

        let reserved = memory_management
            .reserve(512, &mut ErrorGraph::default())
            .unwrap();
        let stale = reserved.clone();
        let assigned = ManagedMemoryHandle::new();
        let assigned_binding = assigned.clone().binding();

        memory_management
            .bind(reserved, assigned, 0, &mut ErrorGraph::default())
            .unwrap();

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
        // Exclusive pages: one page per allocation, so freeing two of three
        // leaves the pool with pages to drop and one to renumber.
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        let handle_a = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        let handle_b = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        let handle_c = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();

        let binding_b = handle_b.binding();
        let reserved = memory_management.memory_usage().bytes_reserved;
        drop(handle_a);
        drop(handle_c);

        // Deallocates the two free pages and renumbers the surviving one.
        memory_management.cleanup(true, &mut ErrorGraph::default());

        assert!(memory_management.get_cursor(binding_b.clone()).is_ok());
        assert!(memory_management.get_storage(binding_b).is_ok());
        assert!(
            memory_management.memory_usage().bytes_reserved < reserved,
            "the two freed pages must have gone back"
        );
    }

    /// Persistent windows nest: a module load arms one window around the whole
    /// load while the parameter machinery arms one per parameter inside it —
    /// an inner window closing must not flip the rest of the outer one back to
    /// `Auto`, or most of the load's weights land in the dynamic pools.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn persistent_windows_nest() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );

        memory_management.mode(MemoryAllocationMode::Persistent); // the load's window
        memory_management.mode(MemoryAllocationMode::Persistent); // one parameter's window
        memory_management.mode(MemoryAllocationMode::Auto); // that parameter is done

        // Still inside the load's window: the allocation must be persistent.
        let weight = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        let report = memory_management.memory_report();
        assert_eq!(
            report.persistent.usage.bytes_in_use, 1024,
            "an allocation inside the outer window is persistent"
        );
        assert_eq!(
            report
                .dynamic
                .iter()
                .map(|pool| pool.pages_peak)
                .sum::<u64>(),
            0,
            "nothing leaked into the dynamic pools"
        );

        // The outer window closes; ordinary allocations are dynamic again.
        memory_management.mode(MemoryAllocationMode::Auto);
        let _transient = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        assert_eq!(
            memory_management
                .memory_report()
                .dynamic
                .iter()
                .map(|pool| pool.pages_peak)
                .sum::<u64>(),
            1
        );

        drop(weight);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn alloc_two_chunks_on_one_page() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());
        let _new_handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        // One page, not two: both slices were carved from the same one.
        assert_eq!(usage.bytes_reserved, METADATA_PAGE);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn alloc_reuses_storage() {
        // If no storage is re-used, this will allocate two pages.
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());
        drop(_handle);
        let _new_handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 1);
        assert_eq!(usage.bytes_in_use, alloc_size);
        assert_eq!(usage.bytes_reserved, METADATA_PAGE);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn alloc_allocs_new_storage() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );

        // Three quarters of the adaptive pool's smallest page each: the second
        // one does not fit beside the first, so the pool has to grow a page.
        let page_size = 2 * MIB;
        let alloc_size = page_size / 4 * 3;
        let _handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());
        let _new_handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());

        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert_eq!(usage.bytes_reserved, page_size * 2);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn alloc_respects_alignment_size() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties::new(DUMMY_MEM_PROPS.max_page_size, 50),
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );
        let alloc_size = 40;
        let _handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());
        let _new_handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());
        let usage = memory_management.memory_usage();
        // Each slice should be aligned to 50 bytes, so 10 padding bytes.
        assert_eq!(usage.bytes_padding, 10 * 2);
    }

    /// Each size lands on the pool that serves it: metadata churn stays on the
    /// metadata pages, and the workload gets pages sized to what it asks for.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn allocs_on_correct_page() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_sizes = [4096, 4 * MIB];
        let handles = alloc_sizes.map(|size| {
            memory_management
                .reserve(size, &mut ErrorGraph::default())
                .unwrap()
        });

        assert_ne!(
            handles[0].descriptor().location().pool,
            handles[1].descriptor().location().pool,
            "metadata and the workload must not share a pool"
        );

        let usage = memory_management.memory_usage();
        // Total memory should be size of all pages, and no more: one metadata
        // page, and one adaptive page grown to the 4 MiB allocation.
        assert_eq!(usage.bytes_in_use, alloc_sizes.iter().sum::<u64>());
        assert_eq!(usage.bytes_reserved, METADATA_PAGE + 5 * MIB);
    }

    // The motivating use case: allocations from different "sequence-length
    // ranges" reuse the same arena instead of each landing in its own
    // size-bucketed pool that keeps a separate reservation.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn single_arena_reuses_across_sizes() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );

        // A "small seq" allocation, then freed.
        let small = memory_management
            .reserve(128 * 1024, &mut ErrorGraph::default())
            .unwrap();
        drop(small);
        // A "large seq" allocation must reuse the same arena page.
        let large = memory_management
            .reserve(512 * 1024, &mut ErrorGraph::default())
            .unwrap();

        let usage = memory_management.memory_usage();
        assert_eq!(
            usage.bytes_reserved,
            2 * MIB,
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
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1), &mut ErrorGraph::default()))
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1), &mut ErrorGraph::default()))
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
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::Adaptive,
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate a mix of small and large chunks
        let sizes = [50, 1000, 100, 5000, 200, 10000, 300];
        let handles: Vec<_> = sizes
            .iter()
            .map(|&size| {
                memory_management
                    .reserve(size, &mut ErrorGraph::default())
                    .unwrap()
            })
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate every other allocation
        for i in (0..handles.len()).step_by(2) {
            drop(handles[i].clone());
        }
        // Reallocate similar sizes
        for &size in &sizes[0..sizes.len() / 2] {
            memory_management
                .reserve(size, &mut ErrorGraph::default())
                .unwrap();
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
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );
        let handle = memory_management
            .reserve(10, &mut ErrorGraph::default())
            .unwrap();
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
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());
        let _new_handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());

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
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 512;
        let _handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());
        drop(_handle);
        let _new_handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());

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
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        let alloc_size = 768;
        let _handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());
        let _new_handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());
        let usage = memory_management.memory_usage();
        assert_eq!(usage.number_allocs, 2);
        assert_eq!(usage.bytes_in_use, alloc_size * 2);
        assert!(usage.bytes_reserved >= alloc_size * 2);
    }

    #[test_log::test]
    fn noslice_alloc_respects_alignment_size() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties::new(DUMMY_MEM_PROPS.max_page_size, 50),
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );
        let alloc_size = 40;
        let _handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());
        let _new_handle = memory_management.reserve(alloc_size, &mut ErrorGraph::default());
        let usage = memory_management.memory_usage();
        // Each slice should be aligned to 50 bytes, so 10 padding bytes.
        assert_eq!(usage.bytes_padding, 10 * 2);
    }

    #[test_log::test]
    fn noslice_allocs_on_correct_page() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &MemoryDeviceProperties::new(DUMMY_MEM_PROPS.max_page_size, 10),
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate one thing on each page.
        let alloc_sizes = [50, 150, 250, 350];
        let _handles =
            alloc_sizes.map(|s| memory_management.reserve(s, &mut ErrorGraph::default()));
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
        let first = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        drop(first);
        drop(memory_management.capture_end());

        // A second capture reuses that now-free slice: the reuse must be pinned
        // even though the slice predates the capture.
        memory_management.capture_begin();
        let second = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        drop(second);
        let pins = memory_management.capture_end();
        assert_eq!(pins.len(), 1, "the reused slice must be retained");

        // While pinned, the pool must not hand the slice to a later allocation.
        let before = memory_management.memory_usage();
        let _other = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
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
        let live = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        drop(memory_management.capture_end()); // release the pin; `live` still holds the slice.

        // The window opens with `live`'s slice in use, then frees it mid-window
        // and reuses that exact slice for a window allocation the graph records
        // against. The old snapshot-of-in-use heuristic excluded it (it was in
        // use at begin); reservation-tracking pins it because the window touched
        // it — the whole point of the redesign.
        memory_management.capture_begin();
        drop(live);
        let reused = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
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
        let earlier = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        drop(earlier);
        drop(memory_management.capture_end());

        // A new capture that only ever touches a different size must not retain
        // that leftover idle slice — reservation-tracking pins exactly what the
        // window used, so no free-slice cleanup at `capture_begin` is needed.
        memory_management.capture_begin();
        let window = memory_management
            .reserve(2048, &mut ErrorGraph::default())
            .unwrap();
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
        let handle = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        drop(handle);
        // An explicit cleanup mid-capture compacts the persistent pool; the
        // capture must keep its pins through the rebuild.
        memory_management.cleanup(true, &mut ErrorGraph::default());
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
        let first = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        // A second begin (defensive: callers arm a capture exactly once) must
        // not discard the pins or the saved mode of the capture already in flight.
        memory_management.capture_begin();
        let second = memory_management
            .reserve(2048, &mut ErrorGraph::default())
            .unwrap();
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
        let preexisting = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        drop(memory_management.capture_end());

        memory_management.capture_begin();
        let window = memory_management
            .reserve(2048, &mut ErrorGraph::default())
            .unwrap();
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

    /// Warmup must leave the pool holding its full *distinct working set*, not
    /// its transient peak.
    ///
    /// This is the property the whole priming phase exists for. If warmup is
    /// allowed to recycle its own slices, the pool only ever grows to the peak
    /// number of slices live *at any one instant* during the pass — and that
    /// peak depends on how far the host runs ahead of the device, so it can
    /// land below what the recorded run asks for. The window then has to
    /// allocate, which a capture records as a memory node, and CUDA refuses to
    /// relaunch a graph holding one: the first launch succeeds and every replay
    /// after it fails.
    ///
    /// Here warmup reserves the same size three times *sequentially*, so its
    /// instantaneous peak is one slice while its working set is three. The
    /// recorded run then holds three at once. Without retention the pool ends
    /// warmup with one slice and the window allocates two more.
    #[test_log::test]
    fn capture_priming_leaves_the_working_set_not_the_peak() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        // Warmup: three sequential reserve/drop cycles. Each drop would hand
        // the slice straight back to the next reserve if priming did not retain
        // it, leaving a one-slice pool.
        memory_management.capture_begin();
        for _ in 0..3 {
            let scratch = memory_management
                .reserve(1024, &mut ErrorGraph::default())
                .unwrap();
            drop(scratch);
        }
        // Warmup is over: release the retained slices. They stay in the pool,
        // now free, which is the entire point.
        memory_management.capture_priming_end();
        let after_warmup = memory_management.memory_usage();

        // The recorded run holds three slices of that size simultaneously —
        // more than warmup's instantaneous peak of one. Every one of them must
        // come from the pool.
        let recorded: Vec<_> = (0..3)
            .map(|_| {
                memory_management
                    .reserve(1024, &mut ErrorGraph::default())
                    .unwrap()
            })
            .collect();
        let after_window = memory_management.memory_usage();

        assert_eq!(
            after_window.bytes_reserved, after_warmup.bytes_reserved,
            "the capture window grew the pool: warmup left only its transient \
             peak, so the recorded run had to allocate — which a capture records \
             as a memory node and makes the graph un-relaunchable"
        );

        drop(recorded);
        drop(memory_management.capture_end());
    }

    /// The mechanism behind [`capture_priming_leaves_the_working_set_not_the_peak`]:
    /// a handle dropped *during* priming must not return its slice to the free
    /// list, and `capture_priming_end` must give every one of them back.
    #[test_log::test]
    fn capture_priming_holds_dropped_slices_until_priming_ends() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        memory_management.capture_begin();
        let first = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        drop(first);

        // Still priming: the dropped slice is retained, so this reserve cannot
        // recycle it and the pool has to grow.
        let before_second = memory_management.memory_usage();
        let second = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        let after_second = memory_management.memory_usage();
        assert!(
            after_second.bytes_reserved > before_second.bytes_reserved,
            "priming must retain a dropped slice instead of recycling it"
        );
        drop(second);

        // Priming over: both slices are free again and must now be reused.
        memory_management.capture_priming_end();
        let before_reuse = memory_management.memory_usage();
        let reused = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        let after_reuse = memory_management.memory_usage();
        assert_eq!(
            after_reuse.bytes_reserved, before_reuse.bytes_reserved,
            "capture_priming_end must release the retained slices for reuse"
        );

        drop(reused);
        drop(memory_management.capture_end());
    }

    /// A backend that never calls `capture_priming_end` (HIP did not, before the
    /// call was added to both) must not leak warmup's slices past the capture.
    #[test_log::test]
    fn capture_end_releases_primed_slices_when_priming_never_ended() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        memory_management.capture_begin();
        let scratch = memory_management
            .reserve(1024, &mut ErrorGraph::default())
            .unwrap();
        drop(scratch);
        // The caller let its handle go, but priming is still holding the slice.
        assert!(
            memory_management.memory_usage().bytes_in_use > 0,
            "priming should still be retaining the dropped slice"
        );

        // No `capture_priming_end` — `capture_end` drops the `CaptureState`,
        // and with it every handle priming retained.
        drop(memory_management.capture_end());
        assert_eq!(
            memory_management.memory_usage().bytes_in_use,
            0,
            "primed slices outlived the capture"
        );
    }

    #[test_log::test]
    fn noslice_allocate_deallocate_reallocate() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1), &mut ErrorGraph::default()))
            .collect();
        let usage_before = memory_management.memory_usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| memory_management.reserve(1000 * (i + 1), &mut ErrorGraph::default()))
            .collect();
        let usage_after = memory_management.memory_usage();
        assert_eq!(usage_before.number_allocs, usage_after.number_allocs);
        assert_eq!(usage_before.bytes_in_use, usage_after.bytes_in_use);
        assert_eq!(usage_before.bytes_reserved, usage_after.bytes_reserved);
    }
}
