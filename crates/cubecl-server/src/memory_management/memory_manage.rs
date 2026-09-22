use super::{
    DEDICATED_POOL_POS, ManagedMemoryBinding, ManagedMemoryDescriptor, ManagedMemoryHandle,
    MemoryAllocationMode, MemoryConfiguration, MemoryLocation, PERSISTENT_POOL_POS, PageGuard,
    StreamMemoryReport,
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
    storage::{ComputeStorage, ManagedResource, StorageHandle},
};

use crate::memory_management::relocation::{CopyQueue, RelocationNeed, RelocationReason};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::ops::Range;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::stream::StreamId;
use cubecl_environment::sync::Arc;
use cubecl_ir::MemoryDeviceProperties;

/// Which pool a slice's location routes to: the two fixed sentinels, or a
/// dynamic pool by index.
#[derive(Clone, Copy)]
enum PoolPosition {
    Persistent,
    Dedicated,
    Dynamic(u8),
}

impl PoolPosition {
    fn new(pool: u8) -> Self {
        match pool {
            PERSISTENT_POOL_POS => PoolPosition::Persistent,
            DEDICATED_POOL_POS => PoolPosition::Dedicated,
            index => PoolPosition::Dynamic(index),
        }
    }

    /// The error for a location naming a dynamic pool the layout does not have.
    fn missing(index: u8) -> IoError {
        IoError::NotFound {
            backtrace: BackTrace::capture(),
            reason: format!("Memory pool {index} doesn't exist").into(),
        }
    }
}

/// What a reservation may do to the pages the memory holds to find room.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageUpdate {
    /// Anything: the periodic cleanup runs first and may release or renumber
    /// pages, and a page is allocated when no room is held.
    Allow,
    /// Only add: no page is released or renumbered, so every page keeps its
    /// number and its address, but a page is allocated when no room is held.
    AddOnly,
    /// Nothing: only room the pages already hold.
    Forbidden,
}

/// Why a cleanup runs, which decides how much it gives back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cleanup {
    /// On the way to a reservation: each pool gives back what its own policy
    /// says is unused, such as a page past its deallocation period or an
    /// outdated page that emptied.
    Periodic,
    /// Asked for: everything nothing holds goes back.
    Explicit,
}

/// Reserves and keeps track of chunks of memory in the storage, and slices upon these chunks.
pub struct MemoryManagement<Storage> {
    name: String,
    persistent: PersistentPool,
    /// Allocations made under [`MemoryAllocationMode::Dedicated`]: each its own
    /// device allocation, returned to the driver on the tick after it is freed.
    dedicated: DirectPool,
    pools: DynamicMemory,
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

        self.mode = mode;
    }

    /// Cleanup allocations in pools that are deemed unnecessary.
    pub fn cleanup(&mut self, cleanup: Cleanup, failures: &mut ErrorGraph) {
        self.logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled) && cleanup == Cleanup::Explicit,
            || "Manual memory cleanup ...".to_string(),
        );

        self.persistent.cleanup(
            &mut self.storage,
            self.alloc_reserve_count,
            cleanup,
            failures,
        );

        // Dedicated buffers never wait for an explicit cleanup: freed is done.
        if self.dedicated.reclaim(&mut self.storage, failures) {
            self.storage.flush();
        }

        self.pools.cleanup(
            &mut self.storage,
            self.alloc_reserve_count,
            cleanup,
            failures,
        );

        // The pools only queue their page deallocations in the storage; an
        // explicit cleanup means "release the memory now", so push them to the
        // driver instead of leaving them pending.
        if cleanup == Cleanup::Explicit {
            self.storage.flush();
        }
    }

    /// Returns the storage from the specified binding
    pub fn get_cursor(&self, binding: ManagedMemoryBinding) -> Result<u64, IoError> {
        let slice = self.find(&binding)?;
        Ok(slice.cursor)
    }

    /// The allocation behind `binding`.
    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let slice = self.pool(allocated(binding.descriptor())?)?.find(binding)?;
        owned_by(slice, binding)?;
        Ok(slice)
    }

    /// [`find`](Self::find), mutably — the path [`taint`](Self::taint) and
    /// [`written`](Self::written) take to reach the slice.
    fn find_mut(&mut self, binding: &ManagedMemoryBinding) -> Result<&mut Slice, IoError> {
        let slice = self
            .pool_mut(allocated(binding.descriptor())?)?
            .find_mut(binding)?;
        owned_by(slice, binding)?;
        Ok(slice)
    }

    /// The pool `location` names.
    fn pool(&self, location: MemoryLocation) -> Result<&dyn MemoryPool, IoError> {
        match PoolPosition::new(location.pool) {
            PoolPosition::Persistent => Ok(&self.persistent),
            PoolPosition::Dedicated => Ok(&self.dedicated),
            PoolPosition::Dynamic(index) => self
                .pools
                .pool(index)
                .ok_or_else(|| PoolPosition::missing(index)),
        }
    }

    /// The pool `location` names, mutably.
    fn pool_mut(&mut self, location: MemoryLocation) -> Result<&mut dyn MemoryPool, IoError> {
        match PoolPosition::new(location.pool) {
            PoolPosition::Persistent => Ok(&mut self.persistent),
            PoolPosition::Dedicated => Ok(&mut self.dedicated),
            PoolPosition::Dynamic(index) => self
                .pools
                .pool_mut(index)
                .ok_or_else(|| PoolPosition::missing(index)),
        }
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

    /// The bytes this memory holds from the device right now, whatever pool
    /// holds them.
    pub fn bytes_allocated(&self) -> u64 {
        self.storage.bytes_allocated()
    }

    /// Whether emptying the outdated pools is wanted, before the bytes the
    /// device holds are known: only while something is outdated, when the
    /// arena is full or the device is short of room, and not again while the
    /// pools look as they did when the last relocation moved nothing.
    pub fn relocation_need(&self) -> RelocationNeed {
        self.pools.relocation_need()
    }

    /// Empty what the outdated pools hold into the current pages, so the
    /// pages they held go back to the driver. `reason` decides whether a
    /// target may take a new page.
    pub fn relocate(
        &mut self,
        copier: &mut dyn CopyQueue<Storage>,
        reason: RelocationReason,
        failures: &mut ErrorGraph,
    ) {
        self.pools
            .relocate(&mut self.storage, copier, reason, failures);
    }

    /// Keep the page `location` names as it is for as long as the guard lives
    /// (see [`PageGuard`]). `None` when no pool holds such a page.
    pub fn guard(&mut self, location: MemoryLocation) -> Option<PageGuard> {
        self.pool_mut(location).ok()?.guard(location)
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
            PoolPosition::Dynamic(_) => self.pools.materialize(&mut self.storage, binding),
        }
    }

    /// The resource behind `binding`, holding the allocation and a guard on
    /// its page for as long as it lives: what a caller that keeps the raw
    /// address past this call is handed.
    pub fn managed_resource(
        &mut self,
        binding: ManagedMemoryBinding,
        offset_start: Option<u64>,
        offset_end: Option<u64>,
    ) -> Result<ManagedResource<Storage::Resource>, IoError> {
        let guard = self.guard(binding.descriptor().location());
        let resource = self.get_resource(binding.clone(), offset_start, offset_end)?;
        Ok(ManagedResource::new(binding, resource, guard))
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

    /// Reserve `size` bytes, changing the pages held only as far as `update`
    /// allows.
    ///
    /// # Errors
    ///
    /// [`IoError::PageUpdateForbidden`] when `update` forbids a new page and
    /// the room held cannot serve it, and whatever the pools or the device
    /// refused otherwise.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip_all))]
    pub fn reserve(
        &mut self,
        size: u64,
        update: PageUpdate,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        match update {
            PageUpdate::Allow => {
                // Drive the pools' periodic deallocation. Each pool gates
                // itself on its own `dealloc_period` (pools without one no-op),
                // so this is a few comparisons per reservation — without it,
                // pages freed long ago are never returned to the driver until
                // an explicit cleanup, which on long-running processes lets
                // every stream's pools grow monotonically.
                self.cleanup(Cleanup::Periodic, failures);
                self.reserve_adding(size, failures)
            }
            PageUpdate::AddOnly => self.reserve_adding(size, failures),
            PageUpdate::Forbidden => {
                self.reserve_held(size, failures)
                    .ok_or_else(|| IoError::PageUpdateForbidden {
                        size,
                        backtrace: BackTrace::capture(),
                    })
            }
        }
    }

    /// Reserve `size` bytes, allocating a page when the room held has none,
    /// without releasing or renumbering any page.
    fn reserve_adding(
        &mut self,
        size: u64,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        // If this happens every nanosecond, counts overflows after 585 years, so not worth thinking too
        // hard about overflow here.
        self.alloc_reserve_count += 1;

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

    /// Reserve `size` bytes in the room the pools already hold, the way
    /// [`reserve_adding`](Self::reserve_adding) would route them: no cleanup,
    /// no new page, nothing moved. `None` when that room cannot serve it.
    fn reserve_held(
        &mut self,
        size: u64,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        match self.mode {
            MemoryAllocationMode::Dedicated => self.dedicated.try_reserve(size, failures),
            MemoryAllocationMode::Persistent => self.persistent.try_reserve(size, failures),
            MemoryAllocationMode::Auto => {
                if matches!(self.config, PersistentMemory::SizeMatch)
                    && let Some(handle) = self.persistent.try_reserve(size, failures)
                {
                    return Some(handle);
                }
                self.pools.try_reserve(size, failures)
            }
        }
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

    /// Everything this memory holds, pool by pool, for the `stream` it
    /// belongs to: each pool's shape, usage and high-water marks, the dynamic
    /// ones in allocation-routing order. [`StreamMemoryReport::usage`] sums
    /// them.
    pub fn memory_report(&self, stream: StreamId) -> StreamMemoryReport {
        StreamMemoryReport {
            stream,
            dynamic: self.pools.report(),
            persistent: self.persistent.report(),
            dedicated: self.dedicated.report(),
        }
    }

    /// Print out a report of the current memory usage.
    pub fn print_memory_usage(&self) {
        #[cfg(feature = "std")]
        log::info!("{}", self.memory_report(StreamId::current()).usage());
    }

    /// Binds the given [handle](HandleId) to a [`MemorySlot`].
    pub fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) -> Result<(), IoError> {
        let location = allocated(reserved.descriptor())?;
        self.pool_mut(location)?
            .bind(reserved, assigned, cursor, failures)
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

/// Where the allocation `descriptor` names sits, once a reservation gave it
/// a place.
fn allocated(descriptor: &ManagedMemoryDescriptor) -> Result<MemoryLocation, IoError> {
    if !descriptor.is_allocated() {
        return Err(IoError::NotFound {
            backtrace: BackTrace::capture(),
            reason: "Memory location was never initialized".into(),
        });
    }
    Ok(descriptor.location())
}

/// Whether `slice` still holds the allocation behind `binding`. A stale
/// location (a page deallocated since, whose index a later cleanup reassigned)
/// must surface as `NotFound`, never as another allocation's slice.
fn owned_by(slice: &Slice, binding: &ManagedMemoryBinding) -> Result<(), IoError> {
    if slice.handle.descriptor() != binding.descriptor() {
        return Err(IoError::NotFound {
            backtrace: BackTrace::capture(),
            reason: "Memory location points to a different allocation".into(),
        });
    }
    Ok(())
}

impl<Storage: ComputeStorage> core::fmt::Display for MemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("\n# MemoryManagement\n\n")?;
        f.write_fmt(format_args!(" - name: {:?}\n", self.name))?;
        f.write_fmt(format_args!("\n## Persistent\n\n{}", self.persistent))?;
        f.write_str("\n## Dynamic\n\n")?;

        f.write_fmt(format_args!("{}", self.pools))?;
        // Only the usage is read, so any stream will do for the report.
        let memory_usage = self.memory_report(StreamId::current()).usage();
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
        let _near = memory
            .reserve(7 * MIB, PageUpdate::Allow, &mut ErrorGraph::default())
            .unwrap();

        let served = memory
            .memory_report(StreamId::current())
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
            .reserve(10, PageUpdate::Allow, &mut ErrorGraph::default())
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
        let handle = memory_management.reserve(100, PageUpdate::Allow, &mut ErrorGraph::default());
        let usage = memory_management.memory_report(StreamId::current()).usage();

        assert_eq!(usage.bytes_in_use, 100);
        // A metadata-sized allocation is carved from one metadata page.
        assert_eq!(usage.bytes_reserved, METADATA_PAGE);

        // Drop and re-alloc.
        drop(handle);
        let _handle = memory_management.reserve(100, PageUpdate::Allow, &mut ErrorGraph::default());
        let usage_new = memory_management.memory_report(StreamId::current()).usage();
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
            .reserve(512, PageUpdate::Allow, &mut ErrorGraph::default())
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
            .reserve(512, PageUpdate::Allow, &mut ErrorGraph::default())
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
            .reserve(1024, PageUpdate::Allow, &mut ErrorGraph::default())
            .unwrap();
        let handle_b = memory_management
            .reserve(1024, PageUpdate::Allow, &mut ErrorGraph::default())
            .unwrap();
        let handle_c = memory_management
            .reserve(1024, PageUpdate::Allow, &mut ErrorGraph::default())
            .unwrap();

        let binding_b = handle_b.binding();
        let reserved = memory_management
            .memory_report(StreamId::current())
            .usage()
            .bytes_reserved;
        drop(handle_a);
        drop(handle_c);

        // Deallocates the two free pages and renumbers the surviving one.
        memory_management.cleanup(Cleanup::Explicit, &mut ErrorGraph::default());

        assert!(memory_management.get_cursor(binding_b.clone()).is_ok());
        assert!(memory_management.get_storage(binding_b).is_ok());
        assert!(
            memory_management
                .memory_report(StreamId::current())
                .usage()
                .bytes_reserved
                < reserved,
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
            .reserve(1024, PageUpdate::Allow, &mut ErrorGraph::default())
            .unwrap();
        let report = memory_management.memory_report(StreamId::current());
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
            .reserve(1024, PageUpdate::Allow, &mut ErrorGraph::default())
            .unwrap();
        assert_eq!(
            memory_management
                .memory_report(StreamId::current())
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
        let _handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());
        let _new_handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());

        let usage = memory_management.memory_report(StreamId::current()).usage();
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
        let _handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());
        drop(_handle);
        let _new_handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());

        let usage = memory_management.memory_report(StreamId::current()).usage();
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
        let _handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());
        let _new_handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());

        let usage = memory_management.memory_report(StreamId::current()).usage();
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
        let _handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());
        let _new_handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());
        let usage = memory_management.memory_report(StreamId::current()).usage();
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
                .reserve(size, PageUpdate::Allow, &mut ErrorGraph::default())
                .unwrap()
        });

        assert_ne!(
            handles[0].descriptor().location().pool,
            handles[1].descriptor().location().pool,
            "metadata and the workload must not share a pool"
        );

        let usage = memory_management.memory_report(StreamId::current()).usage();
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
            .reserve(128 * 1024, PageUpdate::Allow, &mut ErrorGraph::default())
            .unwrap();
        drop(small);
        // A "large seq" allocation must reuse the same arena page.
        let large = memory_management
            .reserve(512 * 1024, PageUpdate::Allow, &mut ErrorGraph::default())
            .unwrap();

        let usage = memory_management.memory_report(StreamId::current()).usage();
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
            .map(|i| {
                memory_management.reserve(
                    1000 * (i + 1),
                    PageUpdate::Allow,
                    &mut ErrorGraph::default(),
                )
            })
            .collect();
        let usage_before = memory_management.memory_report(StreamId::current()).usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| {
                memory_management.reserve(
                    1000 * (i + 1),
                    PageUpdate::Allow,
                    &mut ErrorGraph::default(),
                )
            })
            .collect();
        let usage_after = memory_management.memory_report(StreamId::current()).usage();
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
                    .reserve(size, PageUpdate::Allow, &mut ErrorGraph::default())
                    .unwrap()
            })
            .collect();
        let usage_before = memory_management.memory_report(StreamId::current()).usage();
        // Deallocate every other allocation
        for i in (0..handles.len()).step_by(2) {
            drop(handles[i].clone());
        }
        // Reallocate similar sizes
        for &size in &sizes[0..sizes.len() / 2] {
            memory_management
                .reserve(size, PageUpdate::Allow, &mut ErrorGraph::default())
                .unwrap();
        }
        let usage_after = memory_management.memory_report(StreamId::current()).usage();
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
            .reserve(10, PageUpdate::Allow, &mut ErrorGraph::default())
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
        let _handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());
        let _new_handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());

        let usage = memory_management.memory_report(StreamId::current()).usage();
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
        let _handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());
        drop(_handle);
        let _new_handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());

        let usage = memory_management.memory_report(StreamId::current()).usage();
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
        let _handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());
        let _new_handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());
        let usage = memory_management.memory_report(StreamId::current()).usage();
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
        let _handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());
        let _new_handle =
            memory_management.reserve(alloc_size, PageUpdate::Allow, &mut ErrorGraph::default());
        let usage = memory_management.memory_report(StreamId::current()).usage();
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
        let _handles = alloc_sizes
            .map(|s| memory_management.reserve(s, PageUpdate::Allow, &mut ErrorGraph::default()));
        let usage = memory_management.memory_report(StreamId::current()).usage();
        // Total memory should be size of all pages, and no more.
        assert_eq!(usage.bytes_in_use, alloc_sizes.iter().sum::<u64>());
    }

    #[test_log::test]
    fn a_guarded_allocation_is_never_handed_out_again() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        let recorded = memory_management
            .reserve(1024, PageUpdate::Allow, &mut ErrorGraph::default())
            .unwrap();
        let guard = memory_management
            .guard(recorded.descriptor().location())
            .expect("the pool holds the page");
        drop(recorded);

        // Freed, but guarded: the next allocation of that size needs a page
        // of its own, and a cleanup leaves the guarded one where it is.
        let before = memory_management
            .memory_report(StreamId::current())
            .usage()
            .bytes_reserved;
        let other = memory_management
            .reserve(1024, PageUpdate::Allow, &mut ErrorGraph::default())
            .unwrap();
        let after = memory_management
            .memory_report(StreamId::current())
            .usage()
            .bytes_reserved;
        assert!(after > before, "the guarded page was handed out again");
        memory_management.cleanup(Cleanup::Explicit, &mut ErrorGraph::default());
        assert_eq!(
            memory_management
                .memory_report(StreamId::current())
                .usage()
                .bytes_reserved,
            after,
            "the guarded page was released"
        );

        drop(guard);
        drop(other);
        memory_management.cleanup(Cleanup::Explicit, &mut ErrorGraph::default());
        assert_eq!(
            memory_management
                .memory_report(StreamId::current())
                .usage()
                .bytes_reserved,
            0
        );
    }

    #[test_log::test]
    fn a_guard_leaves_in_place_updates_alone() {
        let mut memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &DUMMY_MEM_PROPS,
            MemoryConfiguration::ExclusivePages,
            Arc::new(ServerLogger::default()),
            options(),
        );

        let cache = memory_management
            .reserve(1024, PageUpdate::Allow, &mut ErrorGraph::default())
            .unwrap();
        let _guard = memory_management.guard(cache.descriptor().location());

        assert!(
            cache.can_mut(),
            "a graph replaying against a buffer must not stop it being updated in place"
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
            .map(|i| {
                memory_management.reserve(
                    1000 * (i + 1),
                    PageUpdate::Allow,
                    &mut ErrorGraph::default(),
                )
            })
            .collect();
        let usage_before = memory_management.memory_report(StreamId::current()).usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| {
                memory_management.reserve(
                    1000 * (i + 1),
                    PageUpdate::Allow,
                    &mut ErrorGraph::default(),
                )
            })
            .collect();
        let usage_after = memory_management.memory_report(StreamId::current()).usage();
        assert_eq!(usage_before.number_allocs, usage_after.number_allocs);
        assert_eq!(usage_before.bytes_in_use, usage_after.bytes_in_use);
        assert_eq!(usage_before.bytes_reserved, usage_after.bytes_reserved);
    }
}
