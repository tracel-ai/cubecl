use super::{
    AdaptiveReserver, BucketReserver, DEDICATED_POOL_POS, ManagedMemoryBinding,
    ManagedMemoryDescriptor, ManagedMemoryHandle, MemoryAllocationMode, MemoryAllocationOption,
    MemoryConfiguration, MemoryLocation, MemoryManagementOptions, MemoryPoolReport,
    MemoryPoolsReport, PERSISTENT_POOL_POS, PageGuard,
    memory_pool::{
        DirectSliceReserver, PageMapping, PersistentSliceReserver, Slice, SliceReserver,
    },
};
use crate::{
    config::{
        CubeClRuntimeConfig, RuntimeConfig,
        memory::{MemoryLogLevel, PersistentMemory},
    },
    logging::ServerLogger,
    memory_management::{BytesFormat, ErrorGraph, FailureId},
    server::{IoError, ServerError},
    storage::{ManagedResource, StorageAllocator, StorageHandle},
};

use crate::memory_management::relocation::{RelocationCopier, RelocationNeed, RelocationReason};
use alloc::boxed::Box;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::ops::Range;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::sync::Arc;
use cubecl_ir::MemoryDeviceProperties;

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

/// What an allocation is for, which decides whether it may carve the free room
/// on pages graphs pinned (see [`PageGuard`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationKind {
    /// Work that runs for real, outside a graph capture: it outlives the next
    /// replay of every graph on the stream, so it never lands where one writes.
    Eager,
    /// A temporary of a graph capture on the stream that owns this memory. Its
    /// graph replays in stream order with every graph that pinned a page, so
    /// it shares the room their temporaries used.
    GraphIntermediate,
    /// What outlives a graph capture: the work's outputs, the metadata its
    /// graph keeps. Held for as long as its graph lives, apart from every page
    /// the stream's graphs share, so no replay writes over it.
    GraphOutput,
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

/// The device memory of one stream: reservations carved from pools of pages,
/// the allocations bound to them, and the failures they carry.
///
/// # Assumptions
///
/// - `failures`, in every function that takes it, is the [`ErrorGraph`] that
///   holds the failures this memory's allocations carry.
/// - Every handle and binding handed to a function was placed by this
///   memory's reservations and names its allocation's current location, or
///   was never placed at all.
///
/// # Guarantees
///
/// A reservation releases or renumbers a page only under
/// [`PageUpdate::Allow`], and allocates one only under `Allow` or
/// [`PageUpdate::AddOnly`].
pub struct MemoryManagement<Storage> {
    reserver: MemoryReserver<Storage>,
}

impl<Storage: StorageAllocator> MemoryManagement<Storage> {
    /// The memory of a device with `properties`, laid out as `config` says,
    /// over `storage`.
    pub fn from_configuration(
        storage: Storage,
        properties: &MemoryDeviceProperties,
        config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
        options: MemoryManagementOptions,
    ) -> Self {
        Self {
            reserver: MemoryReserver::new(storage, properties, config, logger, options),
        }
    }

    /// Open or close an allocation window.
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
        self.reserver.mode(mode);
    }

    /// Give back what the pools hold and nothing uses, as far as `cleanup`
    /// asks; an explicit one hands every deallocation to the driver before it
    /// returns.
    ///
    /// # Panics
    ///
    /// When `failures` is not the graph this memory's failures are in.
    pub fn cleanup(&mut self, cleanup: Cleanup, failures: &mut ErrorGraph) {
        self.reserver.cleanup(cleanup, failures);
    }

    /// The stream cursor the allocation behind `binding` was bound at.
    ///
    /// # Guarantees
    ///
    /// A binding that never had an allocation is refused with
    /// [`IoError::NotFound`], never resolved to another allocation.
    ///
    /// # Errors
    ///
    /// [`IoError::NotFound`] when `binding` was never placed.
    ///
    /// # Panics
    ///
    /// When `binding` names a location this memory does not hold for it.
    pub fn get_cursor(&self, binding: ManagedMemoryBinding) -> Result<u64, IoError> {
        self.reserver.get_cursor(&binding)
    }

    /// The storage behind `binding`, which always refers to mapped memory: the
    /// funnel every buffer dereference passes through, so it is where a
    /// lazily-carved allocation gets its real device backing.
    ///
    /// # Guarantees
    ///
    /// A binding that never had an allocation is refused with
    /// [`IoError::NotFound`], never resolved to another allocation.
    ///
    /// # Errors
    ///
    /// [`IoError::NotFound`] when `binding` was never placed, and
    /// [`IoError::StorageMappingFailed`] when the storage refuses the backing.
    ///
    /// # Panics
    ///
    /// When `binding` names a location this memory does not hold for it.
    pub fn get_storage(&mut self, binding: ManagedMemoryBinding) -> Result<StorageHandle, IoError> {
        self.reserver.get_storage(&binding)
    }

    /// The bytes this memory holds from the device right now, whatever pool
    /// holds them.
    pub fn bytes_allocated(&self) -> u64 {
        self.reserver.bytes_allocated()
    }

    /// Whether emptying the outdated pools is wanted, before the bytes the
    /// device holds are known: only while something is outdated, when the
    /// arena is full or the device may be short of room, and not again while
    /// the pools look as they did when the last relocation moved nothing.
    /// [`RelocationNeed::resolve`] settles it against those bytes.
    pub fn relocation_need(&self) -> RelocationNeed {
        self.reserver.relocation_need()
    }

    /// Empty what the outdated pools hold into the current pages, so the
    /// pages they held go back to the driver, with the copies on `copier`.
    /// `reason` decides whether a target may take a new page.
    ///
    /// # Guarantees
    ///
    /// With nothing outdated, moves nothing and waits on nothing.
    ///
    /// # Panics
    ///
    /// When `failures` is not the graph this memory's failures are in.
    pub fn relocate(
        &mut self,
        copier: &mut dyn RelocationCopier<Storage>,
        reason: RelocationReason,
        failures: &mut ErrorGraph,
    ) {
        self.reserver.relocate(copier, reason, failures);
    }

    /// Move every allocation of `kept` that sits on a page a graph guards onto room no graph's
    /// replay writes, with its bytes: what a graph capture's planning runs left alive, whose
    /// addresses nothing recorded. The copies are queued on `copier`.
    ///
    /// # Assumptions
    ///
    /// - Every binding of `kept` names a page this memory holds.
    /// - Nothing outside `kept` holds a moved allocation's raw address: what
    ///   [`managed_resource`](Self::managed_resource) handed out for one is
    ///   read by no one once it moves. A graph capture holds this, since it
    ///   keeps apart only what its planning runs left alive and recorded
    ///   nowhere.
    ///
    /// # Guarantees
    ///
    /// An allocation `kept` names more than once moves once.
    ///
    /// # Errors
    ///
    /// When no room apart takes one, or a copy fails: nothing moved then.
    ///
    /// # Panics
    ///
    /// When `failures` is not the graph this memory's failures are in, or a
    /// binding of `kept` names a page this memory does not hold.
    pub fn move_apart(
        &mut self,
        kept: &[ManagedMemoryBinding],
        copier: &mut dyn RelocationCopier<Storage>,
        failures: &mut ErrorGraph,
    ) -> Result<(), ServerError> {
        self.reserver.move_apart(kept, copier, failures)
    }

    /// Keep the page `location` names as it is for as long as the guard lives
    /// (see [`PageGuard`]). `None` when no pool holds such a page.
    pub fn guard(&mut self, location: MemoryLocation) -> Option<PageGuard> {
        self.reserver.guard(location)
    }

    /// The resource behind `binding`, trimmed by its offsets, holding the
    /// allocation's binding for as long as it lives: what a caller that keeps
    /// the raw address past this call is handed. The binding keeps the
    /// allocation in place, and leaves the rest of its page to other
    /// allocations.
    ///
    /// # Guarantees
    ///
    /// A binding that never had an allocation is refused with
    /// [`IoError::NotFound`], never resolved to another allocation.
    ///
    /// # Errors
    ///
    /// As [`get_resource`](Self::get_resource).
    ///
    /// # Panics
    ///
    /// As [`get_resource`](Self::get_resource).
    pub fn managed_resource(
        &mut self,
        binding: ManagedMemoryBinding,
        offset_start: Option<u64>,
        offset_end: Option<u64>,
    ) -> Result<ManagedResource<Storage::Resource>, IoError> {
        self.reserver
            .managed_resource(binding, offset_start, offset_end)
    }

    /// The resource behind `binding`, trimmed by its offsets.
    ///
    /// # Guarantees
    ///
    /// A binding that never had an allocation is refused with
    /// [`IoError::NotFound`], never resolved to another allocation.
    ///
    /// # Errors
    ///
    /// [`IoError::BindingOutOfBounds`] when the offsets reach past the end of
    /// the binding's memory, and whatever [`get_storage`](Self::get_storage)
    /// refuses.
    ///
    /// # Panics
    ///
    /// When `binding` names a location this memory does not hold for it.
    pub fn get_resource(
        &mut self,
        binding: ManagedMemoryBinding,
        offset_start: Option<u64>,
        offset_end: Option<u64>,
    ) -> Result<Storage::Resource, IoError> {
        self.reserver
            .get_resource(&binding, offset_start, offset_end)
    }

    /// Reserve `size` bytes for work outside a graph capture, changing the
    /// pages held only as far as `update` allows.
    ///
    /// # Errors
    ///
    /// [`IoError::PageUpdateForbidden`] when `update` forbids a new page and
    /// the room held cannot serve it, and whatever the pools or the device
    /// refused otherwise.
    ///
    /// # Panics
    ///
    /// When `failures` is not the graph this memory's failures are in.
    pub fn reserve(
        &mut self,
        size: u64,
        update: PageUpdate,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        self.reserver
            .reserve(size, update, AllocationKind::Eager, failures)
    }

    /// [`reserve`](Self::reserve), for an allocation of `kind`.
    ///
    /// # Errors
    ///
    /// As [`reserve`](Self::reserve).
    ///
    /// # Panics
    ///
    /// As [`reserve`](Self::reserve).
    pub fn reserve_as(
        &mut self,
        size: u64,
        update: PageUpdate,
        kind: AllocationKind,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        self.reserver.reserve(size, update, kind, failures)
    }

    /// The storage the memory allocates from, to time deallocations against
    /// asynchronous work or change how it allocates. An allocation made on it
    /// directly is no reservation of this memory: no binding resolves to it.
    pub fn storage(&mut self) -> &mut Storage {
        self.reserver.storage()
    }

    /// Everything this memory holds, pool by pool: each pool's shape, usage
    /// and high-water marks, the dynamic ones in allocation-routing order.
    /// [`MemoryPoolsReport::usage`] sums them.
    pub fn memory_report(&self) -> MemoryPoolsReport {
        self.reserver.memory_report()
    }

    /// Log the current memory usage.
    pub fn print_memory_usage(&self) {
        self.reserver.print_memory_usage();
    }

    /// Bind `assigned`, a handle no reservation placed yet, to the slice
    /// `reserved` holds, at `cursor`: the point of the stream from which the
    /// binding is valid. The failures the slice's last allocation carried
    /// are released.
    ///
    /// # Assumptions
    ///
    /// `reserved` came from a reservation of this memory and was not bound
    /// since: a second bind would take the slice from the allocation the
    /// first bound to it.
    ///
    /// # Panics
    ///
    /// When `failures` is not the graph this memory's failures are in, or
    /// `reserved` was never placed or names a slice this memory does not
    /// hold: binding it would overwrite another allocation's handle.
    pub fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) {
        self.reserver.bind(reserved, assigned, cursor, failures)
    }

    /// The failure claiming any byte of `range` in the allocation behind
    /// `binding`, if one does.
    ///
    /// # Assumptions
    ///
    /// `binding` was placed: a caller asks the error graph about a buffer
    /// never allocated.
    ///
    /// # Panics
    ///
    /// When `binding` was never placed, or names a location this memory does
    /// not hold for it.
    pub fn failure(&self, binding: &ManagedMemoryBinding, range: Range<u64>) -> Option<FailureId> {
        self.reserver.failure(binding, range)
    }

    /// Point `range` of the allocation behind `binding` at `failure`.
    ///
    /// # Assumptions
    ///
    /// `failure` is one `failures` holds.
    ///
    /// # Guarantees
    ///
    /// A binding never placed is left alone: it has no slice to carry a
    /// failure, and the callers pass every binding unfiltered.
    ///
    /// # Panics
    ///
    /// When `failure` is not one `failures` holds, `failures` is not the
    /// graph this memory's failures are in, or a placed `binding` names a
    /// location this memory does not hold for it.
    pub fn taint(
        &mut self,
        binding: &ManagedMemoryBinding,
        range: Range<u64>,
        failure: FailureId,
        failures: &mut ErrorGraph,
    ) {
        self.reserver.taint(binding, range, failure, failures);
    }

    /// `range` of the allocation behind `binding` has a writer again: release
    /// every claim on those bytes — and only those bytes, since a partial
    /// write says nothing about the rest of the buffer.
    ///
    /// # Guarantees
    ///
    /// A binding never placed is left alone.
    ///
    /// # Panics
    ///
    /// When `failures` is not the graph this memory's failures are in, or a
    /// placed `binding` names a location this memory does not hold for it.
    pub fn written(
        &mut self,
        binding: &ManagedMemoryBinding,
        range: Range<u64>,
        failures: &mut ErrorGraph,
    ) {
        self.reserver.written(binding, range, failures);
    }
}

impl<Storage: StorageAllocator> core::fmt::Display for MemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.reserver)
    }
}

impl<Storage> core::fmt::Debug for MemoryManagement<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "MemoryManagement {:?}",
            core::any::type_name::<Storage>()
        )
    }
}

/// Reserves and keeps track of slices of the storage's device allocations,
/// routing each to the pool its mode and size call for.
struct MemoryReserver<Storage> {
    name: String,
    persistent: PersistentSliceReserver,
    /// Allocations made under [`MemoryAllocationMode::Dedicated`]: each its own
    /// device allocation, returned to the driver on the tick after it is freed.
    dedicated: DirectSliceReserver,
    dynamic: MemoryReserverDispatch,
    storage: Storage,
    /// The reservations made so far, which the pools' deallocation periods
    /// count in.
    reservation_count: u64,
    mode: MemoryAllocationMode,
    /// The mode no window overrides: the configured or provided one.
    base_mode: MemoryAllocationMode,
    /// The open allocation windows, innermost last (Arrival), each holding
    /// the mode it put in force.
    windows: Vec<MemoryAllocationMode>,
    config: PersistentMemory,
    logger: Arc<ServerLogger>,
}

/// The dynamic pools a stream allocates from: everything a workload
/// allocates and frees as it runs, as against the persistent memory its
/// weights sit in and the dedicated buffers that own their allocation.
enum MemoryReserverDispatch {
    /// One page per allocation, in size buckets. Nothing is ever outdated.
    Buckets(BucketReserver),
    /// Pages sized to the largest allocation served, with a pool per size:
    /// the pool a growth leaves behind drains and is dropped. Boxed: it holds
    /// its pools inline, and a memory is built once.
    Adaptive(Box<AdaptiveReserver>),
}

/// Which pool a slice's location routes to: the two fixed sentinels, or a
/// dynamic pool by index.
#[derive(Clone, Copy)]
enum PoolPosition {
    Persistent,
    Dedicated,
    Dynamic(u8),
}

impl PoolPosition {
    /// The pool position `pool_index` carries.
    fn from_pool_index(pool_index: u8) -> Self {
        match pool_index {
            PERSISTENT_POOL_POS => PoolPosition::Persistent,
            DEDICATED_POOL_POS => PoolPosition::Dedicated,
            index => PoolPosition::Dynamic(index),
        }
    }
}

impl<Storage: StorageAllocator> MemoryReserver<Storage> {
    fn new(
        storage: Storage,
        properties: &MemoryDeviceProperties,
        config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
        options: MemoryManagementOptions,
    ) -> Self {
        let (name, memory) = options.into_parts();
        let dynamic = MemoryReserverDispatch::new(properties, config, logger.clone(), name.clone());
        let config = CubeClRuntimeConfig::get().memory.persistent_memory.clone();
        let mode = match memory {
            MemoryAllocationOption::Provided(mode) => mode,
            MemoryAllocationOption::FromConfig => configured_mode(&config),
        };

        Self {
            name,
            // Both pools hold one device allocation per buffer, so neither is
            // bounded by a page: each takes the largest single allocation.
            persistent: PersistentSliceReserver::new(
                properties.max_allocation_size(),
                properties.alignment,
                PERSISTENT_POOL_POS,
            ),
            // A watermark of zero: every free slice goes back on the next tick.
            dedicated: DirectSliceReserver::new(
                properties.max_allocation_size(),
                properties.alignment,
                DEDICATED_POOL_POS,
                Some(0),
            ),
            dynamic,
            storage,
            reservation_count: 0,
            mode,
            base_mode: mode,
            windows: Vec::new(),
            config,
            logger,
        }
    }

    fn mode(&mut self, mode: MemoryAllocationMode) {
        match mode {
            MemoryAllocationMode::Auto => {
                self.windows.truncate(self.windows.len().saturating_sub(1))
            }
            MemoryAllocationMode::Persistent => self.windows.push(self.persistent_window()),
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

    /// The mode a `Persistent` window puts in force under the persistent-memory config.
    fn persistent_window(&self) -> MemoryAllocationMode {
        match self.config {
            PersistentMemory::Enabled | PersistentMemory::SizeMatch => {
                MemoryAllocationMode::Persistent
            }
            PersistentMemory::Disabled | PersistentMemory::Enforced => self.base_mode,
        }
    }

    fn cleanup(&mut self, cleanup: Cleanup, failures: &mut ErrorGraph) {
        self.logger.log_memory(
            |level| !matches!(level, MemoryLogLevel::Disabled) && cleanup == Cleanup::Explicit,
            || "Manual memory cleanup ...".to_string(),
        );

        self.persistent
            .cleanup(&mut self.storage, self.reservation_count, cleanup, failures);
        // Dedicated buffers never wait for an explicit cleanup: freed is done.
        let reclaimed = self.dedicated.reclaim(&mut self.storage, failures);
        self.cleanup_dynamic(cleanup, failures);

        // The pools only queue their page deallocations in the storage; a
        // reclaimed dedicated buffer, or an explicit cleanup, means "release
        // the memory now", so push them to the driver instead of leaving them
        // pending.
        if reclaimed || cleanup == Cleanup::Explicit {
            self.storage.flush();
        }
    }

    fn get_cursor(&self, binding: &ManagedMemoryBinding) -> Result<u64, IoError> {
        Ok(self.find(binding)?.cursor)
    }

    /// The allocation behind `binding`.
    fn find(&self, binding: &ManagedMemoryBinding) -> Result<&Slice, IoError> {
        let slice = self
            .pool(allocated(binding.descriptor())?)
            .find(binding)
            .expect("a binding should name a slice its pool holds");
        owned_by(slice, binding);
        Ok(slice)
    }

    /// [`find`](Self::find), mutably — the path [`taint`](Self::taint) and
    /// [`written`](Self::written) take to reach the slice.
    fn find_mut(&mut self, binding: &ManagedMemoryBinding) -> Result<&mut Slice, IoError> {
        let slice = self
            .pool_mut(allocated(binding.descriptor())?)
            .find_mut(binding)
            .expect("a binding should name a slice its pool holds");
        owned_by(slice, binding);
        Ok(slice)
    }

    /// The pool `location` names.
    ///
    /// # Panics
    ///
    /// When `location` names no pool of this memory.
    fn pool(&self, location: MemoryLocation) -> &dyn SliceReserver {
        match PoolPosition::from_pool_index(location.pool) {
            PoolPosition::Persistent => &self.persistent,
            PoolPosition::Dedicated => &self.dedicated,
            PoolPosition::Dynamic(index) => self
                .dynamic_pool_by_index(index)
                .expect("a location should name a pool of the memory that placed it"),
        }
    }

    /// The pool `location` names, mutably.
    ///
    /// # Panics
    ///
    /// When `location` names no pool of this memory.
    fn pool_mut(&mut self, location: MemoryLocation) -> &mut dyn SliceReserver {
        self.try_pool_mut(location)
            .expect("a location should name a pool of the memory that placed it")
    }

    /// [`pool_mut`](Self::pool_mut), `None` where it panics.
    fn try_pool_mut(&mut self, location: MemoryLocation) -> Option<&mut dyn SliceReserver> {
        match PoolPosition::from_pool_index(location.pool) {
            PoolPosition::Persistent => Some(&mut self.persistent),
            PoolPosition::Dedicated => Some(&mut self.dedicated),
            PoolPosition::Dynamic(index) => self.dynamic_pool_by_index_mut(index),
        }
    }

    fn get_storage(&mut self, binding: &ManagedMemoryBinding) -> Result<StorageHandle, IoError> {
        self.materialize(binding)?;
        Ok(self.find(binding)?.storage.clone())
    }

    fn bytes_allocated(&self) -> u64 {
        self.storage.bytes_allocated()
    }

    fn relocation_need(&self) -> RelocationNeed {
        match &self.dynamic {
            MemoryReserverDispatch::Buckets(_) => RelocationNeed::Nothing,
            MemoryReserverDispatch::Adaptive(memory) => memory.relocation_need(),
        }
    }

    fn relocate(
        &mut self,
        copier: &mut dyn RelocationCopier<Storage>,
        reason: RelocationReason,
        failures: &mut ErrorGraph,
    ) {
        match &mut self.dynamic {
            MemoryReserverDispatch::Buckets(_) => {}
            MemoryReserverDispatch::Adaptive(memory) => {
                memory.relocate(&mut self.storage, copier, reason, failures)
            }
        }
    }

    fn move_apart(
        &mut self,
        kept: &[ManagedMemoryBinding],
        copier: &mut dyn RelocationCopier<Storage>,
        failures: &mut ErrorGraph,
    ) -> Result<(), ServerError> {
        match &mut self.dynamic {
            MemoryReserverDispatch::Buckets(_) => Ok(()),
            MemoryReserverDispatch::Adaptive(memory) => {
                memory.move_apart(&mut self.storage, kept, copier, failures)
            }
        }
    }

    fn guard(&mut self, location: MemoryLocation) -> Option<PageGuard> {
        self.try_pool_mut(location)?.guard(location)
    }

    /// Install real backing behind `binding` when its allocation was carved
    /// lazily under a process mode that discards launches. Lookup errors are left for
    /// [`find`](Self::find) to report with its usual diagnostics.
    fn materialize(&mut self, binding: &ManagedMemoryBinding) -> Result<(), IoError> {
        let location = binding.descriptor().location();
        if location.init == 0 {
            return Ok(());
        }
        match PoolPosition::from_pool_index(location.pool) {
            PoolPosition::Persistent => self.persistent.materialize(&mut self.storage, binding),
            PoolPosition::Dedicated => self.dedicated.materialize(&mut self.storage, binding),
            PoolPosition::Dynamic(_) => self.materialize_dynamic(binding),
        }
    }

    fn managed_resource(
        &mut self,
        binding: ManagedMemoryBinding,
        offset_start: Option<u64>,
        offset_end: Option<u64>,
    ) -> Result<ManagedResource<Storage::Resource>, IoError> {
        let resource = self.get_resource(&binding, offset_start, offset_end)?;
        Ok(ManagedResource::new(binding, resource))
    }

    fn get_resource(
        &mut self,
        binding: &ManagedMemoryBinding,
        offset_start: Option<u64>,
        offset_end: Option<u64>,
    ) -> Result<Storage::Resource, IoError> {
        let handle = self.get_storage(binding)?;
        let size = handle.size();
        let handle = handle
            .trim_start_bytes(offset_start.unwrap_or(0))
            .and_then(|handle| handle.trim_end_bytes(offset_end.unwrap_or(0)))
            .ok_or_else(|| IoError::BindingOutOfBounds {
                reason: format!("offsets {offset_start:?} and {offset_end:?} on {size} bytes")
                    .into(),
                backtrace: BackTrace::capture(),
            })?;
        Ok(self.storage.get(&handle))
    }

    fn reserve(
        &mut self,
        size: u64,
        update: PageUpdate,
        kind: AllocationKind,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        match update {
            PageUpdate::Allow => self.reserve_after_cleanup(size, kind, failures),
            PageUpdate::AddOnly => self.reserve_adding(size, kind, failures),
            PageUpdate::Forbidden => self
                .reserve_held(size, kind, failures)
                .ok_or_else(|| page_update_forbidden(size)),
        }
    }

    /// Run the periodic cleanup, then [`reserve_adding`](Self::reserve_adding).
    ///
    /// The cleanup drives the pools' periodic deallocation. Each pool gates
    /// itself on its own deallocation period (pools without one no-op), so
    /// this is a few comparisons per reservation — without it, pages freed
    /// long ago are never returned to the driver until an explicit cleanup,
    /// which on long-running processes lets every stream's pools grow
    /// monotonically.
    fn reserve_after_cleanup(
        &mut self,
        size: u64,
        kind: AllocationKind,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        self.cleanup(Cleanup::Periodic, failures);
        self.reserve_adding(size, kind, failures)
    }

    /// Reserve `size` bytes, allocating a page when the room held has none,
    /// without releasing or renumbering any page.
    fn reserve_adding(
        &mut self,
        size: u64,
        kind: AllocationKind,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        // If this happens every nanosecond, counts overflows after 585 years, so not worth thinking too
        // hard about overflow here.
        self.reservation_count += 1;
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
            && let Some(reserved) = self.persistent.try_reserve(size, failures)
        {
            self.log_full(|| format!("Reserved memory {size} using persistent memory"));
            return Ok(reserved);
        }

        if persistent_mode || (size_match && self.persistent.view().has_size_bytes(size)) {
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

        self.log_full(|| {
            format!(
                "Reserved memory {} using dynamic pool",
                BytesFormat::from_byte_count(size)
            )
        });
        let reserved = self.reserve_dynamic(size, mapping, kind, failures)?;
        self.log_full(|| format!("Allocated a new memory page, current usage: \n{self}"));

        Ok(reserved)
    }

    /// Log the message `message` builds, under the name of this memory, at
    /// the full memory log level.
    fn log_full(&self, message: impl FnOnce() -> String) {
        self.logger.log_memory(
            |level| matches!(level, MemoryLogLevel::Full),
            || format!("[{}] {}", self.name, message()),
        );
    }

    /// Reserve `size` bytes in the room the pools already hold, the way
    /// [`reserve_adding`](Self::reserve_adding) would route them: no cleanup,
    /// no new page, nothing moved. `None` when that room cannot serve it.
    fn reserve_held(
        &mut self,
        size: u64,
        kind: AllocationKind,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        match self.mode {
            MemoryAllocationMode::Dedicated => self.dedicated.try_reserve(size, failures),
            MemoryAllocationMode::Persistent => self.persistent.try_reserve(size, failures),
            MemoryAllocationMode::Auto => self.reserve_held_auto(size, kind, failures),
        }
    }

    /// [`reserve_held`](Self::reserve_held) outside any window: the
    /// persistent pool first under the `size-match` config, then the dynamic
    /// pools.
    fn reserve_held_auto(
        &mut self,
        size: u64,
        kind: AllocationKind,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        if matches!(self.config, PersistentMemory::SizeMatch)
            && let Some(handle) = self.persistent.try_reserve(size, failures)
        {
            return Some(handle);
        }
        self.try_reserve_dynamic(size, kind, failures)
    }

    fn storage(&mut self) -> &mut Storage {
        &mut self.storage
    }

    fn memory_report(&self) -> MemoryPoolsReport {
        MemoryPoolsReport {
            dynamic: self.report_dynamic(),
            persistent: self.persistent.report(),
            dedicated: self.dedicated.report(),
        }
    }

    fn print_memory_usage(&self) {
        #[cfg(feature = "std")]
        log::info!("{}", self.memory_report().usage());
    }

    fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        assigned: ManagedMemoryHandle,
        cursor: u64,
        failures: &mut ErrorGraph,
    ) {
        let location = allocated(reserved.descriptor())
            .expect("a reservation should be placed by the memory that reserved it");
        self.pool_mut(location)
            .bind(reserved, assigned, cursor, failures);
    }

    fn failure(&self, binding: &ManagedMemoryBinding, range: Range<u64>) -> Option<FailureId> {
        self.find(binding)
            .expect("a binding asked for its failure should be placed")
            .tainted
            .failure(&range)
    }

    fn taint(
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

    fn written(
        &mut self,
        binding: &ManagedMemoryBinding,
        range: Range<u64>,
        failures: &mut ErrorGraph,
    ) {
        if let Ok(slice) = self.find_mut(binding) {
            slice.tainted.written(range, failures);
        }
    }

    /// The dynamic pool `index` names, while one is there.
    fn dynamic_pool_by_index(&self, index: u8) -> Option<&dyn SliceReserver> {
        match &self.dynamic {
            MemoryReserverDispatch::Buckets(buckets) => Some(buckets.pool_by_index(index)?),
            MemoryReserverDispatch::Adaptive(memory) => memory.pool_by_index(index),
        }
    }

    /// The dynamic pool `index` names, mutably.
    fn dynamic_pool_by_index_mut(&mut self, index: u8) -> Option<&mut dyn SliceReserver> {
        match &mut self.dynamic {
            MemoryReserverDispatch::Buckets(buckets) => Some(buckets.pool_by_index_mut(index)?),
            MemoryReserverDispatch::Adaptive(memory) => memory.pool_by_index_mut(index),
        }
    }

    /// Install real backing behind `binding` when its allocation was carved
    /// lazily. Only the adaptive pages are ever lazy.
    fn materialize_dynamic(&mut self, binding: &ManagedMemoryBinding) -> Result<(), IoError> {
        match &mut self.dynamic {
            MemoryReserverDispatch::Buckets(_) => Ok(()),
            MemoryReserverDispatch::Adaptive(memory) => {
                memory.materialize(&mut self.storage, binding)
            }
        }
    }

    /// Reserve `size` bytes on the dynamic pool that serves them.
    fn reserve_dynamic(
        &mut self,
        size: u64,
        mapping: PageMapping,
        kind: AllocationKind,
        failures: &mut ErrorGraph,
    ) -> Result<ManagedMemoryHandle, IoError> {
        match &mut self.dynamic {
            // An exclusive pool pins a page by holding its one allocation, so
            // its pins are never shared.
            MemoryReserverDispatch::Buckets(buckets) => {
                buckets.reserve(&mut self.storage, size, mapping, failures)
            }
            MemoryReserverDispatch::Adaptive(memory) => {
                memory.reserve(&mut self.storage, size, mapping, kind, failures)
            }
        }
    }

    /// Reserve `size` bytes in the room a dynamic pool already holds,
    /// without growing anything.
    fn try_reserve_dynamic(
        &mut self,
        size: u64,
        kind: AllocationKind,
        failures: &mut ErrorGraph,
    ) -> Option<ManagedMemoryHandle> {
        match &mut self.dynamic {
            MemoryReserverDispatch::Buckets(buckets) => buckets.try_reserve(size, failures),
            MemoryReserverDispatch::Adaptive(memory) => memory.try_reserve(size, kind, failures),
        }
    }

    /// Release what the dynamic pools no longer need.
    fn cleanup_dynamic(&mut self, cleanup: Cleanup, failures: &mut ErrorGraph) {
        match &mut self.dynamic {
            MemoryReserverDispatch::Buckets(buckets) => {
                buckets.cleanup(&mut self.storage, self.reservation_count, cleanup, failures)
            }
            MemoryReserverDispatch::Adaptive(memory) => {
                memory.cleanup(&mut self.storage, self.reservation_count, cleanup, failures)
            }
        }
    }

    /// A report per dynamic pool held, in the order allocations are routed
    /// through them.
    fn report_dynamic(&self) -> Vec<MemoryPoolReport> {
        match &self.dynamic {
            MemoryReserverDispatch::Buckets(buckets) => buckets.report(),
            MemoryReserverDispatch::Adaptive(memory) => memory.report(),
        }
    }
}

impl MemoryReserverDispatch {
    /// The pools `config` lays out on a device with `properties`.
    fn new(
        properties: &MemoryDeviceProperties,
        config: MemoryConfiguration,
        logger: Arc<ServerLogger>,
        name: String,
    ) -> Self {
        match config {
            MemoryConfiguration::ExclusivePages => {
                Self::Buckets(log_buckets(BucketReserver::new(properties), &logger, &name))
            }
            // `Adaptive`, which only exists where `cubecl-runtime` allows
            // sub-slicing. That crate alone decides, so this one cannot name
            // the variant without disagreeing with it once the feature is
            // turned on there directly.
            #[allow(unreachable_patterns)]
            _ => Self::Adaptive(Box::new(AdaptiveReserver::new(properties, logger, name))),
        }
    }
}

/// `buckets`, once logged as the pools the memory `name` uses.
fn log_buckets(buckets: BucketReserver, logger: &ServerLogger, name: &str) -> BucketReserver {
    logger.log_memory(
        |level| !matches!(level, MemoryLogLevel::Disabled),
        || format!("[{name}] Using memory pools:\n{buckets}"),
    );
    buckets
}

/// The mode the persistent-memory `config` puts in force outside any window.
fn configured_mode(config: &PersistentMemory) -> MemoryAllocationMode {
    match config {
        PersistentMemory::Enabled | PersistentMemory::SizeMatch | PersistentMemory::Disabled => {
            MemoryAllocationMode::Auto
        }
        PersistentMemory::Enforced => MemoryAllocationMode::Persistent,
    }
}

/// The refusal of a reservation of `size` bytes the room held cannot serve,
/// when no page may be added.
fn page_update_forbidden(size: u64) -> IoError {
    IoError::PageUpdateForbidden {
        size,
        backtrace: BackTrace::capture(),
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

/// Check that `slice` still holds the allocation behind `binding`: a binding
/// names its allocation's current location, so a slice holding another one
/// is a broken assumption, never an answer.
///
/// # Panics
///
/// When `slice` holds another allocation than `binding`'s.
fn owned_by(slice: &Slice, binding: &ManagedMemoryBinding) {
    assert!(
        slice.descriptor() == binding.descriptor(),
        "a binding should name its allocation's current location"
    );
}

impl<Storage: StorageAllocator> core::fmt::Display for MemoryReserver<Storage> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("\n# MemoryManagement\n\n")?;
        f.write_fmt(format_args!(" - name: {:?}\n", self.name))?;
        f.write_fmt(format_args!("\n## Persistent\n\n{}", self.persistent))?;
        f.write_str("\n## Dynamic\n\n")?;
        match &self.dynamic {
            MemoryReserverDispatch::Buckets(buckets) => write!(f, "{buckets}")?,
            MemoryReserverDispatch::Adaptive(memory) => write!(f, "{memory}")?,
        }
        let memory_usage = self.memory_report().usage();
        f.write_fmt(format_args!("\n## Summary\n\n{memory_usage}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(not(exclusive_memory_only))]
    use crate::memory_management::MemoryPoolKind;
    use crate::storage::BytesStorage;

    const MIB: u64 = 1024 * 1024;
    const DUMMY_MEM_PROPS: MemoryDeviceProperties = MemoryDeviceProperties::new(128 * MIB, 32);
    /// The page the `Adaptive` preset's metadata pool carves.
    #[cfg(not(exclusive_memory_only))]
    const METADATA_PAGE_BYTES: u64 = 8 * MIB;

    fn memory(config: MemoryConfiguration) -> MemoryManagement<BytesStorage> {
        memory_on(&DUMMY_MEM_PROPS, config)
    }

    fn memory_on(
        properties: &MemoryDeviceProperties,
        config: MemoryConfiguration,
    ) -> MemoryManagement<BytesStorage> {
        MemoryManagement::from_configuration(
            BytesStorage::default(),
            properties,
            config,
            Arc::new(ServerLogger::default()),
            MemoryManagementOptions::new("test"),
        )
    }

    fn reserve(memory: &mut MemoryManagement<BytesStorage>, size: u64) -> ManagedMemoryHandle {
        memory
            .reserve(size, PageUpdate::Allow, &mut ErrorGraph::default())
            .expect("the memory should serve the size")
    }

    fn launch_failure(failures: &mut ErrorGraph) -> FailureId {
        failures.insert(ServerError::Generic {
            reason: "launch".into(),
            backtrace: Default::default(),
        })
    }

    /// The adaptive preset's small pool is for metadata: an allocation close
    /// to its page size still goes to the adaptive pool, which sizes pages to
    /// what it serves, rather than onto the metadata pages.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn adaptive_preset_routes_near_page_sizes_to_the_adaptive_pool() {
        let mut memory = memory(MemoryConfiguration::Adaptive);
        let _near = reserve(&mut memory, 7 * MIB);

        let served = memory
            .memory_report()
            .dynamic
            .into_iter()
            .find(|pool| pool.largest_alloc_bytes == 7 * MIB)
            .expect("some pool should have served the allocation");
        assert!(
            matches!(served.kind, MemoryPoolKind::Adaptive { .. }),
            "served by {:?}",
            served.kind
        );
    }

    #[test_log::test]
    fn a_binding_whose_offsets_reach_past_its_memory_is_refused() {
        let mut memory_management = memory(MemoryConfiguration::Adaptive);
        let handle = reserve(&mut memory_management, 16);
        let resource = memory_management.get_resource(handle.binding(), Some(12), Some(8));
        assert!(matches!(resource, Err(IoError::BindingOutOfBounds { .. })));
    }

    /// A binding's offsets trim the resource it resolves to.
    #[test_log::test]
    fn a_binding_resolves_to_its_trimmed_resource() {
        let mut memory_management = memory(MemoryConfiguration::Adaptive);
        let handle = reserve(&mut memory_management, 16);

        let resource = memory_management
            .get_resource(handle.binding(), Some(4), Some(4))
            .expect("the offsets should fit the binding");

        assert_eq!(resource.read().len(), 8);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn test_memory_usage() {
        let mut memory_management = memory(MemoryConfiguration::Adaptive);
        let handle = reserve(&mut memory_management, 100);
        let usage = memory_management.memory_report().usage();

        assert_eq!(usage.in_use_bytes, 100);
        // A metadata-sized allocation is carved from one metadata page.
        assert_eq!(usage.reserved_bytes, METADATA_PAGE_BYTES);

        // Drop and re-alloc.
        drop(handle);
        let _handle = reserve(&mut memory_management, 100);
        let usage_new = memory_management.memory_report().usage();
        assert_eq!(usage, usage_new);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn find_uninit_binding_returns_not_found() {
        let mut memory_management = memory(MemoryConfiguration::Adaptive);

        // Even with a live page at index 0, a never-initialized descriptor must
        // not resolve to it.
        let _live = reserve(&mut memory_management, 512);

        let binding = ManagedMemoryHandle::new().binding();
        assert!(matches!(
            memory_management.get_cursor(binding),
            Err(IoError::NotFound { .. })
        ));
    }

    /// A binding naming a slice another allocation took is never resolved to
    /// that allocation's data.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    #[should_panic(expected = "a binding should name its allocation's current location")]
    fn a_stale_binding_panics() {
        let mut memory_management = memory(MemoryConfiguration::Adaptive);
        let reserved = reserve(&mut memory_management, 512);
        let stale = reserved.clone();
        memory_management.bind(
            reserved,
            ManagedMemoryHandle::new(),
            0,
            &mut ErrorGraph::default(),
        );

        let _ = memory_management.get_cursor(stale.binding());
    }

    /// Binding a handle no reservation placed panics.
    #[test_log::test]
    #[should_panic(expected = "a reservation should be placed by the memory that reserved it")]
    fn binding_an_unplaced_handle_panics() {
        let mut memory_management = memory(MemoryConfiguration::ExclusivePages);

        memory_management.bind(
            ManagedMemoryHandle::new(),
            ManagedMemoryHandle::new(),
            0,
            &mut ErrorGraph::default(),
        );
    }

    /// A location naming a dynamic pool the layout does not have panics.
    #[test_log::test]
    #[should_panic(expected = "a location should name a pool of the memory that placed it")]
    fn a_location_naming_no_pool_panics() {
        let memory_management = memory(MemoryConfiguration::ExclusivePages);
        let handle = ManagedMemoryHandle::new();
        handle
            .descriptor()
            .update_location(MemoryLocation::new(200, 0, 0));

        let _ = memory_management.get_cursor(handle.binding());
    }

    /// A taint and a write of a binding never placed leave everything alone.
    #[test_log::test]
    fn the_taint_bookkeeping_leaves_unknown_bindings_alone() {
        let mut memory_management = memory(MemoryConfiguration::ExclusivePages);
        let mut failures = ErrorGraph::default();
        let failure = launch_failure(&mut failures);
        let unknown = ManagedMemoryHandle::new().binding();

        memory_management.taint(&unknown, 0..8, failure, &mut failures);
        memory_management.written(&unknown, 0..8, &mut failures);

        failures.prune(failure);
        assert!(failures.is_empty(), "nothing took the failure");
    }

    /// The failure of a binding never placed is a broken assumption.
    #[test_log::test]
    #[should_panic(expected = "a binding asked for its failure should be placed")]
    fn the_failure_of_an_unplaced_binding_panics() {
        let memory_management = memory(MemoryConfiguration::ExclusivePages);

        memory_management.failure(&ManagedMemoryHandle::new().binding(), 0..8);
    }

    /// A taint claims the bytes it names until they are written again.
    #[test_log::test]
    fn a_taint_claims_its_bytes_until_they_are_written() {
        let mut memory_management = memory(MemoryConfiguration::ExclusivePages);
        let mut failures = ErrorGraph::default();
        let failure = launch_failure(&mut failures);
        let handle = reserve(&mut memory_management, 64);
        let binding = handle.clone().binding();

        memory_management.taint(&binding, 0..64, failure, &mut failures);
        assert_eq!(memory_management.failure(&binding, 8..16), Some(failure));
        memory_management.written(&binding, 0..64, &mut failures);

        assert_eq!(memory_management.failure(&binding, 8..16), None);
        assert!(failures.is_empty());
    }

    /// A reservation forbidden to update the pages is refused when the room
    /// held cannot serve it.
    #[test_log::test]
    fn a_forbidden_page_update_is_refused_without_room() {
        let mut memory_management = memory(MemoryConfiguration::ExclusivePages);

        let refused =
            memory_management.reserve(1024, PageUpdate::Forbidden, &mut ErrorGraph::default());

        assert!(matches!(refused, Err(IoError::PageUpdateForbidden { .. })));
    }

    /// A dedicated window allocates outside every pool.
    #[test_log::test]
    fn a_dedicated_window_reserves_its_own_allocation() {
        let mut memory_management = memory(MemoryConfiguration::ExclusivePages);
        memory_management.mode(MemoryAllocationMode::Dedicated);

        let first = memory_management
            .reserve(1024, PageUpdate::AddOnly, &mut ErrorGraph::default())
            .expect("a dedicated allocation should be made");

        assert_eq!(first.descriptor().location().pool, DEDICATED_POOL_POS);
    }

    /// A forbidden page update in a dedicated window reuses a freed dedicated
    /// slice of the same size.
    #[test_log::test]
    fn a_dedicated_window_reuses_a_freed_dedicated_slice() {
        let mut memory_management = memory(MemoryConfiguration::ExclusivePages);
        memory_management.mode(MemoryAllocationMode::Dedicated);
        drop(
            memory_management
                .reserve(1024, PageUpdate::AddOnly, &mut ErrorGraph::default())
                .expect("a dedicated allocation should be made"),
        );

        let reused = memory_management
            .reserve(1024, PageUpdate::Forbidden, &mut ErrorGraph::default())
            .expect("the freed dedicated slice should serve it");

        assert_eq!(reused.descriptor().location().pool, DEDICATED_POOL_POS);
    }

    #[test_log::test]
    fn held_binding_survives_explicit_cleanup_renumber() {
        // Exclusive pages: one page per allocation, so freeing two of three
        // leaves the pool with pages to drop and one to renumber.
        let mut memory_management = memory(MemoryConfiguration::ExclusivePages);

        let handle_a = reserve(&mut memory_management, 1024);
        let handle_b = reserve(&mut memory_management, 1024);
        let handle_c = reserve(&mut memory_management, 1024);

        let binding_b = handle_b.binding();
        let reserved = memory_management.memory_report().usage().reserved_bytes;
        drop(handle_a);
        drop(handle_c);

        // Deallocates the two free pages and renumbers the surviving one.
        memory_management.cleanup(Cleanup::Explicit, &mut ErrorGraph::default());

        assert!(memory_management.get_cursor(binding_b.clone()).is_ok());
        assert!(memory_management.get_storage(binding_b).is_ok());
        assert!(
            memory_management.memory_report().usage().reserved_bytes < reserved,
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
        let mut memory_management = memory(MemoryConfiguration::Adaptive);

        memory_management.mode(MemoryAllocationMode::Persistent); // the load's window
        memory_management.mode(MemoryAllocationMode::Persistent); // one parameter's window
        memory_management.mode(MemoryAllocationMode::Auto); // that parameter is done

        // Still inside the load's window: the allocation must be persistent.
        let weight = reserve(&mut memory_management, 1024);
        let report = memory_management.memory_report();
        assert_eq!(
            report.persistent.usage.in_use_bytes, 1024,
            "an allocation inside the outer window is persistent"
        );
        assert_eq!(
            report
                .dynamic
                .iter()
                .map(|pool| pool.peak_page_count)
                .sum::<u64>(),
            0,
            "nothing leaked into the dynamic pools"
        );

        // The outer window closes; ordinary allocations are dynamic again.
        memory_management.mode(MemoryAllocationMode::Auto);
        let _transient = reserve(&mut memory_management, 1024);
        assert_eq!(
            memory_management
                .memory_report()
                .dynamic
                .iter()
                .map(|pool| pool.peak_page_count)
                .sum::<u64>(),
            1
        );

        drop(weight);
    }

    /// Closing a window none is open in keeps the configured mode.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn closing_no_window_keeps_the_configured_mode() {
        let mut memory_management = memory(MemoryConfiguration::Adaptive);

        memory_management.mode(MemoryAllocationMode::Auto);
        let _transient = reserve(&mut memory_management, 1024);

        assert_eq!(
            memory_management
                .memory_report()
                .persistent
                .usage
                .in_use_bytes,
            0
        );
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn alloc_two_chunks_on_one_page() {
        let mut memory_management = memory(MemoryConfiguration::Adaptive);

        let alloc_size = 512;
        let _handle = reserve(&mut memory_management, alloc_size);
        let _new_handle = reserve(&mut memory_management, alloc_size);

        let usage = memory_management.memory_report().usage();
        assert_eq!(usage.allocation_count, 2);
        assert_eq!(usage.in_use_bytes, alloc_size * 2);
        // One page, not two: both slices were carved from the same one.
        assert_eq!(usage.reserved_bytes, METADATA_PAGE_BYTES);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn alloc_reuses_storage() {
        // If no storage is re-used, this will allocate two pages.
        let mut memory_management = memory(MemoryConfiguration::Adaptive);

        let alloc_size = 512;
        let handle = reserve(&mut memory_management, alloc_size);
        drop(handle);
        let _new_handle = reserve(&mut memory_management, alloc_size);

        let usage = memory_management.memory_report().usage();
        assert_eq!(usage.allocation_count, 1);
        assert_eq!(usage.in_use_bytes, alloc_size);
        assert_eq!(usage.reserved_bytes, METADATA_PAGE_BYTES);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn alloc_allocs_new_storage() {
        let mut memory_management = memory(MemoryConfiguration::Adaptive);

        // Three quarters of the adaptive pool's smallest page each: the second
        // one does not fit beside the first, so the pool has to grow a page.
        let page_size = 2 * MIB;
        let alloc_size = page_size / 4 * 3;
        let _handle = reserve(&mut memory_management, alloc_size);
        let _new_handle = reserve(&mut memory_management, alloc_size);

        let usage = memory_management.memory_report().usage();
        assert_eq!(usage.allocation_count, 2);
        assert_eq!(usage.in_use_bytes, alloc_size * 2);
        assert_eq!(usage.reserved_bytes, page_size * 2);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn alloc_respects_alignment_size() {
        let mut memory_management = memory_on(
            &MemoryDeviceProperties::new(DUMMY_MEM_PROPS.max_page_size, 50),
            MemoryConfiguration::Adaptive,
        );
        let alloc_size = 40;
        let _handle = reserve(&mut memory_management, alloc_size);
        let _new_handle = reserve(&mut memory_management, alloc_size);
        let usage = memory_management.memory_report().usage();
        // Each slice should be aligned to 50 bytes, so 10 padding bytes.
        assert_eq!(usage.padding_bytes, 10 * 2);
    }

    /// Each size lands on the pool that serves it: metadata churn stays on the
    /// metadata pages, and the workload gets pages sized to what it asks for.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn allocs_on_correct_page() {
        let mut memory_management = memory(MemoryConfiguration::Adaptive);

        let alloc_sizes = [4096, 4 * MIB];
        let handles = alloc_sizes.map(|size| reserve(&mut memory_management, size));

        assert_ne!(
            handles[0].descriptor().location().pool,
            handles[1].descriptor().location().pool,
            "metadata and the workload must not share a pool"
        );

        let usage = memory_management.memory_report().usage();
        // Total memory should be size of all pages, and no more: one metadata
        // page, and one adaptive page grown to the 4 MiB allocation.
        assert_eq!(usage.in_use_bytes, alloc_sizes.iter().sum::<u64>());
        assert_eq!(usage.reserved_bytes, METADATA_PAGE_BYTES + 5 * MIB);
    }

    // The motivating use case: allocations from different "sequence-length
    // ranges" reuse the same arena instead of each landing in its own
    // size-bucketed pool that keeps a separate reservation.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn single_arena_reuses_across_sizes() {
        let mut memory_management = memory(MemoryConfiguration::Adaptive);

        // A "small seq" allocation, then freed.
        let small = reserve(&mut memory_management, 128 * 1024);
        drop(small);
        // A "large seq" allocation must reuse the same arena page.
        let large = reserve(&mut memory_management, 512 * 1024);

        let usage = memory_management.memory_report().usage();
        assert_eq!(
            usage.reserved_bytes,
            2 * MIB,
            "both sizes must share a single arena page"
        );
        assert_eq!(usage.allocation_count, 1);
        drop(large);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn allocate_deallocate_reallocate() {
        let mut memory_management = memory(MemoryConfiguration::Adaptive);
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| reserve(&mut memory_management, 1000 * (i + 1)))
            .collect();
        let usage_before = memory_management.memory_report().usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| reserve(&mut memory_management, 1000 * (i + 1)))
            .collect();
        let usage_after = memory_management.memory_report().usage();
        assert_eq!(usage_before.allocation_count, usage_after.allocation_count);
        assert_eq!(usage_before.in_use_bytes, usage_after.in_use_bytes);
        // Usage after can actually be _less_ because of defragging.
        assert!(usage_before.reserved_bytes >= usage_after.reserved_bytes);
    }

    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn test_fragmentation_resistance() {
        let mut memory_management = memory(MemoryConfiguration::Adaptive);
        // Allocate a mix of small and large chunks
        let sizes = [50, 1000, 100, 5000, 200, 10000, 300];
        let handles: Vec<_> = sizes
            .iter()
            .map(|&size| reserve(&mut memory_management, size))
            .collect();
        let usage_before = memory_management.memory_report().usage();
        // Deallocate every other allocation
        for i in (0..handles.len()).step_by(2) {
            drop(handles[i].clone());
        }
        // Reallocate similar sizes
        for &size in &sizes[0..sizes.len() / 2] {
            reserve(&mut memory_management, size);
        }
        let usage_after = memory_management.memory_report().usage();
        // Check that we haven't increased our memory usage significantly
        assert!(usage_after.reserved_bytes <= (usage_before.reserved_bytes as f64 * 1.1) as u64);
    }

    // Test pools without slices. More or less same as tests above.
    #[test_log::test]
    fn noslice_alloc_two_chunk() {
        let mut memory_management = memory(MemoryConfiguration::ExclusivePages);

        let alloc_size = 512;
        let _handle = reserve(&mut memory_management, alloc_size);
        let _new_handle = reserve(&mut memory_management, alloc_size);

        let usage = memory_management.memory_report().usage();
        assert_eq!(usage.allocation_count, 2);
        assert_eq!(usage.in_use_bytes, alloc_size * 2);
        assert!(usage.reserved_bytes >= alloc_size * 2);
    }

    #[test_log::test]
    fn noslice_alloc_allocs_new_storage() {
        let mut memory_management = memory(MemoryConfiguration::ExclusivePages);

        let alloc_size = 768;
        let _handle = reserve(&mut memory_management, alloc_size);
        let _new_handle = reserve(&mut memory_management, alloc_size);
        let usage = memory_management.memory_report().usage();
        assert_eq!(usage.allocation_count, 2);
        assert_eq!(usage.in_use_bytes, alloc_size * 2);
        assert!(usage.reserved_bytes >= alloc_size * 2);
    }

    #[test_log::test]
    fn noslice_alloc_respects_alignment_size() {
        let mut memory_management = memory_on(
            &MemoryDeviceProperties::new(DUMMY_MEM_PROPS.max_page_size, 50),
            MemoryConfiguration::ExclusivePages,
        );
        let alloc_size = 40;
        let _handle = reserve(&mut memory_management, alloc_size);
        let _new_handle = reserve(&mut memory_management, alloc_size);
        let usage = memory_management.memory_report().usage();
        // Each slice should be aligned to 50 bytes, so 10 padding bytes.
        assert_eq!(usage.padding_bytes, 10 * 2);
    }

    #[test_log::test]
    fn noslice_allocs_on_correct_page() {
        let mut memory_management = memory_on(
            &MemoryDeviceProperties::new(DUMMY_MEM_PROPS.max_page_size, 10),
            MemoryConfiguration::ExclusivePages,
        );
        // Allocate one thing on each page.
        let alloc_sizes = [50, 150, 250, 350];
        let _handles = alloc_sizes.map(|s| reserve(&mut memory_management, s));
        let usage = memory_management.memory_report().usage();
        // Total memory should be size of all pages, and no more.
        assert_eq!(usage.in_use_bytes, alloc_sizes.iter().sum::<u64>());
    }

    #[test_log::test]
    fn a_guarded_allocation_is_never_handed_out_again() {
        let mut memory_management = memory(MemoryConfiguration::ExclusivePages);

        let recorded = reserve(&mut memory_management, 1024);
        let guard = memory_management
            .guard(recorded.descriptor().location())
            .expect("the pool should hold the page");
        drop(recorded);

        // Freed, but guarded: the next allocation of that size needs a page
        // of its own, and a cleanup leaves the guarded one where it is.
        let before = memory_management.memory_report().usage().reserved_bytes;
        let other = reserve(&mut memory_management, 1024);
        let after = memory_management.memory_report().usage().reserved_bytes;
        assert!(after > before, "the guarded page was handed out again");
        memory_management.cleanup(Cleanup::Explicit, &mut ErrorGraph::default());
        assert_eq!(
            memory_management.memory_report().usage().reserved_bytes,
            after,
            "the guarded page was released"
        );

        drop(guard);
        drop(other);
        memory_management.cleanup(Cleanup::Explicit, &mut ErrorGraph::default());
        assert_eq!(memory_management.memory_report().usage().reserved_bytes, 0);
    }

    /// The room a graph's temporaries left free on a page it pinned serves
    /// another capture, and only a capture: work outside one outlives the next
    /// replay, which would write over it. A slice a handle holds is never
    /// handed out, pinned or not.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn a_pinned_page_is_carved_only_by_a_capture() {
        let mut memory = memory(MemoryConfiguration::Adaptive);
        let share = |memory: &mut MemoryManagement<BytesStorage>, update| {
            memory.reserve_as(
                MIB,
                update,
                AllocationKind::GraphIntermediate,
                &mut ErrorGraph::default(),
            )
        };
        let temporary = share(&mut memory, PageUpdate::Allow).expect("a capture should reserve");
        let _pin = memory
            .guard(temporary.descriptor().location())
            .expect("the pool should hold the page");
        drop(temporary);
        let pinned = memory.memory_report().usage().reserved_bytes;

        let later = share(&mut memory, PageUpdate::Forbidden)
            .expect("a capture should carve the room the pin keeps");
        assert_eq!(memory.memory_report().usage().reserved_bytes, pinned);

        let _eager = reserve(&mut memory, MIB);
        assert!(
            memory.memory_report().usage().reserved_bytes > pinned,
            "work outside a capture took a page of its own"
        );

        let held = share(&mut memory, PageUpdate::Forbidden)
            .expect("the pinned page should have room past the held slice");
        assert_ne!(
            held.descriptor().location(),
            later.descriptor().location(),
            "a slice a handle holds is never handed out again"
        );
    }

    #[test_log::test]
    fn a_guard_leaves_in_place_updates_alone() {
        let mut memory_management = memory(MemoryConfiguration::ExclusivePages);

        let cache = reserve(&mut memory_management, 1024);
        let _guard = memory_management.guard(cache.descriptor().location());

        assert!(
            cache.can_mut(),
            "a graph replaying against a buffer must not stop it being updated in place"
        );
    }

    /// A dedicated allocation is its own device allocation, returned on the
    /// tick after it is freed, whatever mode encloses it.
    #[test_log::test]
    #[cfg(not(exclusive_memory_only))]
    fn dedicated_allocations_are_released_once_freed() {
        /// The floor the `Adaptive` preset carves its pages at.
        const FLOOR_BYTES: u64 = 2 * MIB;
        let mut memory = memory_on(
            &MemoryDeviceProperties::new(1024 * MIB, 32),
            MemoryConfiguration::Adaptive,
        );

        memory.mode(MemoryAllocationMode::Persistent);
        memory.mode(MemoryAllocationMode::Dedicated);
        let probe = reserve(&mut memory, 200 * MIB);
        memory.mode(MemoryAllocationMode::Auto);
        let weight = reserve(&mut memory, MIB);
        memory.mode(MemoryAllocationMode::Auto);

        assert_eq!(
            memory.memory_report().persistent.usage.in_use_bytes,
            MIB,
            "closing the dedicated window restores the persistent one"
        );
        let report = memory.memory_report();
        assert_eq!(report.usage().reserved_bytes, 200 * MIB + MIB);
        assert_eq!(
            (report.dedicated.kind, report.dedicated.page_count),
            (MemoryPoolKind::Direct, 1),
            "the dedicated allocation is reported as its own pool"
        );

        drop(probe);
        let _tick = reserve(&mut memory, MIB);
        assert_eq!(
            memory.memory_report().usage().reserved_bytes,
            MIB + FLOOR_BYTES,
            "the probe buffer is gone; the weight and one adaptive page remain"
        );
        drop(weight);
    }

    #[test_log::test]
    fn noslice_allocate_deallocate_reallocate() {
        let mut memory_management = memory(MemoryConfiguration::ExclusivePages);
        // Allocate a bunch
        let handles: Vec<_> = (0..5)
            .map(|i| reserve(&mut memory_management, 1000 * (i + 1)))
            .collect();
        let usage_before = memory_management.memory_report().usage();
        // Deallocate
        drop(handles);
        // Reallocate
        let _new_handles: Vec<_> = (0..5)
            .map(|i| reserve(&mut memory_management, 1000 * (i + 1)))
            .collect();
        let usage_after = memory_management.memory_report().usage();
        assert_eq!(usage_before.allocation_count, usage_after.allocation_count);
        assert_eq!(usage_before.in_use_bytes, usage_after.in_use_bytes);
        assert_eq!(usage_before.reserved_bytes, usage_after.reserved_bytes);
    }

    /// A memory of buckets never has anything outdated.
    #[test_log::test]
    fn buckets_never_need_a_relocation() {
        let mut memory_management = memory(MemoryConfiguration::ExclusivePages);
        let _held = reserve(&mut memory_management, 1024);

        assert_eq!(memory_management.relocation_need(), RelocationNeed::Nothing);
    }
}
