use cubecl_core::server::{IoError, ServerError};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;
use cubecl_server::memory_management::ManagedMemoryBinding;
use cubecl_server::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use std::{
    num::NonZeroU64,
    ops::Deref,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};
use wgpu::BufferUsages;

/// Minimum buffer size in bytes. The WebGPU spec requires buffer sizes > 0, and shaders
/// declare typed arrays (e.g. `array<vec4<f32>>`) that impose a minimum binding size.
/// 32 bytes covers the largest possible binding type (`vec4<f64>`).
const MIN_BUFFER_SIZE: u64 = 32;

/// No host mapping and no queued GPU use: the allocation is free.
const ACCESS_GPU_IDLE: usize = 0;
/// A host mapping is active; the flag occupies the sign bit so the GPU
/// reservation count below it can never reach it.
const ACCESS_HOST_MAPPED: usize = 1 << (usize::BITS - 1);
/// The low bits count queued GPU reservations.
const ACCESS_GPU_COUNT_MASK: usize = ACCESS_HOST_MAPPED - 1;

/// Hands out the process-unique identity every allocation's access state carries.
static NEXT_ALLOCATION_ID: AtomicU64 = AtomicU64::new(1);

/// Why an allocation cannot be handed to the host, or to the GPU, right now.
///
/// Host and GPU access to one allocation exclude each other: a mapping keeps a
/// buffer out of every launch until it is dropped, and queued GPU work keeps it
/// un-mappable until that work is submitted.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HostAccessError {
    /// The allocation was created without host-visible primary memory enabled.
    DeviceLocalAllocation,
    /// Another host mapping is already active for the physical allocation.
    OverlappingHostMapping,
    /// GPU work has reserved the allocation and has not yet been submitted.
    GpuAccessInProgress,
    /// GPU use was requested while a host mapping is active.
    MappedForHost,
    /// WGPU failed to complete the asynchronous map callback.
    MapCallbackFailure(String),
    /// WGPU failed while polling the device for mapping completion.
    DevicePollFailure(String),
    /// Synchronous host mapping is unavailable on this target.
    UnsupportedTarget,
}

impl core::fmt::Display for HostAccessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::DeviceLocalAllocation => {
                f.write_str("the WGPU allocation is device-local and cannot be mapped by the host")
            }
            Self::OverlappingHostMapping => {
                f.write_str("the WGPU allocation already has an active host mapping")
            }
            Self::GpuAccessInProgress => f.write_str(
                "the WGPU allocation has GPU work pending submission and cannot be mapped by the host",
            ),
            Self::MappedForHost => f.write_str(
                "the WGPU allocation is mapped for host access and cannot be used by the GPU",
            ),
            Self::MapCallbackFailure(reason) => {
                write!(f, "the WGPU map callback failed: {reason}")
            }
            Self::DevicePollFailure(reason) => {
                write!(
                    f,
                    "polling the WGPU device for host mapping failed: {reason}"
                )
            }
            Self::UnsupportedTarget => {
                f.write_str("synchronous host mapping is unsupported on this target")
            }
        }
    }
}

impl std::error::Error for HostAccessError {}

impl From<HostAccessError> for ServerError {
    fn from(error: HostAccessError) -> Self {
        Self::Generic {
            reason: error.to_string(),
            backtrace: BackTrace::capture(),
        }
    }
}

/// What one physical allocation's host and GPU access is doing right now.
///
/// The state and the allocation id live behind one [`Arc`] so every
/// [`WgpuResource`], mapped guard and queued reservation that names the
/// allocation sees the same state.
#[derive(Debug)]
struct AllocationAccessInner {
    allocation_id: u64,
    state: AtomicUsize,
    /// Whether the pool this allocation came from is host-visible.
    host_visible: bool,
    /// The device to poll for a mapping; `None` for a resource a caller built
    /// around a buffer of its own.
    device: Option<wgpu::Device>,
}

#[derive(Clone, Debug)]
struct AllocationAccess {
    inner: Arc<AllocationAccessInner>,
}

impl AllocationAccess {
    fn new(host_visible: bool, device: Option<wgpu::Device>) -> Self {
        let allocation_id = NEXT_ALLOCATION_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("WGPU allocation ID overflowed");

        Self {
            inner: Arc::new(AllocationAccessInner {
                allocation_id,
                state: AtomicUsize::new(ACCESS_GPU_IDLE),
                host_visible,
                device,
            }),
        }
    }

    fn allocation_id(&self) -> u64 {
        self.inner.allocation_id
    }

    /// Takes the allocation for the host, failing when the GPU holds it.
    ///
    /// # Errors
    ///
    /// [`HostAccessError::OverlappingHostMapping`] when this allocation is
    /// already mapped, [`HostAccessError::GpuAccessInProgress`] when queued GPU
    /// work still names it.
    fn acquire_host(&self) -> Result<HostAccessToken, HostAccessError> {
        self.inner
            .state
            .compare_exchange(
                ACCESS_GPU_IDLE,
                ACCESS_HOST_MAPPED,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .map_err(|state| {
                if state == ACCESS_HOST_MAPPED {
                    HostAccessError::OverlappingHostMapping
                } else {
                    HostAccessError::GpuAccessInProgress
                }
            })?;

        Ok(HostAccessToken {
            access: Some(self.inner.clone()),
        })
    }

    /// Checks that the allocation can be used by the GPU.
    ///
    /// # Errors
    ///
    /// [`HostAccessError::MappedForHost`] while a host mapping is active.
    fn ensure_gpu_access(&self) -> Result<(), HostAccessError> {
        if self.inner.state.load(Ordering::Acquire) == ACCESS_HOST_MAPPED {
            Err(HostAccessError::MappedForHost)
        } else {
            Ok(())
        }
    }

    /// Reserves the allocation for GPU work that has not been submitted yet.
    ///
    /// # Errors
    ///
    /// [`HostAccessError::MappedForHost`] while a host mapping is active.
    fn acquire_gpu(&self) -> Result<GpuAccessToken, HostAccessError> {
        let mut state = self.inner.state.load(Ordering::Acquire);
        loop {
            if state == ACCESS_HOST_MAPPED {
                return Err(HostAccessError::MappedForHost);
            }
            assert_ne!(
                state, ACCESS_GPU_COUNT_MASK,
                "too many simultaneous WGPU GPU reservations"
            );
            match self.inner.state.compare_exchange_weak(
                state,
                state + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Ok(GpuAccessToken {
                        access: Some(self.inner.clone()),
                        lease: None,
                    });
                }
                Err(current) => state = current,
            }
        }
    }
}

/// RAII reservation holding queued GPU use against host mapping until it is
/// submitted.
///
/// The reservation keeps the slice of the pool the resource names alive (the
/// lease), so a mapping cannot see the bytes reused under it.
#[doc(hidden)]
#[derive(Debug)]
pub struct GpuAccessToken {
    access: Option<Arc<AllocationAccessInner>>,
    lease: Option<ManagedMemoryBinding>,
}

impl Drop for GpuAccessToken {
    fn drop(&mut self) {
        if let Some(access) = self.access.take() {
            let previous = access.state.fetch_sub(1, Ordering::Release);
            debug_assert!(previous > 0 && previous < ACCESS_HOST_MAPPED);
        }
    }
}

/// RAII marker for an active host mapping. Dropping it frees the allocation.
#[derive(Debug)]
struct HostAccessToken {
    access: Option<Arc<AllocationAccessInner>>,
}

impl Drop for HostAccessToken {
    fn drop(&mut self) {
        if let Some(access) = self.access.take() {
            access.state.store(ACCESS_GPU_IDLE, Ordering::Release);
        }
    }
}

/// A buffer's persistent host mapping. Crate private: writes through it need an idle queue.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HostPtr(pub(crate) NonNull<u8>);

// SAFETY: Only the stream owning the buffer writes through it.
unsafe impl Send for HostPtr {}
unsafe impl Sync for HostPtr {}

/// One wgpu buffer the storage allocated, with the access state the resources
/// cut from it share.
#[derive(Debug)]
pub struct WgpuMemory {
    /// The wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// The buffer device address, if supported
    pub address: Option<NonZeroU64>,
    /// The buffer host mapping, if host visible.
    pub(crate) host_ptr: Option<HostPtr>,
    allocation_access: AllocationAccess,
}

impl WgpuMemory {
    /// A buffer the storage allocated for a device-local pool.
    pub(crate) fn device_local(
        buffer: wgpu::Buffer,
        address: Option<NonZeroU64>,
        host_ptr: Option<HostPtr>,
    ) -> Self {
        Self {
            buffer,
            address,
            host_ptr,
            allocation_access: AllocationAccess::new(false, None),
        }
    }
}

/// The memory resource that can be allocated for wgpu.
///
/// A resource names a slice of one physical allocation. Host access to that
/// allocation is exclusive with GPU use of it: [`Self::map_read`] and
/// [`Self::map_write`] refuse to run while queued GPU work still names the
/// allocation, and [`Self::ensure_gpu_access`] refuses GPU use while a mapping
/// is alive. Every resource cloned from the same allocation shares that state,
/// and [`Self::allocation_id`] names the allocation they share.
#[derive(Debug, Clone)]
pub struct WgpuResource {
    /// The wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// The buffer device address, if supported
    pub address: Option<NonZeroU64>,
    /// The buffer host mapping, if host visible.
    pub(crate) host_ptr: Option<HostPtr>,
    /// The buffer offset.
    pub offset: u64,
    /// The size of the resource.
    ///
    /// # Notes
    ///
    /// The result considers the offset.
    pub size: u64,
    /// The access state shared by everything cut from this allocation.
    allocation_access: AllocationAccess,
    /// Keeps the memory-manager slice this resource names alive for as long as
    /// the resource can be mapped.
    lease: Option<ManagedMemoryBinding>,
}

impl WgpuResource {
    /// Creates a device-local resource around a buffer the caller owns.
    ///
    /// Resources [`WgpuStorage`] allocates carry the pool's host-visibility and
    /// device instead; one built here can never be mapped by the host.
    pub fn new(buffer: wgpu::Buffer, address: Option<NonZeroU64>, offset: u64, size: u64) -> Self {
        Self {
            buffer,
            address,
            host_ptr: None,
            offset,
            size,
            allocation_access: AllocationAccess::new(false, None),
            lease: None,
        }
    }

    fn from_memory(memory: &WgpuMemory, offset: u64, size: u64) -> Self {
        Self {
            buffer: memory.buffer.clone(),
            address: memory.address,
            host_ptr: memory.host_ptr,
            offset,
            size,
            allocation_access: memory.allocation_access.clone(),
            lease: None,
        }
    }

    /// Keeps `lease` — the memory-manager slice this resource names — alive for
    /// as long as the resource is held, in place.
    pub(crate) fn set_lease(&mut self, lease: ManagedMemoryBinding) {
        self.lease = Some(lease);
    }

    /// The process-unique identity of the physical allocation this resource
    /// names, stable for as long as the allocation lives and never reused by a
    /// later one.
    ///
    /// Two resources that share an allocation — different slices of one pooled
    /// buffer — report the same id; a resource re-cut from a retired
    /// allocation's page reports a new one.
    pub fn allocation_id(&self) -> u64 {
        self.allocation_access.allocation_id()
    }

    /// Whether this resource's pool was created for host-visible primary memory.
    pub fn is_host_visible(&self) -> bool {
        self.allocation_access.inner.host_visible
    }

    /// Checks that the allocation can be used by the GPU.
    ///
    /// # Errors
    ///
    /// [`HostAccessError::MappedForHost`] while a host mapping of the same
    /// allocation is alive.
    pub fn ensure_gpu_access(&self) -> Result<(), HostAccessError> {
        self.allocation_access.ensure_gpu_access()
    }

    /// Reserves this allocation for GPU work that has not been submitted yet.
    ///
    /// The reservation excludes host mapping until the returned token drops,
    /// which the stream does once the work naming the resource is submitted.
    ///
    /// # Errors
    ///
    /// [`HostAccessError::MappedForHost`] while a host mapping is alive.
    pub(crate) fn acquire_gpu(&self) -> Result<GpuAccessToken, HostAccessError> {
        let mut token = self.allocation_access.acquire_gpu()?;
        token.lease = self.lease.clone();
        Ok(token)
    }

    fn acquire_host(&self) -> Result<HostAccessToken, HostAccessError> {
        if !self.is_host_visible() {
            return Err(HostAccessError::DeviceLocalAllocation);
        }
        self.allocation_access.acquire_host()
    }

    #[cfg(not(target_family = "wasm"))]
    fn map_range(&self) -> core::ops::Range<u64> {
        // A mapping is page-granular, so it covers the offset-aligned span the
        // logical range sits in rather than the logical bytes alone.
        let end = self.offset + self.size;
        self.offset..end.next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT)
    }

    /// Maps the allocation for synchronous host reads.
    ///
    /// The guard derefs to exactly the logical bytes of this resource; the
    /// mapping is released and the allocation freed for the GPU when it drops.
    ///
    /// # Errors
    ///
    /// [`HostAccessError::DeviceLocalAllocation`] when the pool is not
    /// host-visible, [`HostAccessError::OverlappingHostMapping`] when another
    /// mapping is alive, [`HostAccessError::GpuAccessInProgress`] when queued GPU
    /// work still names the allocation, and the two device errors when WGPU
    /// cannot complete the mapping.
    #[cfg(not(target_family = "wasm"))]
    pub fn map_read(&self) -> Result<WgpuMappedReadGuard, HostAccessError> {
        let token = self.acquire_host()?;
        if self.size == 0 {
            return Ok(WgpuMappedReadGuard {
                buffer: self.buffer.clone(),
                view: None,
                logical_len: 0,
                token: Some(token),
                _lease: self.lease.clone(),
            });
        }

        let range = self.map_range();
        self.wait_for_mapping(wgpu::MapMode::Read, range.clone())?;
        let view = self
            .buffer
            .get_mapped_range(range)
            .map_err(|err| HostAccessError::MapCallbackFailure(err.to_string()))?;
        Ok(WgpuMappedReadGuard {
            buffer: self.buffer.clone(),
            view: Some(view),
            logical_len: self.size as usize,
            token: Some(token),
            _lease: self.lease.clone(),
        })
    }

    /// Maps the allocation for synchronous host writes.
    ///
    /// Bytes copied through [`WgpuMappedWriteGuard::copy_from_slice`] reach the
    /// buffer when the guard drops.
    ///
    /// # Errors
    ///
    /// As [`Self::map_read`].
    #[cfg(not(target_family = "wasm"))]
    pub fn map_write(&self) -> Result<WgpuMappedWriteGuard, HostAccessError> {
        let token = self.acquire_host()?;
        if self.size == 0 {
            return Ok(WgpuMappedWriteGuard {
                buffer: self.buffer.clone(),
                view: None,
                logical_len: 0,
                token: Some(token),
                _lease: self.lease.clone(),
            });
        }

        let range = self.map_range();
        self.wait_for_mapping(wgpu::MapMode::Write, range.clone())?;
        let view = self
            .buffer
            .get_mapped_range_mut(range)
            .map_err(|err| HostAccessError::MapCallbackFailure(err.to_string()))?;
        Ok(WgpuMappedWriteGuard {
            buffer: self.buffer.clone(),
            view: Some(view),
            logical_len: self.size as usize,
            token: Some(token),
            _lease: self.lease.clone(),
        })
    }

    /// Synchronous mapping is unavailable on WebAssembly because `Device::poll`
    /// cannot block there.
    ///
    /// # Errors
    ///
    /// Always [`HostAccessError::UnsupportedTarget`].
    #[cfg(target_family = "wasm")]
    pub fn map_read(&self) -> Result<WgpuMappedReadGuard, HostAccessError> {
        Err(HostAccessError::UnsupportedTarget)
    }

    /// Synchronous mapping is unavailable on WebAssembly because `Device::poll`
    /// cannot block there.
    ///
    /// # Errors
    ///
    /// Always [`HostAccessError::UnsupportedTarget`].
    #[cfg(target_family = "wasm")]
    pub fn map_write(&self) -> Result<WgpuMappedWriteGuard, HostAccessError> {
        Err(HostAccessError::UnsupportedTarget)
    }

    /// Asks WGPU to map `range` and blocks the calling thread until it did.
    ///
    /// The mapping is only complete once the work that still reads or writes
    /// the buffer has finished, which is what makes a mapping taken after a
    /// submission observe the submitted results.
    #[cfg(not(target_family = "wasm"))]
    fn wait_for_mapping(
        &self,
        mode: wgpu::MapMode,
        range: core::ops::Range<u64>,
    ) -> Result<(), HostAccessError> {
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        self.buffer.map_async(mode, range, move |result| {
            let _ = sender.send(result);
        });

        let Some(device) = self.allocation_access.inner.device.as_ref() else {
            // Only a resource built around a foreign buffer lacks a device, and
            // such a resource is never host-visible: `acquire_host` rejects it
            // before this point.
            self.buffer.unmap();
            return Err(HostAccessError::DeviceLocalAllocation);
        };
        if let Err(err) = device.poll(wgpu::PollType::wait_indefinitely()) {
            self.buffer.unmap();
            return Err(HostAccessError::DevicePollFailure(err.to_string()));
        }

        match receiver.recv() {
            Ok(Ok(())) => Ok(()),
            Ok(Err(err)) => {
                self.buffer.unmap();
                Err(HostAccessError::MapCallbackFailure(err.to_string()))
            }
            Err(err) => {
                self.buffer.unmap();
                Err(HostAccessError::MapCallbackFailure(err.to_string()))
            }
        }
    }

    /// Return the binding view of the buffer.
    pub fn as_wgpu_bind_resource(&self) -> wgpu::BindingResource<'_> {
        // wgpu enforces 4-byte alignment for buffer binding sizes per the WebGPU spec.
        // - https://github.com/gfx-rs/wgpu/pull/8041
        //
        // This padding is safe because:
        // 1. In checked mode, bounds checks prevent reading beyond the logical size.
        // 2. In unchecked mode, OOB access is already undefined behavior.
        //
        // For zero-sized resources, pass None (use rest of buffer from offset).
        // The allocator guarantees the buffer is at least MIN_BUFFER_SIZE bytes.
        let size = NonZeroU64::new(self.size.next_multiple_of(4));

        let binding = wgpu::BufferBinding {
            buffer: &self.buffer,
            offset: self.offset,
            size,
        };
        wgpu::BindingResource::Buffer(binding)
    }

    /// Returns the binding view after checking that no host mapping is active.
    ///
    /// # Errors
    ///
    /// [`HostAccessError::MappedForHost`] while a host mapping is alive.
    pub fn try_as_wgpu_bind_resource(&self) -> Result<wgpu::BindingResource<'_>, HostAccessError> {
        self.ensure_gpu_access()?;
        Ok(self.as_wgpu_bind_resource())
    }
}

/// Owning RAII guard for a host-readable mapping of a WGPU allocation.
#[derive(Debug)]
pub struct WgpuMappedReadGuard {
    buffer: wgpu::Buffer,
    view: Option<wgpu::BufferView>,
    logical_len: usize,
    token: Option<HostAccessToken>,
    _lease: Option<ManagedMemoryBinding>,
}

impl Deref for WgpuMappedReadGuard {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match &self.view {
            Some(view) => &view[..self.logical_len],
            None => &[],
        }
    }
}

impl Drop for WgpuMappedReadGuard {
    fn drop(&mut self) {
        drop(self.view.take());
        if self.logical_len != 0 {
            self.buffer.unmap();
        }
        drop(self.token.take());
    }
}

/// Owning RAII guard for a host-writable mapping of a WGPU allocation.
///
/// WGPU write mappings may be write-combining memory, so wgpu exposes them as
/// [`wgpu::WriteOnly`] rather than `&mut [u8]`. Use [`Self::as_write_only`] for
/// zero-copy writes.
#[derive(Debug)]
pub struct WgpuMappedWriteGuard {
    buffer: wgpu::Buffer,
    view: Option<wgpu::BufferViewMut>,
    logical_len: usize,
    token: Option<HostAccessToken>,
    _lease: Option<ManagedMemoryBinding>,
}

impl WgpuMappedWriteGuard {
    /// Returns a zero-copy write-only view of the exact logical resource range.
    pub fn as_write_only(&mut self) -> Option<wgpu::WriteOnly<'_, [u8]>> {
        self.view
            .as_mut()
            .map(|view| view.slice(..self.logical_len))
    }

    /// Copies `bytes` into the mapped logical resource range.
    ///
    /// # Panics
    ///
    /// When `bytes` does not fill the resource exactly.
    pub fn copy_from_slice(&mut self, bytes: &[u8]) {
        assert_eq!(bytes.len(), self.logical_len);
        if let Some(mut view) = self.as_write_only() {
            view.copy_from_slice(bytes);
        }
    }

    /// Returns the logical resource length in bytes.
    pub fn len(&self) -> usize {
        self.logical_len
    }

    /// Returns whether the logical resource is empty.
    pub fn is_empty(&self) -> bool {
        self.logical_len == 0
    }
}

impl Drop for WgpuMappedWriteGuard {
    fn drop(&mut self) {
        drop(self.view.take());
        if self.logical_len != 0 {
            self.buffer.unmap();
        }
        drop(self.token.take());
    }
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
pub struct WgpuStorage {
    memory: HashMap<StorageId, WgpuMemory>,
    /// The bytes `memory` holds.
    allocated: u64,
    device: wgpu::Device,
    buffer_usages: BufferUsages,
    mem_alignment: usize,
    #[allow(unused, reason = "keep it simple")]
    vk_storage: bool,
    /// Whether every buffer this storage allocates is mappable by the host.
    host_visible: bool,
}

impl core::fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("WgpuStorage {{ device: {:?} }}", self.device).as_str())
    }
}

impl WgpuStorage {
    /// Create a new device-local storage on the given [device](wgpu::Device).
    pub fn new(
        mem_alignment: usize,
        device: wgpu::Device,
        usages: BufferUsages,
        vk_storage: bool,
    ) -> Self {
        Self {
            memory: HashMap::new(),
            allocated: 0,
            device,
            buffer_usages: usages,
            mem_alignment,
            vk_storage,
            host_visible: false,
        }
    }

    /// Create a storage whose buffers the host can map, for the main pool of a
    /// runtime brought up on host-visible primary memory.
    ///
    /// Every allocation adds `MAP_READ | MAP_WRITE` to `usages`, which the
    /// device must have enabled `Features::MAPPABLE_PRIMARY_BUFFERS` for. The
    /// Vulkan unified-memory path is deliberately not taken here: it maps the
    /// buffer persistently, which a second `map_async` of the same buffer
    /// cannot share.
    pub fn new_with_host_visibility(
        mem_alignment: usize,
        device: wgpu::Device,
        usages: BufferUsages,
    ) -> Self {
        Self {
            memory: HashMap::new(),
            allocated: 0,
            device,
            buffer_usages: usages,
            mem_alignment,
            vk_storage: false,
            host_visible: true,
        }
    }

    /// The usages an allocation is created with, which host-visible memory
    /// extends with the mapping the mode promises.
    fn allocation_usages(&self) -> BufferUsages {
        if self.host_visible {
            self.buffer_usages | BufferUsages::MAP_READ | BufferUsages::MAP_WRITE
        } else {
            self.buffer_usages
        }
    }
}

impl ComputeStorage for WgpuStorage {
    type Resource = WgpuResource;

    fn alignment(&self) -> usize {
        self.mem_alignment
    }

    fn get(&mut self, handle: &StorageHandle) -> Result<Self::Resource, IoError> {
        let memory = self
            .memory
            .get(&handle.id)
            .ok_or_else(|| IoError::StorageHandleNotFound {
                reason: format!("{} in the wgpu buffer storage", handle.id).into(),
                backtrace: BackTrace::capture(),
            })?;
        Ok(WgpuResource::from_memory(
            memory,
            handle.offset(),
            handle.size(),
        ))
    }

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(level = "trace", skip(self, size))
    )]
    fn alloc(&mut self, size: u64) -> Result<StorageHandle, IoError> {
        let id = StorageId::new();

        let alloc_size = size.max(MIN_BUFFER_SIZE);

        let memory = self.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: alloc_size,
            usage: self.allocation_usages(),
            mapped_at_creation: false,
        })?;

        self.memory.insert(id, memory);
        self.allocated += alloc_size;
        Ok(StorageHandle::new(
            id,
            StorageUtilization { offset: 0, size },
        ))
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    fn dealloc(&mut self, id: StorageId) {
        if let Some(memory) = self.memory.remove(&id) {
            self.allocated -= memory.buffer.size();
        }
    }

    fn flush(&mut self) {
        // We don't wait for dealloc
    }

    fn bytes_allocated(&self) -> u64 {
        self.allocated
    }
}

impl WgpuStorage {
    #[cfg(feature = "spirv")]
    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> Result<WgpuMemory, IoError> {
        if self.vk_storage {
            // wgpu currently doesn't expose this, even though it's used internally for acceleration
            // structures. While we could use the acceleration structure input flag, that would
            // then require ray tracing to be supported. So we need to allocate manually, then import
            // the native buffer into wgpu using `from_raw_managed`.
            // This actually skips some of the buffer batching stuff we don't really want in `gpu_allocator`.
            crate::backend::vulkan::create_storage_buffer(&self.device, desc)
        } else {
            Ok(self.allocated_memory(desc))
        }
    }

    #[cfg(not(feature = "spirv"))]
    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> Result<WgpuMemory, IoError> {
        Ok(self.allocated_memory(desc))
    }

    /// Creates a buffer with the access state its `host_visible` flag calls for.
    fn allocated_memory(&self, desc: &wgpu::BufferDescriptor<'_>) -> WgpuMemory {
        let buffer = self.device.create_buffer(desc);
        match self.host_visible {
            true => WgpuMemory {
                buffer,
                address: None,
                host_ptr: None,
                allocation_access: AllocationAccess::new(true, Some(self.device.clone())),
            },
            false => WgpuMemory::device_local(buffer, None, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloned_allocation_access_retains_nonzero_identity() {
        let access = AllocationAccess::new(false, None);
        let cloned = access.clone();

        assert_ne!(access.allocation_id(), 0);
        assert_eq!(access.allocation_id(), cloned.allocation_id());
    }

    #[test]
    fn host_access_excludes_other_host_and_gpu_access_until_drop() {
        let access = AllocationAccess::new(true, None);
        let token = access.acquire_host().unwrap();

        assert_eq!(
            access.acquire_host().unwrap_err(),
            HostAccessError::OverlappingHostMapping
        );
        assert_eq!(
            access.ensure_gpu_access(),
            Err(HostAccessError::MappedForHost)
        );

        drop(token);

        assert_eq!(access.ensure_gpu_access(), Ok(()));
        assert!(access.acquire_host().is_ok());
    }

    #[test]
    fn gpu_reservation_excludes_host_access_until_submission() {
        let access = AllocationAccess::new(true, None);
        let worker_access = access.clone();
        let (reserved_tx, reserved_rx) = std::sync::mpsc::sync_channel(1);
        let (submitted_tx, submitted_rx) = std::sync::mpsc::sync_channel(1);

        let worker = std::thread::spawn(move || {
            let reservation = worker_access.acquire_gpu().unwrap();
            reserved_tx.send(()).unwrap();
            submitted_rx.recv().unwrap();
            drop(reservation);
        });

        reserved_rx.recv().unwrap();
        assert_eq!(
            access.acquire_host().unwrap_err(),
            HostAccessError::GpuAccessInProgress
        );

        submitted_tx.send(()).unwrap();
        worker.join().unwrap();
        assert!(access.acquire_host().is_ok());
    }

    #[test]
    fn device_local_allocation_never_reports_host_visibility() {
        let access = AllocationAccess::new(false, None);

        assert!(!access.inner.host_visible);
    }
}
