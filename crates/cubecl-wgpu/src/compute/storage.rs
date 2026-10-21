use cubecl_core::server::IoError;
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::collections::HashMap;
use cubecl_server::storage::{ComputeStorage, StorageHandle, StorageId, StorageUtilization};
use std::{num::NonZeroU64, ops::Range, ptr::NonNull};
use wgpu::BufferUsages;

/// Minimum buffer size in bytes. The WebGPU spec requires buffer sizes > 0, and shaders
/// declare typed arrays (e.g. `array<vec4<f32>>`) that impose a minimum binding size.
/// 32 bytes covers the largest possible binding type (`vec4<f64>`).
const MIN_BUFFER_SIZE: u64 = 32;

/// The span a host mapping has to cover for the logical slice `offset..offset + size`,
/// and where the slice starts in it. WGPU requires the start to be
/// `MAP_ALIGNMENT`-aligned and the length to be a multiple of
/// `COPY_BUFFER_ALIGNMENT`; mapping the whole physical buffer instead would charge a
/// shared page's full size to a single small read.
fn host_mapping_range(offset: u64, size: u64) -> (Range<u64>, usize) {
    let start = offset & !(wgpu::MAP_ALIGNMENT - 1);
    let end = (offset + size).next_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT);
    (start..end, (offset - start) as usize)
}

/// Buffer storage for wgpu.
pub struct WgpuStorage {
    memory: HashMap<StorageId, WgpuMemory>,
    /// The bytes `memory` holds.
    allocated: u64,
    device: wgpu::Device,
    buffer_usages: BufferUsages,
    mem_alignment: usize,
    #[allow(unused, reason = "keep it simple")]
    vk_storage: bool,
    /// Whether every buffer this storage allocates carries `MAP_READ | MAP_WRITE`.
    /// Only a primary pool on host-visible memory does.
    host_visible: bool,
}

impl core::fmt::Debug for WgpuStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(format!("WgpuStorage {{ device: {:?} }}", self.device).as_str())
    }
}

/// A buffer's persistent host mapping. Crate private: writes through it need an idle queue.
#[derive(Debug, Clone, Copy)]
pub(crate) struct HostPtr(pub(crate) NonNull<u8>);

// SAFETY: Only the stream owning the buffer writes through it.
unsafe impl Send for HostPtr {}
unsafe impl Sync for HostPtr {}

/// The memory resource that can be allocated for wgpu.
#[derive(new, Debug, Clone)]
pub struct WgpuResource {
    /// The wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// The buffer device address, if supported
    pub address: Option<NonZeroU64>,
    /// The buffer host mapping, if host visible.
    #[new(default)]
    pub(crate) host_ptr: Option<HostPtr>,
    /// Whether the buffer came from a host-visible primary pool.
    #[new(default)]
    pub(crate) host_visible: bool,
    /// The buffer offset.
    pub offset: u64,
    /// The size of the resource.
    ///
    /// # Notes
    ///
    /// The result considers the offset.
    pub size: u64,
}

/// The memory that can be allocated for wgpu.
#[derive(new, Debug)]
pub struct WgpuMemory {
    /// The wgpu buffer.
    pub buffer: wgpu::Buffer,
    /// The buffer device address, if supported
    pub address: Option<NonZeroU64>,
    /// The buffer host mapping, if host visible.
    #[new(default)]
    pub(crate) host_ptr: Option<HostPtr>,
    /// Whether the host can map this buffer.
    #[new(default)]
    pub(crate) host_visible: bool,
}

/// How a host write can reach an allocation.
#[derive(Clone, Copy, Debug)]
pub(crate) enum WriteTarget {
    /// There is no mapping to take: only the queue can carry the bytes.
    QueueOnly,
    /// A host-visible pool allocation, which the write maps on demand.
    Mapped,
    /// A device-local allocation with a persistent host pointer (the native
    /// Vulkan allocator), which the write copies through.
    Pointer(HostPtr),
}

impl WriteTarget {
    /// Whether the bytes can be copied in place now.
    ///
    /// Both in-place paths write outside the queue, so both need a device with
    /// nothing in flight: a write that arrives while it is busy takes the queue's
    /// staged copy, which the queue orders.
    pub(crate) fn in_place_now(self, device_idle: bool) -> bool {
        match self {
            Self::QueueOnly => false,
            Self::Mapped | Self::Pointer(_) => device_idle,
        }
    }
}

impl WgpuResource {
    /// How a host write can reach this allocation.
    pub(crate) fn write_target(&self) -> WriteTarget {
        match (self.host_visible, self.host_ptr) {
            (true, _) => WriteTarget::Mapped,
            (false, Some(host_ptr)) => WriteTarget::Pointer(host_ptr),
            (false, None) => WriteTarget::QueueOnly,
        }
    }

    /// Whether this resource came from a host-visible primary pool, so the host can
    /// map it in place instead of staging a copy through the queue.
    pub(crate) fn is_host_visible(&self) -> bool {
        self.host_visible
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

    /// The mapping span covering this logical slice, and the slice's start in it.
    pub(crate) fn map_range(&self) -> (Range<u64>, usize) {
        host_mapping_range(self.offset, self.size)
    }

    /// Copies this resource out of its own host mapping.
    ///
    /// Only valid while the device is idle, for the same reason as
    /// [`write_host_visible`](Self::write_host_visible): the mapping is not
    /// ordered by the queue. Mapping and unmapping inside the call is also what
    /// keeps reads from colliding — wgpu refuses a second mapping *and* a
    /// submission that uses a mapped buffer, so a mapping that outlived the call
    /// would block every other user of the buffer, neighbours on its page included.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn read_host_visible(
        &self,
        device: &wgpu::Device,
        size: usize,
    ) -> Result<Vec<u8>, String> {
        // A shape can ask for less than the resource holds; it can never ask for
        // more than the span the mapping covers.
        let size = size.min(self.size as usize);
        if size == 0 {
            return Ok(Vec::new());
        }
        let (range, view_offset) = self.map_range();
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        self.buffer
            .slice(range.clone())
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });

        let polled = device.poll(wgpu::PollType::Poll);
        // The callback is what says whether the mapping landed: only one that did
        // may be released, and `unmap` on a buffer that is not mapped is a
        // validation error rather than a no-op.
        mapped(receiver.try_recv(), &self.buffer)?;

        let result = match self.buffer.get_mapped_range(range.clone()) {
            Ok(view) => Ok(view[view_offset..view_offset + size].to_vec()),
            Err(err) => Err(format!("the mapped range could not be read: {err}")),
        };
        self.buffer.unmap();
        polled
            .map(|_| ())
            .map_err(|err| format!("polling the device for the mapping failed: {err}"))?;
        result
    }

    /// Writes `data` into this allocation through a host mapping.
    ///
    /// The caller has established that the device has nothing in flight — a mapped
    /// write is not ordered by the queue, so idleness is what makes in-place
    /// writing safe — and that the write fits the resource, which the server
    /// checks when it schedules it. The single non-blocking poll below is all the
    /// mapping needs, since it has no submission to wait for. A mapping that is
    /// still pending after it is aborted and reported, so the caller can leave the
    /// write to the queue instead of stalling the device thread.
    #[cfg(not(target_family = "wasm"))]
    pub(crate) fn write_host_visible(
        &self,
        data: &[u8],
        device: &wgpu::Device,
    ) -> Result<(), String> {
        if data.is_empty() || self.size == 0 {
            return Ok(());
        }
        let (range, view_offset) = self.map_range();
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        self.buffer
            .slice(range.clone())
            .map_async(wgpu::MapMode::Write, move |result| {
                let _ = sender.send(result);
            });

        let polled = device.poll(wgpu::PollType::Poll);
        // The callback is what says whether the mapping landed: only one that did
        // may be released, and `unmap` on a buffer that is not mapped is a
        // validation error rather than a no-op.
        mapped(receiver.try_recv(), &self.buffer)?;

        let result = match self.buffer.get_mapped_range_mut(range.clone()) {
            Ok(mut view) => {
                view.slice(view_offset..view_offset + data.len())
                    .copy_from_slice(data);
                Ok(())
            }
            Err(err) => Err(format!("the mapped range could not be written: {err}")),
        };
        self.buffer.unmap();
        polled
            .map(|_| ())
            .map_err(|err| format!("polling the device for the mapping failed: {err}"))?;
        result
    }
}

/// Reads a mapping's callback outcome: `Ok` when the mapping landed and is ours
/// to release, `Err` with what to report otherwise.
///
/// A rejected mapping is left alone — the buffer may be mapped by someone else,
/// and unmapping it would be a validation error. A mapping still pending after
/// one non-blocking poll is ours and is cancelled, so that a host access which
/// finds the device busy does not leave the buffer unusable.
#[cfg(not(target_family = "wasm"))]
fn mapped(
    callback: Result<Result<(), wgpu::BufferAsyncError>, std::sync::mpsc::TryRecvError>,
    buffer: &wgpu::Buffer,
) -> Result<(), String> {
    match callback {
        Ok(Ok(())) => Ok(()),
        Ok(Err(err)) => Err(format!("the host mapping failed: {err}")),
        Err(_) => {
            buffer.unmap();
            Err("the device was not idle, so the mapping did not complete".into())
        }
    }
}

/// Keeps actual wgpu buffer references in a hashmap with ids as key.
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

    /// Create a storage whose buffers the host can map, for the primary pool of a
    /// runtime brought up on host-visible memory.
    ///
    /// Every allocation adds `MAP_READ | MAP_WRITE` to `usages`; the device must
    /// have enabled [`wgpu::Features::MAPPABLE_PRIMARY_BUFFERS`] for it.
    pub(crate) fn new_with_host_visibility(
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
            // The native Vulkan allocator maps persistently outside wgpu; a
            // host-visible pool maps through wgpu instead.
            vk_storage: false,
            host_visible: true,
        }
    }

    fn allocation_usages(&self) -> BufferUsages {
        if self.host_visible {
            self.buffer_usages | BufferUsages::MAP_READ | BufferUsages::MAP_WRITE
        } else {
            self.buffer_usages
        }
    }

    /// Creates a buffer, remembering whether the host can map it.
    fn create_memory(&self, desc: &wgpu::BufferDescriptor<'_>) -> WgpuMemory {
        WgpuMemory {
            buffer: self.device.create_buffer(desc),
            address: None,
            host_ptr: None,
            host_visible: self.host_visible,
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
        Ok(WgpuResource {
            host_ptr: memory.host_ptr,
            host_visible: memory.host_visible,
            ..WgpuResource::new(
                memory.buffer.clone(),
                memory.address,
                handle.offset(),
                handle.size(),
            )
        })
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
            Ok(self.create_memory(desc))
        }
    }

    #[cfg(not(feature = "spirv"))]
    fn create_buffer(&self, desc: &wgpu::BufferDescriptor<'_>) -> Result<WgpuMemory, IoError> {
        Ok(self.create_memory(desc))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A host-visible resource on the CPU adapter, or `None` where the machine
    /// cannot bring one up. The device is returned with it, since every host
    /// access needs one to poll.
    #[cfg(not(target_family = "wasm"))]
    fn host_visible_buffer(size: u64) -> Option<(wgpu::Device, WgpuResource)> {
        let selector = crate::WgpuDevice::new(crate::WgpuDeviceKind::Cpu);
        let setup = cubecl_environment::future::block_on(crate::runtime::try_create_setup(
            &selector,
            crate::PrimaryMemoryMode::HostVisible,
        ))
        .ok()?;
        let mut storage = WgpuStorage::new_with_host_visibility(
            32,
            setup.device.clone(),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        let handle = storage.alloc(size).ok()?;
        let resource = storage.get(&handle).ok()?;
        Some((setup.device, resource))
    }

    /// The whole point of a host-visible pool: the bytes go in and come out of
    /// the allocation itself, and a partial write leaves the rest alone.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn a_host_visible_resource_writes_and_reads_in_place() {
        let Some((device, resource)) = host_visible_buffer(16) else {
            eprintln!("skipping the host-visible round trip: no CPU adapter");
            return;
        };

        resource
            .write_host_visible(&[7; 16], &device)
            .expect("the write maps in place");
        assert_eq!(
            resource
                .read_host_visible(&device, 16)
                .expect("the read maps"),
            [7; 16].to_vec()
        );

        resource
            .write_host_visible(&[9; 4], &device)
            .expect("the partial write maps in place");
        assert_eq!(
            resource
                .read_host_visible(&device, 16)
                .expect("the read maps"),
            [9, 9, 9, 9, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7].to_vec(),
            "a partial write into the mapping leaves the rest of the slice alone"
        );
    }

    /// A read copies out exactly the logical slice: the rest of the physical
    /// buffer is another allocation's, and must not be handed to the caller.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn a_host_visible_read_stops_at_the_logical_size() {
        let Some((device, resource)) = host_visible_buffer(16) else {
            eprintln!("skipping the host-visible read bound: no CPU adapter");
            return;
        };

        resource
            .write_host_visible(&[1; 16], &device)
            .expect("the write maps in place");

        assert_eq!(resource.read_host_visible(&device, 16).unwrap().len(), 16);
        // A shape may ask for less than the resource holds.
        assert_eq!(
            resource.read_host_visible(&device, 4).unwrap(),
            [1; 4].to_vec()
        );
    }

    /// A mapping that cannot be taken reports it and leaves the buffer usable:
    /// the caller stages the write instead, and a later mapping is not refused
    /// because this one left the buffer pending.
    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn a_mapping_that_cannot_be_taken_leaves_the_buffer_usable() {
        let Some((device, resource)) = host_visible_buffer(16) else {
            eprintln!("skipping the busy host-visible mapping: no CPU adapter");
            return;
        };

        // Hold a read mapping so the write's cannot be taken.
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        resource
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        assert!(
            resource.write_host_visible(&[7; 16], &device).is_err(),
            "a second mapping of one buffer cannot be taken"
        );
        // Release the one we hold, whichever way it went.
        let _ = receiver.recv();
        resource.buffer.unmap();

        resource
            .write_host_visible(&[7; 16], &device)
            .expect("the buffer is usable again");
        assert_eq!(
            resource.read_host_visible(&device, 16).unwrap(),
            [7; 16].to_vec()
        );
    }

    #[test]
    fn a_write_copies_in_place_only_on_an_idle_device() {
        let pointer = WriteTarget::Pointer(HostPtr(NonNull::dangling()));

        assert!(
            !WriteTarget::QueueOnly.in_place_now(true),
            "there is no mapping to take, however idle the device is"
        );
        for target in [WriteTarget::Mapped, pointer] {
            assert!(
                !target.in_place_now(false),
                "{target:?} writes outside the queue, so a busy device stages it"
            );
            assert!(target.in_place_now(true));
        }
    }

    #[test]
    fn host_mapping_range_covers_the_logical_slice_in_aligned_bytes() {
        let (range, view_offset) = host_mapping_range(4, 12);

        assert_eq!(view_offset, 4);
        assert_eq!(range, 0..16);
        assert!(range.start.is_multiple_of(wgpu::MAP_ALIGNMENT));
        assert!((range.end - range.start).is_multiple_of(wgpu::COPY_BUFFER_ALIGNMENT));
        assert!(range.start + view_offset as u64 <= 4);
        assert!(4 + 12 <= range.end);
    }
}
