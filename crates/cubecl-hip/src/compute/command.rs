//! One unit of work against the device.
//!
//! Every operation the server exposes that touches memory or launches a
//! kernel goes through a [`Command`]: it pairs the context holding the
//! compiled kernels with the streams the operation was resolved against, and
//! resolving is what orders the current stream behind whichever streams own
//! the buffers it was handed.
//!
//! The copies at the bottom are where the driver's pitched layouts are dealt
//! with. Both directions ask [`Pitch`] the same question — is there padding
//! between the rows — because a linear copy of a pitched buffer scrambles it,
//! and a 2D copy of a contiguous one is slower and refused outright for very
//! tall transfers.

use crate::compute::status::checked;
use crate::compute::{
    MB, context::HipContext, fence::Fence, gpu::GpuResource,
    io::controller::PinnedMemoryManagedAllocController, stream::HipStreamBackend,
};
use cubecl_common::bytes::Bytes;
use cubecl_core::{
    MemoryConfiguration, MemoryUsage,
    bytes::AllocationProperty,
    ir::MemoryDeviceProperties,
    server::{BufferBinding, CopyDescriptor, Handle, IoError, LaunchError, ServerError},
    zspace::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use cubecl_environment::backtrace::BackTrace;
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_hip_sys::{
    hipMemcpyKind_hipMemcpyDeviceToHost, hipMemcpyKind_hipMemcpyHostToDevice, ihipStream_t,
};
use cubecl_runtime::{
    id::KernelId,
    memory_management::{
        InstallMemoryPoolsError, ManagedMemoryHandle, MemoryAllocationMode, MemoryHandle,
        MemoryReport,
    },
    stream::ResolvedStreams,
};
use std::ffi::c_void;

/// One unit of work against the device: the context that holds its compiled
/// kernels, and the streams it was resolved against.
///
/// Built per operation rather than held, because resolving is what orders the
/// current stream behind whichever streams own the buffers it was given.
#[derive(new)]
pub struct Command<'a> {
    ctx: &'a mut HipContext,
    pub(crate) streams: ResolvedStreams<'a, HipStreamBackend>,
}

impl<'a> Command<'a> {
    /// The device allocation `binding` names, resolved on the stream that
    /// created it rather than the current one.
    ///
    /// # Errors
    ///
    /// [`IoError::InvalidHandle`] when the binding names no live allocation.
    pub fn resource(&mut self, binding: BufferBinding) -> Result<GpuResource, IoError> {
        self.streams
            .get(&binding.stream)
            .memory_management_gpu
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
    }

    /// The current stream's GPU memory usage.
    pub fn memory_usage(&mut self) -> MemoryUsage {
        self.streams.current().memory_management_gpu.memory_usage()
    }

    /// Structured per-pool report of the current stream's main GPU memory.
    pub fn memory_report(&mut self) -> MemoryReport {
        self.streams.current().memory_management_gpu.memory_report()
    }

    /// Explicitly cleanup gpu memory on the current stream.
    pub fn memory_cleanup(&mut self) {
        let stream = self.streams.current();
        // Deferred frees sit in the drop queue until a fenced flush, so an
        // explicit cleanup must drain it first or the pools still see those
        // slices as live. The queue is a double buffer (one flush only rotates
        // the current batch), so flush twice. Skipped mid-capture: a host sync
        // aborts the capture, and the capture path drains the queue itself.
        // The cleanups below stay safe mid-capture: `cleanup` defers all frees
        // while a capture is active.
        if !stream.capturing.is_recording() {
            let sys = stream.sys;
            stream.drop_queue.flush(|| Fence::new(sys));
            stream.drop_queue.flush(|| Fence::new(sys));
            // The info cache's buffers are live slices in the dynamic pools;
            // an explicit cleanup exists to leave those pools empty (e.g. for
            // a rebuild sized to the next workload), so every entry not
            // pinned by a live graph goes too. Skipped while recording for the
            // same reason the flush is: an entry the recording has not touched
            // yet would come back as a fresh allocation inside the capture
            // window, which is illegal.
            stream.info_cache.clear_unpinned();
        }
        let (stream, failures) = self.streams.current_and_failures();
        stream.memory_management_gpu.cleanup(true, failures);
        stream.memory_management_cpu.cleanup(true, failures);
    }

    /// Set the [`MemoryAllocationMode`] for the current stream.
    pub fn allocation_mode(&mut self, mode: MemoryAllocationMode) {
        self.streams.current().memory_management_gpu.mode(mode)
    }

    /// Rebuild the current stream's main-GPU pools with a new layout, keeping
    /// the old one when something is still live in them.
    ///
    /// # Errors
    ///
    /// [`InstallMemoryPoolsError::PoolsInUse`] when the rebuild was refused.
    pub fn install_memory_pools(
        &mut self,
        config: MemoryConfiguration,
        props: &MemoryDeviceProperties,
    ) -> Result<(), InstallMemoryPoolsError> {
        let (stream, failures) = self.streams.current_and_failures();
        stream
            .memory_management_gpu
            .install_pools(config, props, failures)
    }

    /// Allocate `size` bytes of GPU memory on the current stream.
    ///
    /// # Errors
    ///
    /// [`IoError::BufferTooBig`] when no device could ever fit it, and
    /// whatever the allocator reports when a reclaim-and-retry still cannot.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        let (stream, failures) = self.streams.current_and_failures();
        match stream.memory_management_gpu.reserve(size, failures) {
            Ok(handle) => Ok(handle),
            // The allocation can never fit; reclaiming would not change that.
            Err(err @ IoError::BufferTooBig { .. }) => Err(err),
            // Out of memory *right now* is not out of memory for good: pool
            // pages whose slices have all been dropped are still resident, and
            // the frees that would release them may sit in the deferred drop
            // queue. Reclaim this stream's memory and retry once; only a
            // failure after that is reported. Without the retry, a transient
            // peak — a model build holding float weights while their quantized
            // copies allocate, an autotune sample on a full device — becomes a
            // never-initialized handle whose every downstream use fails.
            Err(err) => {
                log::warn!("device allocation of {size} B failed ({err}); reclaiming and retrying");
                self.memory_cleanup();
                let (stream, failures) = self.streams.current_and_failures();
                stream.memory_management_gpu.reserve(size, failures)
            }
        }
    }

    /// Get the stream cursor.
    pub fn cursor(&self) -> u64 {
        self.streams.cursor
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn empty(&mut self, size: u64) -> Result<Handle, IoError> {
        let handle = Handle::new(self.streams.current, size);
        let reserved = self.reserve(size)?;
        self.bind(reserved, handle.memory.clone())?;

        Ok(handle)
    }

    /// Give `reserved`'s storage to `new`, so handles issued against `new`
    /// resolve to it.
    ///
    /// # Errors
    ///
    /// [`IoError`] when the reservation has no initialized storage to give.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn bind(
        &mut self,
        reserved: ManagedMemoryHandle,
        new: ManagedMemoryHandle,
    ) -> Result<(), IoError> {
        let cursor = self.cursor();
        let (stream, failures) = self.streams.current_and_failures();
        stream
            .memory_management_gpu
            .bind(reserved, new, cursor, failures)
    }

    /// `size` bytes of host memory, pinned when the pool can serve it.
    ///
    /// Pinned pages transfer by DMA without a bounce, but they are scarce, so
    /// an exhausted pool falls back to the heap rather than failing: this
    /// always answers with a buffer of the size asked for.
    pub fn reserve_cpu(&mut self, size: usize, origin: Option<StreamId>) -> Bytes {
        self.reserve_pinned(size, origin)
            .unwrap_or_else(|| Bytes::from_bytes_vec(vec![0; size]))
    }

    fn reserve_pinned(&mut self, size: usize, origin: Option<StreamId>) -> Option<Bytes> {
        let (stream, failures) = match origin {
            Some(id) => self.streams.get_and_failures(&id),
            None => self.streams.current_and_failures(),
        };
        let handle = stream
            .memory_management_cpu
            .reserve(size as u64, failures)
            .ok()?;

        let binding = MemoryHandle::binding(handle);
        let resource = stream
            .memory_management_cpu
            .get_resource(binding.clone(), None, None)
            .ok()?;

        let controller = Box::new(PinnedMemoryManagedAllocController::init(binding, resource));
        // SAFETY: The binding has initialized memory for at least `size` bytes.
        Some(unsafe { Bytes::from_controller(controller, size) })
    }

    /// Copy each descriptor's device memory back to the host, resolving once
    /// the copies have landed.
    ///
    /// The copies are enqueued before the future is returned; awaiting it
    /// waits on the fence that follows them.
    ///
    /// # Errors
    ///
    /// [`IoError::UnsupportedStrides`] for a layout the driver cannot copy,
    /// and whatever the fence reports when the stream itself failed.
    pub fn read_async(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
    ) -> impl Future<Output = Result<Vec<Bytes>, ServerError>> + Send + use<> {
        let descriptors_moved = descriptors
            .iter()
            .map(|b| b.handle.clone())
            .collect::<Vec<_>>();
        let result = self.copies_to_bytes(descriptors);
        let fence = Fence::new(self.streams.current().sys);

        async move {
            let sync = fence.wait_sync();
            // Release memory handle.
            core::mem::drop(descriptors_moved);

            sync?;
            let bytes = result?;

            Ok(bytes)
        }
    }

    fn copies_to_bytes(&mut self, descriptors: Vec<CopyDescriptor>) -> Result<Vec<Bytes>, IoError> {
        let mut result = Vec::with_capacity(descriptors.len());

        for descriptor in descriptors {
            result.push(self.copy_to_bytes(descriptor, None)?);
        }

        Ok(result)
    }

    fn copy_to_bytes(
        &mut self,
        descriptor: CopyDescriptor,
        stream_id: Option<StreamId>,
    ) -> Result<Bytes, IoError> {
        let num_bytes = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        let mut bytes = self.reserve_cpu(num_bytes, stream_id);
        self.write_to_cpu(descriptor, &mut bytes, stream_id)?;

        Ok(bytes)
    }

    /// Enqueue a copy of `descriptor`'s device memory into `bytes`.
    ///
    /// # Errors
    ///
    /// [`IoError::UnsupportedStrides`] for a layout that is not pitched
    /// row-major, [`IoError::InvalidHandle`] for a binding that names no live
    /// allocation, and the driver's refusal to copy.
    pub fn write_to_cpu(
        &mut self,
        descriptor: CopyDescriptor,
        bytes: &mut Bytes,
        stream_id: Option<StreamId>,
    ) -> Result<(), IoError> {
        let CopyDescriptor {
            handle: binding,
            shape,
            strides,
            elem_size,
        } = descriptor;

        if !has_pitched_row_major_strides(&shape, &strides) {
            return Err(IoError::UnsupportedStrides {
                backtrace: BackTrace::capture(),
            });
        }

        let resource = self.resource(binding)?;
        let stream = match stream_id {
            Some(id) => self.streams.get(&id),
            None => self.streams.current(),
        };

        // SAFETY: `resource.ptr` is a valid device pointer obtained from the memory manager,
        // `stream.sys` is an initialized HIP stream, and `bytes` is pre-allocated with
        // sufficient capacity for the copy.
        unsafe { write_to_cpu(&shape, &strides, elem_size, bytes, resource.ptr, stream.sys) }
    }

    /// Enqueue a copy of `data` into the device memory `descriptor` names.
    ///
    /// # Errors
    ///
    /// [`IoError::UnsupportedStrides`] for a layout that is not pitched
    /// row-major, [`IoError::InvalidHandle`] for a binding that names no live
    /// allocation, and the driver's refusal to copy.
    pub fn write_to_gpu(&mut self, descriptor: CopyDescriptor, data: Bytes) -> Result<(), IoError> {
        let CopyDescriptor {
            handle: binding,
            shape,
            strides,
            elem_size,
        } = descriptor;
        if !has_pitched_row_major_strides(&shape, &strides) {
            return Err(IoError::UnsupportedStrides {
                backtrace: BackTrace::capture(),
            });
        }

        let resource = self.resource(binding)?;
        let size = data.len();

        // An empty tensor (a zero dim in its shape) has nothing to copy. Bail
        // before staging: the zero-size staging buffer has no real backing (a
        // dangling pointer), and the 2D copy below would still transfer
        // `width_bytes` from it when only the leading dims are zero.
        if size == 0 {
            return Ok(());
        }

        let property = data.property();

        // Transfers up to this size go through a pinned staging buffer (faster DMA).
        const STAGE_MAX: usize = 100 * MB;
        // Above this size we flush the drop queue so the source buffer is released promptly.
        const FLUSH_MIN: usize = 10 * MB;

        // Stage file-backed data, and small host data that isn't already pinned. Re-staging
        // already-pinned memory would be a redundant pinned-to-pinned copy.
        let should_stage = matches!(property, AllocationProperty::File)
            || (size < STAGE_MAX && !matches!(property, AllocationProperty::Pinned));
        let should_flush = size > FLUSH_MIN || matches!(property, AllocationProperty::File);

        let data = match should_stage {
            true => {
                // Pinned staging is a DMA optimization, not a requirement, so
                // an exhausted pinned pool falls back to a plain heap buffer
                // rather than failing the write — the same answer `reserve_cpu`
                // gives for the same condition. File-backed data still lands in
                // real memory before the driver reads it asynchronously, which
                // is the half of the staging that is mandatory.
                let mut buffer = self
                    .reserve_pinned(size, None)
                    .unwrap_or_else(|| Bytes::from_bytes_vec(vec![0; size]));
                data.copy_into(&mut buffer);
                buffer
            }
            false => data,
        };

        let current = self.streams.current();

        // SAFETY: `resource` is a valid GPU allocation, `data` is a valid host buffer,
        // and `current.sys` is an initialized HIP stream. The shape/strides have been
        // validated above to be pitched row-major.
        unsafe {
            write_to_gpu(resource, &shape, &strides, elem_size, &data, current.sys)?;
        };

        current.drop_queue.push(data);

        // Defer fenced flushes while capturing — a host sync aborts the capture.
        if (should_flush || current.drop_queue.should_flush()) && !current.capturing.is_recording()
        {
            current.drop_queue.flush(|| Fence::new(current.sys));
        }

        Ok(())
    }

    /// Allocate device memory for `data` and enqueue the copy into it.
    ///
    /// # Errors
    ///
    /// Whatever the allocation or the copy reports.
    pub fn create_with_data(&mut self, data: &[u8]) -> Result<Handle, IoError> {
        let mut staging =
            self.reserve_pinned(data.len(), None)
                .ok_or_else(|| IoError::Unknown {
                    backtrace: BackTrace::capture(),
                    description: "Unable to reserve pinned memory".into(),
                })?;

        staging.copy_from_slice(data);

        let handle = self.empty(staging.len() as u64)?;

        self.write_to_gpu(
            CopyDescriptor {
                handle: handle.clone().binding(),
                shape: [data.len()].into(),
                strides: [1].into(),
                elem_size: 1,
            },
            staging,
        )?;

        Ok(handle)
    }

    /// Wait for everything already enqueued on the current stream to finish.
    ///
    /// # Errors
    ///
    /// The fault the barrier reveals, when the stream itself failed.
    pub fn sync(&mut self) -> DynFut<Result<(), ServerError>> {
        let fence = Fence::new(self.streams.current().sys);

        Box::pin(async { fence.wait_sync() })
    }

    /// Executes a registered HIP kernel with the specified parameters.
    ///
    /// Always launches an already-compiled kernel: the server compiles before
    /// entering its write scope, and a skipped launch stops there, before any
    /// resource is resolved, so it never reaches here.
    ///
    /// # Errors
    ///
    /// The driver's refusal to enqueue the launch, returned whether or not a
    /// profile is open. An open profile is not a reason to hold the failure
    /// here: the caller's write scope is what claims the buffers the launch
    /// never wrote, and the caller invalidates every open profile on the same
    /// path, so keeping it would lose the claim and duplicate the report.
    pub fn kernel(
        &mut self,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        resources: &[GpuResource],
    ) -> Result<(), LaunchError> {
        let stream = self.streams.current();

        let result = self
            .ctx
            .execute_task(stream, kernel_id, dispatch_count, resources);

        // A fenced flush during capture would abort it; defer until the capture
        // ends (the deferred staging buffers are reclaimed then).
        if !stream.capturing.is_recording() && stream.drop_queue.should_flush() {
            stream.drop_queue.flush(|| Fence::new(stream.sys));
        }

        result
    }
}

/// The 2D shape of a copy to or from a pitched buffer: how wide each row is,
/// how many rows there are, and the stride between their starts.
///
/// [`of`](Self::of) answers `None` for a buffer that needs no 2D copy at all.
/// A row stride equal to the row width means no padding, so the whole buffer
/// is one contiguous span and the plain linear copy is both correct and
/// faster — the driver also refuses the 2D copy for very tall transfers, an
/// embedding table's 128k rows say, which the linear path handles.
struct Pitch {
    width_bytes: usize,
    height: usize,
    stride_bytes: usize,
}

impl Pitch {
    /// The pitch of a buffer with this layout, or `None` when its rows are
    /// contiguous. The caller has already validated the strides as pitched
    /// row-major.
    fn of(shape: &[usize], strides: &[usize], elem_size: usize) -> Option<Self> {
        let rank = shape.len();
        let width = *shape.last().unwrap_or(&1);
        if rank < 2 || strides[rank - 2] == width {
            return None;
        }
        Some(Self {
            width_bytes: width * elem_size,
            height: shape.iter().rev().skip(1).product(),
            stride_bytes: strides[rank - 2] * elem_size,
        })
    }
}

/// Asynchronously copies data from GPU device memory to host memory.
///
/// # Safety
///
/// - `resource_ptr` must be a valid HIP device pointer with at least `bytes.len()` readable bytes.
/// - `stream` must be a valid, initialized HIP stream.
/// - `bytes` must have sufficient capacity for the copy.
/// - The caller must synchronize the stream before reading from `bytes`.
unsafe fn write_to_cpu(
    shape: &[usize],
    strides: &[usize],
    elem_size: usize,
    bytes: &mut Bytes,
    resource_ptr: *mut c_void,
    stream: *mut ihipStream_t,
) -> Result<(), IoError> {
    // Nothing to copy for an empty tensor; `bytes` has no real backing (a
    // dangling zero-size buffer) and must not reach the driver.
    if bytes.is_empty() {
        return Ok(());
    }

    let Some(pitch) = Pitch::of(shape, strides, elem_size) else {
        // SAFETY: The data is contiguous. `resource_ptr` and `bytes` are valid
        // and `bytes.len()` does not exceed the device allocation size.
        let status = unsafe {
            cubecl_hip_sys::hipMemcpyDtoHAsync(
                bytes.as_mut_ptr() as *mut _,
                resource_ptr,
                bytes.len(),
                stream,
            )
        };

        checked("hipMemcpyDtoHAsync", status)?;
        return Ok(());
    };

    // SAFETY: The source is pitched. The 2D async copy respects the stride of
    // the second-to-last dimension; a flat copy would scramble the rows, so a
    // failure is an error rather than a fallback.
    let status = unsafe {
        cubecl_hip_sys::hipMemcpy2DAsync(
            bytes.as_mut_ptr() as *mut _,
            pitch.width_bytes,
            resource_ptr,
            pitch.stride_bytes,
            pitch.width_bytes,
            pitch.height,
            hipMemcpyKind_hipMemcpyDeviceToHost,
            stream,
        )
    };

    checked("hipMemcpy2DAsync", status).map_err(|err| IoError::Unknown {
        description: format!(
            "{err}; copying to the host from shape {shape:?}, strides {strides:?}, \
             elem_size {elem_size}, spitch {}, width {}, height {}",
            pitch.stride_bytes, pitch.width_bytes, pitch.height
        ),
        backtrace: BackTrace::capture(),
    })?;

    Ok(())
}

/// Asynchronously copies data from host memory to GPU device memory.
///
/// # Safety
///
/// - `resource.ptr` must be a valid HIP device pointer with at least `data.len()` writable bytes.
/// - `stream` must be a valid, initialized HIP stream.
/// - `data` must remain valid until the stream is synchronized.
/// - The shape and strides must describe a valid pitched row-major layout.
unsafe fn write_to_gpu(
    resource: GpuResource,
    shape: &Shape,
    strides: &Strides,
    elem_size: usize,
    data: &[u8],
    stream: *mut ihipStream_t,
) -> Result<(), IoError> {
    // Nothing to copy for an empty tensor; `data` may be a dangling (zero-size)
    // staging buffer that must not reach the driver.
    if data.is_empty() {
        return Ok(());
    }

    if !has_pitched_row_major_strides(shape, strides) {
        return Err(IoError::UnsupportedStrides {
            backtrace: BackTrace::capture(),
        });
    }

    let ptr = data as *const _ as *mut _;

    if let Some(pitch) = Pitch::of(shape, strides, elem_size) {
        // SAFETY: The destination is pitched. The 2D copy lays the rows out at
        // the stride the allocation was sized for; `resource.ptr` was allocated
        // with that pitched size.
        let status = unsafe {
            cubecl_hip_sys::hipMemcpy2DAsync(
                resource.ptr,
                pitch.stride_bytes,
                ptr,
                pitch.width_bytes,
                pitch.width_bytes,
                pitch.height,
                hipMemcpyKind_hipMemcpyHostToDevice,
                stream,
            )
        };
        checked("hipMemcpy2DAsync", status).map_err(|err| IoError::Unknown {
            description: format!(
                "{err}; copying to the device from shape {shape:?}, strides {strides:?}, \
                 elem_size {elem_size}, dpitch {}, width {}, height {}, resource size {}",
                pitch.stride_bytes, pitch.width_bytes, pitch.height, resource.size
            ),
            backtrace: BackTrace::capture(),
        })?;
    } else {
        if resource.size < data.len() as u64 {
            return Err(IoError::Unknown {
                description: format!(
                    "write of {} bytes exceeds the target buffer of {} bytes",
                    data.len(),
                    resource.size
                ),
                backtrace: BackTrace::capture(),
            });
        }
        // SAFETY: For rank <= 1 data is contiguous, the bound check above ensures the
        // device allocation is large enough, and `ptr` points to valid host data of
        // `data.len()` bytes.
        let status =
            unsafe { cubecl_hip_sys::hipMemcpyHtoDAsync(resource.ptr, ptr, data.len(), stream) };
        checked("hipMemcpyHtoDAsync", status)?;
    };

    Ok(())
}
