use crate::{
    compute::{
        MB, context::HipContext, fence::Fence, gpu::GpuResource,
        io::controller::PinnedMemoryManagedAllocController, stream::HipStreamBackend,
    },
    runtime::HipCompiler,
};
use cubecl_common::{backtrace::BackTrace, bytes::Bytes, stream_id::StreamId};
use cubecl_core::{
    MemoryUsage,
    future::DynFut,
    server::{
        Binding, CopyDescriptor, ExecutionMode, HandleId, IoError, LaunchError, MemorySlot,
        ProfileError, ServerError,
    },
    zspace::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use cubecl_hip_sys::{
    HIP_SUCCESS, hipMemcpyKind_hipMemcpyDeviceToHost, hipMemcpyKind_hipMemcpyHostToDevice,
    ihipStream_t,
};
use cubecl_runtime::{
    compiler::CubeTask,
    id::KernelId,
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, MemoryHandle},
    stream::ResolvedStreams,
};
use std::{ffi::c_void, sync::Arc};

#[derive(new)]
/// The `Command` struct encapsulates a HIP context and a set of resolved HIP streams, providing an
/// interface for executing GPU-related operations such as memory allocation, data transfers, kernel
/// registration, and task execution.
pub struct Command<'a> {
    ctx: &'a mut HipContext,
    pub(crate) streams: ResolvedStreams<'a, HipStreamBackend>,
}

impl<'a> Command<'a> {
    /// Retrieves a GPU resource associated with the provided binding.
    ///
    /// # Parameters
    ///
    /// * `binding` - The binding specifying the stream, memory, and offsets for the resource.
    ///
    /// # Returns
    ///
    /// * `Ok(GpuResource)` - The GPU resource associated with the binding.
    /// * `Err(IoError::InvalidHandle)` - If the binding does not correspond to a valid resource.
    pub fn resource(
        &mut self,
        handle: Binding,
    ) -> Result<(GpuResource, ManagedMemoryHandle), IoError> {
        let mm = &mut self.streams.get(&handle.stream).memory_management_gpu;
        let slot = mm.get_slot(handle)?;

        let resource = mm
            .get_resource(
                slot.memory.clone().binding(),
                slot.offset_start,
                slot.offset_end,
            )
            .ok_or(IoError::InvalidHandle {
                backtrace: BackTrace::capture(),
            })?;

        Ok((resource, slot.memory))
    }

    /// Retrieves the gpu memory usage of the current stream.
    ///
    /// # Returns
    ///
    /// * The [`MemoryUsage`] struct.
    pub fn memory_usage(&mut self) -> MemoryUsage {
        self.streams.current().memory_management_gpu.memory_usage()
    }

    /// Explicitly cleanup gpu memory on the current stream.
    pub fn memory_cleanup(&mut self) {
        self.streams.current().memory_management_gpu.cleanup(true)
    }

    /// Set the [`MemoryAllocationMode`] for the current stream.
    ///
    /// # Parameters
    ///
    /// * `mode` - The allocation mode to be used.
    pub fn allocation_mode(&mut self, mode: MemoryAllocationMode) {
        self.streams.current().memory_management_gpu.mode(mode)
    }

    /// Allocates a new GPU memory buffer of the specified size.
    ///
    /// # Parameters
    ///
    /// * `size` - The size of the memory to allocate (in bytes).
    ///
    /// # Returns
    ///
    /// * `Ok(Handle)` - A handle to the newly allocated GPU memory.
    /// * `Err(IoError)` - If the allocation fails.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        let handle = self.streams.current().memory_management_gpu.reserve(size)?;

        Ok(handle)
    }

    /// Get the stream cursor.
    pub fn cursor(&self) -> u64 {
        self.streams.cursor
    }

    /// * `Err(IoError)` - If the allocation fails.
    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn empty(&mut self, size: u64, single_use: bool) -> Result<Binding, IoError> {
        let handle = Binding::new_manual(self.streams.current, size, single_use);
        let memory = self.reserve(handle.size())?;
        let slot = memory.into_slot(handle.clone(), self.streams.cursor, self.streams.current);
        self.bind(handle.clone(), slot);

        Ok(handle)
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn bind(&mut self, handle: Binding, memory: MemorySlot) {
        self.streams
            .current()
            .memory_management_gpu
            .bind(handle.id, memory);
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(level = "trace", skip(self)))]
    pub fn free(&mut self, handle: HandleId) {
        let stream = self.streams.current();

        stream.memory_management_gpu.free(handle);
    }

    /// Creates a [Bytes] instance from pinned memory, if suitable for the given size.
    ///
    /// For small data transfers (<= 100 MB) or when explicitly marked as pinned, this function
    /// uses pinned memory to optimize performance. For larger transfers, it falls back to regular memory.
    ///
    /// # Arguments
    ///
    /// * `size` - The number of bytes to allocate.
    /// * `marked_pinned` - Whether to force the use of pinned memory.
    ///
    /// # Returns
    ///
    /// A [Bytes] instance of the correct size.
    pub fn reserve_cpu(
        &mut self,
        size: usize,
        marked_pinned: bool,
        origin: Option<StreamId>,
    ) -> Bytes {
        // Use pinned memory for small transfers (<= 100 MB) or when explicitly marked.
        if !marked_pinned && size > 100 * MB {
            return Bytes::from_bytes_vec(vec![0; size]);
        }

        self.reserve_pinned(size, origin)
            .unwrap_or_else(|| Bytes::from_bytes_vec(vec![0; size]))
    }

    fn reserve_pinned(&mut self, size: usize, origin: Option<StreamId>) -> Option<Bytes> {
        let stream = match origin {
            Some(id) => self.streams.get(&id),
            None => self.streams.current(),
        };
        let handle = stream.memory_management_cpu.reserve(size as u64).ok()?;

        let binding = MemoryHandle::binding(handle);
        let resource = stream
            .memory_management_cpu
            .get_resource(binding.clone(), None, None)
            .ok_or(IoError::InvalidHandle {
                backtrace: BackTrace::capture(),
            })
            .ok()?;

        let controller = Box::new(PinnedMemoryManagedAllocController::init(binding, resource));
        // SAFETY: The binding has initialized memory for at least `size` bytes.
        Some(unsafe { Bytes::from_controller(controller, size) })
    }

    /// Asynchronously reads data from GPU memory to host memory based on the provided copy descriptors.
    ///
    /// # Parameters
    ///
    /// * `descriptors` - A vector of descriptors specifying the source GPU memory and its layout.
    ///
    /// # Returns
    ///
    /// * A `Future` resolving to:
    ///   * `Ok(Vec<Bytes>)` - The data read from the GPU as a vector of byte arrays.
    ///   * `Err(IoError)` - If the read operation fails.
    pub fn read_async(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
    ) -> impl Future<Output = Result<Vec<Bytes>, ServerError>> + Send + use<> {
        let descriptors_moved = descriptors
            .iter()
            .map(|b| b.handle.clone())
            .collect::<Vec<_>>();
        let result = self.copies_to_bytes(descriptors, true);
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

    #[allow(unused)]
    /// TODO: Read data using the origin stream where the data was allocated.
    pub fn read_async_origin(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
    ) -> impl Future<Output = Result<Vec<Bytes>, IoError>> + Send + use<> {
        let results = self.copies_to_bytes_origin(descriptors, true);

        async move {
            let (bytes, fences) = results?;

            for fence in fences {
                fence.wait_sync();
            }
            Ok(bytes)
        }
    }

    fn copies_to_bytes(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        pinned: bool,
    ) -> Result<Vec<Bytes>, IoError> {
        let mut result = Vec::with_capacity(descriptors.len());

        for descriptor in descriptors {
            result.push(self.copy_to_bytes(descriptor, pinned, None)?);
        }

        Ok(result)
    }

    fn copies_to_bytes_origin(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        pinned: bool,
    ) -> Result<(Vec<Bytes>, Vec<Fence>), IoError> {
        let mut data = Vec::with_capacity(descriptors.len());
        let mut fences = Vec::with_capacity(descriptors.len());
        let mut fenced = Vec::with_capacity(descriptors.len());

        for descriptor in descriptors {
            let stream = descriptor.handle.stream;
            let bytes = self.copy_to_bytes(descriptor, pinned, Some(stream))?;

            if !fenced.contains(&stream) {
                let fence = Fence::new(self.streams.get(&stream).sys);
                fenced.push(stream);
                fences.push(fence);
            }

            data.push(bytes);
        }

        Ok((data, fences))
    }

    fn copy_to_bytes(
        &mut self,
        descriptor: CopyDescriptor,
        pinned: bool,
        stream_id: Option<StreamId>,
    ) -> Result<Bytes, IoError> {
        let num_bytes = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        let mut bytes = self.reserve_cpu(num_bytes, pinned, stream_id);
        self.write_to_cpu(descriptor, &mut bytes, stream_id)?;

        Ok(bytes)
    }

    /// Writes data to the host from the GPU memory as specified by the copy descriptor.
    ///
    /// # Parameters
    ///
    /// * `descriptor` - Describes the source GPU memory, its shape, strides, and element size.
    /// * `bytes` - The host bytes to write from the GPU.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the write operation succeeds.
    /// * `Err(IoError)` - If the strides are invalid or the resource cannot be accessed.
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

        let (resource, _handle) = self.resource(binding)?;
        let stream = match stream_id {
            Some(id) => self.streams.get(&id),
            None => self.streams.current(),
        };

        unsafe { write_to_cpu(&shape, &strides, elem_size, bytes, resource.ptr, stream.sys) }
    }

    /// Writes data from the host to GPU memory as specified by the copy descriptor.
    ///
    /// # Parameters
    ///
    /// * `descriptor` - Describes the destination GPU memory, its shape, strides, and element size.
    /// * `data` - The host data to write to the GPU.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the write operation succeeds.
    /// * `Err(IoError)` - If the strides are invalid or the resource cannot be accessed.
    pub fn write_to_gpu(
        &mut self,
        descriptor: CopyDescriptor,
        bytes: &Bytes,
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

        let (resource, _handle) = self.resource(binding)?;
        let current = self.streams.current();

        unsafe {
            write_to_gpu(resource, &shape, &strides, elem_size, bytes, current.sys)?;
        };

        Ok(())
    }

    /// Allocates a new GPU memory buffer and immediately copies contiguous host data into it.
    ///
    /// # Parameters
    ///
    /// * `data` - The host data to copy to the GPU.
    ///
    /// # Returns
    ///
    /// * `Ok(Handle)` - A handle to the newly allocated and populated GPU memory.
    /// * `Err(IoError)` - If the allocation or data copy fails.
    pub fn create_with_data(&mut self, data: &[u8], single_use: bool) -> Result<Binding, IoError> {
        let handle = self.empty(data.len() as u64, single_use)?;
        let shape: Shape = [data.len()].into();
        let elem_size = 1;
        let strides: Strides = [1].into();
        if !has_pitched_row_major_strides(&shape, &strides) {
            return Err(IoError::UnsupportedStrides {
                backtrace: BackTrace::capture(),
            });
        }

        let (resource, _handle) = self.resource(handle.clone())?;

        let current = self.streams.current();

        unsafe {
            write_to_gpu(resource, &shape, &strides, elem_size, data, current.sys)?;
        }

        Ok(handle)
    }

    /// Synchronizes the current stream, ensuring all pending operations are complete.
    ///
    /// # Returns
    ///
    /// * A `DynFut<()>` future that resolves when the stream is synchronized.
    pub fn sync(&mut self) -> DynFut<Result<(), ServerError>> {
        let fence = Fence::new(self.streams.current().sys);

        Box::pin(async { fence.wait_sync() })
    }

    /// Executes a registered CUDA kernel with the specified parameters.
    ///
    /// # Parameters
    ///
    /// * `kernel_id` - The identifier of the kernel to execute.
    /// * `kernel` - The cube task to compile if not cached.
    /// * `mode` - The execution mode for the current kernel.
    /// * `dispatch_count` - The number of thread blocks in the x, y, and z dimensions.
    /// * `resources` - GPU resources (e.g., buffers) used by the kernel.
    /// * `logger` - The logger to use to write compilation & runtime info.
    ///
    /// # Panics
    ///
    /// * If the execution fails, with an error message or profiling error.
    pub fn kernel(
        &mut self,
        kernel_id: KernelId,
        kernel: Box<dyn CubeTask<HipCompiler>>,
        mode: ExecutionMode,
        dispatch_count: (u32, u32, u32),
        resources: &[GpuResource],
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        if !self.ctx.module_names.contains_key(&kernel_id) {
            self.ctx.compile_kernel(&kernel_id, kernel, mode, logger)?;
        }

        let stream = self.streams.current();

        let result = self
            .ctx
            .execute_task(stream, kernel_id, dispatch_count, resources);

        if let Err(err) = result {
            match self.ctx.timestamps.is_empty() {
                true => Err(err)?,
                false => self.ctx.timestamps.error(ProfileError::Launch(err)),
            }
        };

        Ok(())
    }

    pub fn error(&mut self, error: ServerError) {
        let stream = self.streams.current();
        stream.errors.push(error);
    }
}

pub(crate) unsafe fn write_to_cpu(
    shape: &[usize],
    strides: &[usize],
    elem_size: usize,
    bytes: &mut Bytes,
    resource_ptr: *mut c_void,
    stream: *mut ihipStream_t,
) -> Result<(), IoError> {
    let rank = shape.len();

    if rank <= 1 {
        let status = unsafe {
            cubecl_hip_sys::hipMemcpyDtoHAsync(
                bytes.as_mut_ptr() as *mut _,
                resource_ptr,
                bytes.len(),
                stream,
            )
        };

        if status != HIP_SUCCESS {
            return Err(IoError::Unknown {
                description: format!("HIP memcpy failed: {status}"),
                backtrace: BackTrace::capture(),
            });
        }
        return Ok(());
    }

    let dim_x = shape[rank - 1];
    let width_bytes = dim_x * elem_size;
    let dim_y: usize = shape.iter().rev().skip(1).product();
    let pitch = strides[rank - 2] * elem_size;

    unsafe {
        let status = cubecl_hip_sys::hipMemcpy2DAsync(
            bytes.as_mut_ptr() as *mut _,
            width_bytes,
            resource_ptr,
            pitch,
            width_bytes,
            dim_y,
            hipMemcpyKind_hipMemcpyDeviceToHost,
            stream,
        );

        // Fallback, sometimes the copy doesn't work.
        if status != HIP_SUCCESS {
            let status = cubecl_hip_sys::hipMemcpyDtoHAsync(
                bytes.as_mut_ptr() as *mut _,
                resource_ptr,
                bytes.len(),
                stream,
            );
            assert_eq!(status, HIP_SUCCESS, "Should send data to device");
        }
    }

    Ok(())
}

unsafe fn write_to_gpu(
    resource: GpuResource,
    shape: &Shape,
    strides: &Strides,
    elem_size: usize,
    data: &[u8],
    stream: *mut ihipStream_t,
) -> Result<(), IoError> {
    let rank = shape.len();

    if !has_pitched_row_major_strides(shape, strides) {
        return Err(IoError::UnsupportedStrides {
            backtrace: BackTrace::capture(),
        });
    }

    if rank > 1 {
        let stride = strides[rank - 2];
        let width = *shape.last().unwrap_or(&1);
        let height: usize = shape.iter().rev().skip(1).product();
        let width_bytes = width * elem_size;
        let stride_bytes = stride * elem_size;

        unsafe {
            let status = cubecl_hip_sys::hipMemcpy2DAsync(
                resource.ptr,
                stride_bytes,
                data as *const _ as *mut _,
                width_bytes,
                width_bytes,
                height.max(1),
                hipMemcpyKind_hipMemcpyHostToDevice,
                stream,
            );
            assert_eq!(status, HIP_SUCCESS, "Should send data to device");
        }
    } else {
        unsafe {
            let status = cubecl_hip_sys::hipMemcpyHtoDAsync(
                resource.ptr,
                data as *const _ as *mut _,
                data.len(),
                stream,
            );
            assert_eq!(status, HIP_SUCCESS, "Should send data to device");
        }
    };

    Ok(())
}
