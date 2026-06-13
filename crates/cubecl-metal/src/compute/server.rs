use crate::{
    MetalCompiler,
    compute::context::MetalContext,
    compute::stream::MetalStreamBackend,
    memory::{MetalBufferHandle, MetalStorage},
};
use cubecl_common::{bytes::Bytes, stream_id::StreamId};
use cubecl_core::{
    MemoryConfiguration,
    future::DynFut,
    prelude::*,
    server::{
        Binding, CopyDescriptor, IoError, KernelArguments, ProfileError, ProfilingToken,
        ServerCommunication, ServerError, ServerUtilities,
    },
};
use cubecl_runtime::{
    allocator::ContiguousMemoryLayoutPolicy,
    compiler::CubeTask,
    logging::ServerLogger,
    memory_management::ManagedMemoryHandle,
    server::ComputeServer,
    storage::{ComputeStorage, ManagedResource},
    stream::{EventStreamBackend, MultiStream, ResolvedStreams},
    timestamp_profiler::TimestampProfiler,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use std::sync::Arc;

enum DispatchInfo {
    Static(u32, u32, u32),
    Dynamic(Binding),
}

/// Metal compute server.
#[derive(Debug)]
pub struct MetalServer {
    context: MetalContext,
    streams: MultiStream<MetalStreamBackend>,
    pub(crate) utilities: Arc<ServerUtilities<Self>>,
    timestamps: TimestampProfiler,
    errors: Vec<ServerError>,
}

impl MetalServer {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        mem_props: cubecl_ir::MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        utilities: Arc<ServerUtilities<Self>>,
    ) -> Self {
        let logger = utilities.logger.clone();

        let compilation_options = cubecl_cpp::shared::CompilationOptions::default();
        let context = MetalContext::new(device.clone(), compilation_options);

        let backend = MetalStreamBackend::new(device, mem_props, mem_config, logger.clone());

        let config = {
            use cubecl_runtime::config::RuntimeConfig;
            cubecl_runtime::config::CubeClRuntimeConfig::get()
        };
        let max_streams = config.streaming.max_streams;

        Self {
            context,
            streams: MultiStream::new(logger, backend, max_streams),
            utilities,
            timestamps: TimestampProfiler::default(),
            errors: Vec::new(),
        }
    }
}

// SAFETY: Only accessed from the server thread. GPU work is serialized through command queue ordering.
unsafe impl Send for MetalServer {}

/// Resolves a binding's GPU resource from its origin stream ([`Binding::stream`]),
/// not the stream issuing work. Each stream owns its own `memory_management`, so a
/// buffer only lives in its origin's manager.
fn resolve_origin_resource(
    resolved: &mut ResolvedStreams<'_, MetalStreamBackend>,
    binding: &Binding,
) -> Result<(MetalBufferHandle, u64), IoError> {
    let stream = resolved.get(&binding.stream);

    let mut storage_handle = stream
        .memory_management
        .get_storage(binding.memory.clone())?;
    if let Some(offset) = binding.offset_start {
        storage_handle = storage_handle.offset_start(offset);
    }
    if let Some(offset) = binding.offset_end {
        storage_handle = storage_handle.offset_end(offset);
    }

    let offset = storage_handle.offset();
    let resource = stream.memory_management.storage().get(&storage_handle);

    Ok((resource, offset))
}

impl MetalServer {
    /// Collects synchronous failures buffered on the server together with asynchronous GPU
    /// faults recorded by stream completion handlers. Draining the stream sink clears the
    /// poison so a recovered stream can serve new work.
    fn flush_errors(&mut self, stream_id: StreamId) -> Vec<ServerError> {
        let mut errors = core::mem::take(&mut self.errors);

        if let Ok(mut resolved) = self.streams.resolve(stream_id, std::iter::empty(), false) {
            errors.append(&mut resolved.current().take_errors());
        }

        if !errors.is_empty() {
            self.timestamps.error(ProfileError::Unknown {
                reason: format!("{errors:?}"),
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
            });
        }

        errors
    }
}

impl ServerCommunication for MetalServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl ComputeServer for MetalServer {
    type Kernel = Box<dyn CubeTask<MetalCompiler>>;
    type Storage = MetalStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.utilities.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        // Unnecessary: shared storage gives direct CPU access to GPU buffers.
        Err(IoError::UnsupportedIoOperation {
            backtrace: cubecl_common::backtrace::BackTrace::capture(),
        }
        .into())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let mut resolved = self
            .streams
            .resolve(stream_id, std::iter::empty(), false)
            .expect("Failed to resolve stream for initialize_memory");
        let cursor = resolved.cursor;
        let stream = resolved.current();

        let reserved = stream
            .memory_management
            .reserve(size)
            .expect("Failed to reserve memory");
        stream
            .memory_management
            .bind(reserved, memory, cursor)
            .expect("Failed to bind memory");
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        use objc2_metal::MTLBuffer;

        let errors = self.flush_errors(stream_id);
        if !errors.is_empty() {
            return Box::pin(async move {
                Err(ServerError::ServerUnhealthy {
                    errors,
                    backtrace: cubecl_common::backtrace::BackTrace::capture(),
                })
            });
        }

        let mut resolved =
            match self
                .streams
                .resolve(stream_id, descriptors.iter().map(|d| &d.handle), false)
            {
                Ok(r) => r,
                Err(e) => return Box::pin(async move { Err(e) }),
            };

        // Flush, wait, then read.
        let stream = resolved.current();
        let event = MetalStreamBackend::flush(stream);

        if let Err(e) = MetalStreamBackend::wait_event_sync(event) {
            return Box::pin(async move { Err(e) });
        }

        let results: Result<Vec<_>, ServerError> = descriptors
            .iter()
            .map(|descriptor| {
                let (resource, offset) = resolve_origin_resource(&mut resolved, &descriptor.handle)
                    .map_err(ServerError::from)?;

                let size: usize = descriptor.shape.iter().product();
                let size_bytes = size * descriptor.elem_size;

                let buffer = resource.inner();
                let protocol_obj: &ProtocolObject<dyn MTLBuffer> = buffer.as_ref();
                let base_ptr = protocol_obj.contents().as_ptr() as *const u8;
                let contents_ptr = unsafe { base_ptr.add(offset as usize) };
                let data = unsafe { std::slice::from_raw_parts(contents_ptr, size_bytes) };

                Ok(Bytes::from_bytes_vec(data.to_vec()))
            })
            .collect();

        Box::pin(async move { results })
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        use objc2_metal::MTLBuffer;

        let mut resolved =
            match self
                .streams
                .resolve(stream_id, descriptors.iter().map(|(d, _)| &d.handle), false)
            {
                Ok(r) => r,
                Err(e) => {
                    log::warn!("metal write: failed to resolve stream: {e}");
                    return;
                }
            };

        let stream = resolved.current();
        let event = MetalStreamBackend::flush(stream);
        if let Err(err) = MetalStreamBackend::wait_event_sync(event) {
            log::warn!("metal write: sync failed: {err}");
            return;
        }

        for (descriptor, data) in descriptors {
            let (resource, offset) =
                match resolve_origin_resource(&mut resolved, &descriptor.handle) {
                    Ok(r) => r,
                    Err(e) => {
                        log::warn!("metal write: buffer not found: {e}");
                        continue;
                    }
                };

            let size: usize = descriptor.shape.iter().product();
            let size_bytes = size * descriptor.elem_size;

            let buffer = resource.inner();
            let protocol_obj: &ProtocolObject<dyn MTLBuffer> = buffer.as_ref();
            let base_ptr = protocol_obj.contents().as_ptr() as *mut u8;
            let write_ptr = unsafe { base_ptr.add(offset as usize) };

            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), write_ptr, size_bytes);
            }
        }
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions};

        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        if let Err(err) =
            cubecl_runtime::validation::validate_cube_dim(&self.utilities.properties, &kernel_id)
        {
            self.errors.push(ServerError::Launch(err));
            return;
        }
        if let Err(err) =
            cubecl_runtime::validation::validate_units(&self.utilities.properties, &kernel_id)
        {
            self.errors.push(ServerError::Launch(err));
            return;
        }

        let compiled = match self.context.compile_kernel(
            &kernel_id,
            kernel,
            mode,
            self.utilities.logger.clone(),
        ) {
            Ok(c) => c,
            Err(err) => {
                self.errors.push(ServerError::Launch(
                    cubecl_core::prelude::LaunchError::CompilationError(err),
                ));
                return;
            }
        };

        // Validate shared memory usage
        let max_smem = self.utilities.properties.hardware.max_shared_memory_size;
        if compiled.shared_memory_bytes > max_smem {
            use cubecl_core::server::ResourceLimitError;
            self.errors.push(ServerError::Launch(
                cubecl_core::prelude::LaunchError::TooManyResources(
                    ResourceLimitError::SharedMemory {
                        requested: compiled.shared_memory_bytes,
                        max: max_smem,
                        backtrace: cubecl_common::backtrace::BackTrace::capture(),
                    },
                ),
            ));
            return;
        }

        let dispatch_info = match count {
            CubeCount::Static(x, y, z) => DispatchInfo::Static(x, y, z),
            CubeCount::Dynamic(binding) => DispatchInfo::Dynamic(binding),
        };

        // Resolve every binding (including the dynamic count) so the current stream
        // waits on each binding's origin stream before dispatching.
        let mut resolved = match self.streams.resolve(
            stream_id,
            bindings.buffers.iter().chain(match &dispatch_info {
                DispatchInfo::Dynamic(binding) => Some(binding),
                DispatchInfo::Static(..) => None,
            }),
            false,
        ) {
            Ok(r) => r,
            Err(_) => return,
        };

        let mut resources = Vec::with_capacity(bindings.buffers.len());
        let mut total_buffer_bytes: usize = 0;
        for binding in bindings.buffers.iter() {
            let (resource, offset) = match resolve_origin_resource(&mut resolved, binding) {
                Ok(r) => r,
                Err(_) => return,
            };

            total_buffer_bytes += binding.size_in_used() as usize;

            resources.push((resource, offset));
        }

        // The indirect count buffer is read GPU-side, so it too comes from its origin stream.
        let indirect_buffer_info = match &dispatch_info {
            DispatchInfo::Dynamic(binding) => {
                match resolve_origin_resource(&mut resolved, binding) {
                    Ok(r) => Some(r),
                    Err(_) => return,
                }
            }
            _ => None,
        };

        // Work is issued on the current stream; resources above came from their origins.
        let stream = resolved.current();

        let device = stream.device.clone();
        let active = stream.get_or_create_encoder();
        let encoder = &active.encoder;

        (*encoder).setComputePipelineState(&compiled.pipeline);

        for (index, (resource, offset)) in resources.iter().enumerate() {
            let buffer: &ProtocolObject<dyn MTLBuffer> = resource.inner().as_ref();
            unsafe {
                (*encoder).setBuffer_offset_atIndex(Some(buffer), *offset as usize, index);
            }
        }

        let buffer_index = resources.len();

        if !bindings.info.data.is_empty() {
            let info_bytes: &[u8] = bytemuck::cast_slice(&bindings.info.data);
            if info_bytes.len() <= 4096 {
                use std::ptr::NonNull;
                unsafe {
                    (*encoder).setBytes_length_atIndex(
                        NonNull::new(info_bytes.as_ptr() as *mut _).unwrap(),
                        info_bytes.len(),
                        buffer_index,
                    );
                }
            } else {
                use std::ptr::NonNull;
                let info_buffer = unsafe {
                    (*device).newBufferWithBytes_length_options(
                        NonNull::new(info_bytes.as_ptr() as *mut _).unwrap(),
                        info_bytes.len(),
                        MTLResourceOptions::StorageModeShared,
                    )
                };
                match info_buffer {
                    Some(buf) => {
                        unsafe {
                            (*encoder).setBuffer_offset_atIndex(Some(&buf), 0, buffer_index);
                        }
                        active.temporaries.push(buf);
                    }
                    None => return,
                }
            }
        }

        let cube_dim = compiled.cube_dim;
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: cube_dim.x as usize,
            height: cube_dim.y as usize,
            depth: cube_dim.z as usize,
        };

        match dispatch_info {
            DispatchInfo::Static(grid_x, grid_y, grid_z) => {
                let threadgroups = objc2_metal::MTLSize {
                    width: grid_x as usize,
                    height: grid_y as usize,
                    depth: grid_z as usize,
                };

                (*encoder).dispatchThreadgroups_threadsPerThreadgroup(
                    threadgroups,
                    threads_per_threadgroup,
                );
            }
            DispatchInfo::Dynamic(_) => {
                let (resource, offset) = indirect_buffer_info.unwrap();
                let buffer: &ProtocolObject<dyn MTLBuffer> = resource.inner().as_ref();

                unsafe {
                    (*encoder)
                        .dispatchThreadgroupsWithIndirectBuffer_indirectBufferOffset_threadsPerThreadgroup(
                            buffer,
                            offset as usize,
                            threads_per_threadgroup,
                        );
                }
            }
        }

        stream.batch_ops += 1;
        stream.batch_bytes += total_buffer_bytes;

        let needs_flush = stream.batch_ops > stream.max_ops_per_batch
            || (stream.batch_bytes >> 20) > stream.max_mb_per_batch;

        if needs_flush {
            MetalStreamBackend::flush(stream);
        }
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        let errors = self.flush_errors(stream_id);
        if !errors.is_empty() {
            return Box::pin(async move {
                Err(ServerError::ServerUnhealthy {
                    errors,
                    backtrace: cubecl_common::backtrace::BackTrace::capture(),
                })
            });
        }

        let mut resolved = match self.streams.resolve(stream_id, std::iter::empty(), false) {
            Ok(r) => r,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        let fence = MetalStreamBackend::flush(resolved.current());

        Box::pin(async move { MetalStreamBackend::wait_event_sync(fence) })
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let errors = self.flush_errors(stream_id);
        if !errors.is_empty() {
            return Err(ServerError::ServerUnhealthy {
                errors,
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
            });
        }

        let mut resolved = self.streams.resolve(stream_id, std::iter::empty(), false)?;
        MetalStreamBackend::flush(resolved.current());
        Ok(())
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        if let Err(err) = cubecl_common::future::block_on(self.sync(stream_id)) {
            log::warn!("{err}");
        }
        Ok(self.timestamps.start())
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<cubecl_common::profile::ProfileDuration, ProfileError> {
        if let Err(err) = cubecl_common::future::block_on(self.sync(stream_id)) {
            self.timestamps.error(ProfileError::Unknown {
                reason: format!("{err}"),
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
            });
        }
        self.timestamps.stop(token)
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<<MetalStorage as ComputeStorage>::Resource>, ServerError> {
        let mut resolved = self
            .streams
            .resolve(stream_id, std::iter::once(&binding), false)?;
        // Resolve from the binding's origin stream; see `resolve_origin_resource`.
        let stream = resolved.get(&binding.stream);

        let memory = binding.memory.clone();
        let resource = stream
            .memory_management
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
            .map_err(ServerError::from)?;

        Ok(ManagedResource::new(memory, resource))
    }

    fn memory_usage(
        &mut self,
        stream_id: StreamId,
    ) -> Result<cubecl_runtime::memory_management::MemoryUsage, ServerError> {
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty(), false)?;
        let stream = resolved.current();
        Ok(stream.memory_management.memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        if let Ok(mut resolved) = self.streams.resolve(stream_id, std::iter::empty(), false) {
            let stream = resolved.current();
            stream.memory_management.cleanup(true);
        }
    }

    fn allocation_mode(
        &mut self,
        mode: cubecl_runtime::memory_management::MemoryAllocationMode,
        stream_id: StreamId,
    ) {
        if let Ok(mut resolved) = self.streams.resolve(stream_id, std::iter::empty(), false) {
            let stream = resolved.current();
            stream.memory_management.mode(mode);
        }
    }
}
