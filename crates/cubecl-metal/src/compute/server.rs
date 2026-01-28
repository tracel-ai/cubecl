use crate::{
    compute::context::MetalContext,
    compute::stream::{MetalStreamBackend, PendingCommandBuffer},
    memory::MetalStorage,
    MetalCompiler,
};
use cubecl_common::{bytes::Bytes, stream_id::StreamId};
use cubecl_core::{
    future::DynFut,
    prelude::*,
    server::{
        Allocation, AllocationDescriptor, Binding, Bindings, CopyDescriptor, ExecutionError,
        Handle, IoError, LaunchError, ProfileError, ProfilingToken, ServerCommunication,
        ServerUtilities,
    },
    MemoryConfiguration,
};
use cubecl_runtime::{
    compiler::CubeTask,
    logging::ServerLogger,
    server::ComputeServer,
    storage::{BindingResource, ComputeStorage},
    stream::MultiStream,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLDevice};
use std::sync::Arc;

/// Metal compute server
#[derive(Debug)]
pub struct MetalServer {
    context: MetalContext,
    streams: MultiStream<MetalStreamBackend>,
    utilities: Arc<ServerUtilities<Self>>,
}

impl MetalServer {
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        mem_props: cubecl_ir::MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        utilities: Arc<ServerUtilities<Self>>,
    ) -> Self {
        let logger = utilities.logger.clone();

        // Create compilation context
        let compilation_options = cubecl_cpp::shared::CompilationOptions::default();
        let context = MetalContext::new(device.clone(), compilation_options);

        let backend = MetalStreamBackend::new(device, mem_props, mem_config, logger.clone());

        let config = cubecl_runtime::config::GlobalConfig::get();
        // Use at least 32 streams to ensure each thread gets its own stream.
        // With fewer streams, different thread IDs map to the same physical stream,
        // causing synchronization issues since cross-stream sync only detects conflicts
        // between different stream IDs, not when they share the same underlying stream.
        let max_streams = config.streaming.max_streams.max(32);

        Self {
            context,
            streams: MultiStream::new(logger, backend, max_streams),
            utilities,
        }
    }
}

unsafe impl Send for MetalServer {}

impl ServerCommunication for MetalServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl ComputeServer for MetalServer {
    type Kernel = Box<dyn CubeTask<MetalCompiler>>;
    type Storage = MetalStorage;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.utilities.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(&mut self, _sizes: &[usize], _stream_id: StreamId) -> Result<Vec<Bytes>, IoError> {
        // Not needed - Metal's shared memory mode allows direct CPU-GPU buffer access
        Err(IoError::UnsupportedIoOperation {
            backtrace: cubecl_common::backtrace::BackTrace::capture(),
        })
    }

    fn create(
        &mut self,
        descriptors: Vec<AllocationDescriptor<'_>>,
        stream_id: StreamId,
    ) -> Result<Vec<Allocation>, IoError> {
        let mut allocations = Vec::with_capacity(descriptors.len());
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        let cursor = resolved.cursor;
        let stream = resolved.current();

        for descriptor in descriptors {
            // Calculate size and strides
            let size: usize = descriptor.shape.iter().product();
            let size_bytes = size * descriptor.elem_size;

            let mut strides = vec![1; descriptor.shape.len()];
            for i in (0..descriptor.shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * descriptor.shape[i + 1];
            }

            // Reserve memory through the memory management system
            let slice_handle = stream.memory_management.reserve(size_bytes as u64)?;

            // Create Handle with the slice handle
            let handle = Handle::new(
                slice_handle,
                None,
                None,
                stream_id,
                cursor,
                size_bytes as u64,
            );

            allocations.push(Allocation::new(handle, strides));
        }

        Ok(allocations)
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor<'_>>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, IoError>> {
        use cubecl_runtime::stream::EventStreamBackend;
        use objc2_metal::MTLBuffer;

        let mut resolved = self
            .streams
            .resolve(stream_id, descriptors.iter().map(|d| &d.binding));
        let stream = resolved.current();

        // Flush and wait for all GPU work to complete before reading buffer contents
        let event = MetalStreamBackend::flush(stream);
        MetalStreamBackend::wait_event_sync(event).expect("Failed to wait for stream sync");

        let results: Result<Vec<_>, IoError> = descriptors
            .iter()
            .map(|descriptor| {
                let handle = stream
                    .memory_management
                    .get(descriptor.binding.memory.clone())
                    .expect("Handle should exist");

                let handle = match descriptor.binding.offset_start {
                    Some(offset) => handle.offset_start(offset),
                    None => handle,
                };
                let handle = match descriptor.binding.offset_end {
                    Some(offset) => handle.offset_end(offset),
                    None => handle,
                };

                let offset = handle.offset();
                let resource = stream.memory_management.storage().get(&handle);

                let size: usize = descriptor.shape.iter().product();
                let size_bytes = size * descriptor.elem_size;

                let buffer = resource.as_ref();
                let protocol_obj: &ProtocolObject<dyn MTLBuffer> = buffer.as_ref();
                let base_ptr = protocol_obj.contents().as_ptr() as *const u8;
                let contents_ptr = unsafe { base_ptr.add(offset as usize) };

                let data = unsafe { std::slice::from_raw_parts(contents_ptr, size_bytes) };
                Ok(Bytes::from_bytes_vec(data.to_vec()))
            })
            .collect();

        Box::pin(async move { results })
    }

    fn write(
        &mut self,
        descriptors: Vec<(CopyDescriptor<'_>, Bytes)>,
        stream_id: StreamId,
    ) -> Result<(), IoError> {
        use cubecl_runtime::stream::EventStreamBackend;
        use objc2_metal::MTLBuffer;

        let mut resolved = self
            .streams
            .resolve(stream_id, descriptors.iter().map(|(d, _)| &d.binding));
        let stream = resolved.current();

        // Flush and wait for all GPU work to complete before CPU writes
        let event = MetalStreamBackend::flush(stream);
        MetalStreamBackend::wait_event_sync(event).expect("Failed to wait for stream sync");

        for (descriptor, data) in descriptors {
            let handle = stream
                .memory_management
                .get(descriptor.binding.memory)
                .expect("Handle should exist");

            let handle = match descriptor.binding.offset_start {
                Some(offset) => handle.offset_start(offset),
                None => handle,
            };
            let handle = match descriptor.binding.offset_end {
                Some(offset) => handle.offset_end(offset),
                None => handle,
            };

            let offset = handle.offset();
            let resource = stream.memory_management.storage().get(&handle);

            let size: usize = descriptor.shape.iter().product();
            let size_bytes = size * descriptor.elem_size;

            let buffer = resource.as_ref();
            let protocol_obj: &ProtocolObject<dyn MTLBuffer> = buffer.as_ref();
            let base_ptr = protocol_obj.contents().as_ptr() as *mut u8;
            let write_ptr = unsafe { base_ptr.add(offset as usize) };

            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), write_ptr, size_bytes);
            }
        }

        Ok(())
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: Bindings,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), LaunchError> {
        use objc2_metal::{
            MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
            MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions,
        };

        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        let compiled = self
            .context
            .compile_kernel(&kernel_id, kernel, mode, self.utilities.logger.clone())
            .map_err(LaunchError::CompilationError)?;

        let (grid_x, grid_y, grid_z) = match count {
            CubeCount::Static(x, y, z) => (x, y, z),
            CubeCount::Dynamic(_binding) => {
                return Err(LaunchError::Unknown {
                    reason: "Dynamic dispatch not yet implemented".to_string(),
                    backtrace: cubecl_common::backtrace::BackTrace::capture(),
                });
            }
        };

        let mut resolved = self.streams.resolve(stream_id, bindings.buffers.iter());
        let stream = resolved.current();

        // Collect buffer resources with their offsets for binding
        let mut resources = Vec::with_capacity(bindings.buffers.len());
        for binding in bindings.buffers.iter() {
            let handle = stream
                .memory_management
                .get(binding.memory.clone())
                .expect("Handle should exist");

            let handle = match binding.offset_start {
                Some(offset) => handle.offset_start(offset),
                None => handle,
            };
            let handle = match binding.offset_end {
                Some(offset) => handle.offset_end(offset),
                None => handle,
            };

            let offset = handle.offset();
            let resource = stream.memory_management.storage().get(&handle);

            resources.push((resource, offset));
        }

        let command_buffer =
            (*stream.queue)
                .commandBuffer()
                .ok_or_else(|| LaunchError::Unknown {
                    reason: "Failed to create command buffer".to_string(),
                    backtrace: cubecl_common::backtrace::BackTrace::capture(),
                })?;

        let encoder =
            (*command_buffer)
                .computeCommandEncoder()
                .ok_or_else(|| LaunchError::Unknown {
                    reason: "Failed to create compute command encoder".to_string(),
                    backtrace: cubecl_common::backtrace::BackTrace::capture(),
                })?;

        (*encoder).setComputePipelineState(&compiled.pipeline);

        for (index, (resource, offset)) in resources.iter().enumerate() {
            let buffer: &ProtocolObject<dyn MTLBuffer> = resource.as_ref().as_ref();
            (*encoder).setBuffer_offset_atIndex(Some(buffer), *offset as usize, index);
        }

        // Temporary buffers for metadata/scalars - must stay alive until GPU execution completes
        let mut temporaries: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> = Vec::new();
        let mut buffer_index = resources.len();

        if !bindings.metadata.data.is_empty() {
            use std::ptr::NonNull;
            let metadata_bytes: &[u8] = bytemuck::cast_slice(&bindings.metadata.data);
            let metadata_buffer = unsafe {
                (*stream.device).newBufferWithBytes_length_options(
                    NonNull::new(metadata_bytes.as_ptr() as *mut _).unwrap(),
                    metadata_bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or_else(|| LaunchError::Unknown {
                reason: "Failed to create metadata buffer".to_string(),
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
            })?;
            (*encoder).setBuffer_offset_atIndex(Some(&metadata_buffer), 0, buffer_index);
            temporaries.push(metadata_buffer);
            buffer_index += 1;
        }

        for scalar_binding in bindings.scalars.values() {
            use std::ptr::NonNull;
            let scalar_bytes = scalar_binding.data();
            let scalar_buffer = unsafe {
                (*stream.device).newBufferWithBytes_length_options(
                    NonNull::new(scalar_bytes.as_ptr() as *mut _).unwrap(),
                    scalar_bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or_else(|| LaunchError::Unknown {
                reason: "Failed to create scalar buffer".to_string(),
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
            })?;
            (*encoder).setBuffer_offset_atIndex(Some(&scalar_buffer), 0, buffer_index);
            temporaries.push(scalar_buffer);
            buffer_index += 1;
        }

        let cube_dim = compiled.cube_dim;
        let threads_per_threadgroup = objc2_metal::MTLSize {
            width: cube_dim.x as usize,
            height: cube_dim.y as usize,
            depth: cube_dim.z as usize,
        };
        let threadgroups = objc2_metal::MTLSize {
            width: grid_x as usize,
            height: grid_y as usize,
            depth: grid_z as usize,
        };

        (*encoder)
            .dispatchThreadgroups_threadsPerThreadgroup(threadgroups, threads_per_threadgroup);
        (*encoder).endEncoding();

        // Store command buffer with its temporaries for later flush
        stream.pending_buffers.push(PendingCommandBuffer {
            command_buffer,
            temporaries,
        });

        // Auto-flush when buffer limit reached to provide backpressure
        const MAX_PENDING_BUFFERS: usize = 1;
        if stream.pending_buffers.len() >= MAX_PENDING_BUFFERS {
            let pending_buffers: Vec<_> = stream.pending_buffers.drain(..).collect();
            for pending in pending_buffers.iter() {
                (*pending.command_buffer).commit();
            }

            let fence = (*stream.queue)
                .commandBuffer()
                .expect("Failed to create fence buffer");
            (*fence).commit();
            (*fence).waitUntilCompleted();
            drop(pending_buffers);
        }

        Ok(())
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ExecutionError>> {
        use cubecl_runtime::stream::EventStreamBackend;

        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        let fence = MetalStreamBackend::flush(resolved.current());

        Box::pin(async move { MetalStreamBackend::wait_event_sync(fence) })
    }

    fn flush(&mut self, _stream_id: StreamId) {
        // No-op: flushing is handled by commit in launch/read/write
    }

    fn start_profile(&mut self, _stream_id: StreamId) -> ProfilingToken {
        ProfilingToken { id: 0 }
    }

    fn end_profile(
        &mut self,
        _stream_id: StreamId,
        _token: ProfilingToken,
    ) -> Result<cubecl_common::profile::ProfileDuration, ProfileError> {
        // Profiling not yet implemented
        Err(ProfileError::NotRegistered {
            backtrace: cubecl_common::backtrace::BackTrace::capture(),
        })
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> BindingResource<crate::memory::MetalBufferHandle> {
        let mut resolved = self.streams.resolve(stream_id, std::iter::once(&binding));
        let stream = resolved.current();

        let resource = stream
            .memory_management
            .get_resource(
                binding.memory.clone(),
                binding.offset_start,
                binding.offset_end,
            )
            .expect("Resource should exist");

        BindingResource::new(binding, resource)
    }

    fn memory_usage(
        &mut self,
        stream_id: StreamId,
    ) -> cubecl_runtime::memory_management::MemoryUsage {
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        let stream = resolved.current();
        stream.memory_management.memory_usage()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        let stream = resolved.current();
        stream.memory_management.cleanup(true);
    }

    fn allocation_mode(
        &mut self,
        mode: cubecl_runtime::memory_management::MemoryAllocationMode,
        stream_id: StreamId,
    ) {
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        let stream = resolved.current();
        stream.memory_management.mode(mode);
    }
}
