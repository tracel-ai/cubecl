use crate::{
    MetalCompiler, compute::context::MetalContext, compute::stream::MetalStreamBackend,
    memory::MetalStorage,
};
use cubecl_common::{bytes::Bytes, stream_id::StreamId};
use cubecl_core::{
    MemoryConfiguration,
    future::DynFut,
    prelude::*,
    server::{
        Allocation, AllocationDescriptor, Binding, Bindings, CopyDescriptor, ExecutionError,
        Handle, IoError, LaunchError, ProfileError, ProfilingToken, ServerCommunication,
        ServerUtilities,
    },
};
use cubecl_runtime::{
    compiler::CubeTask,
    logging::ServerLogger,
    server::ComputeServer,
    storage::{BindingResource, ComputeStorage},
    stream::{EventStreamBackend, GcTask, MultiStream},
    timestamp_profiler::TimestampProfiler,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;
use std::sync::Arc;

/// Dispatch information for kernel launches.
enum DispatchInfo {
    Static(u32, u32, u32),
    Dynamic(Binding),
}

/// Metal compute server
#[derive(Debug)]
pub struct MetalServer {
    context: MetalContext,
    streams: MultiStream<MetalStreamBackend>,
    utilities: Arc<ServerUtilities<Self>>,
    timestamps: TimestampProfiler,
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

        let config = cubecl_runtime::config::GlobalConfig::get();
        let max_streams = config.streaming.max_streams;

        Self {
            context,
            streams: MultiStream::new(logger, backend, max_streams),
            utilities,
            timestamps: TimestampProfiler::default(),
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
        // Shared storage allows direct CPU-GPU buffer access
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
            let size: usize = descriptor.shape.iter().product();
            let size_bytes = size * descriptor.elem_size;

            let mut strides = vec![1; descriptor.shape.len()];
            for i in (0..descriptor.shape.len().saturating_sub(1)).rev() {
                strides[i] = strides[i + 1] * descriptor.shape[i + 1];
            }

            let slice_handle = stream.memory_management.reserve(size_bytes as u64)?;

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
        use objc2_metal::MTLBuffer;

        let mut resolved = self
            .streams
            .resolve(stream_id, descriptors.iter().map(|d| &d.binding));

        // Flush pending_bindings, wait, then read.
        let stream = resolved.current();
        let pending = std::mem::take(&mut stream.pending_bindings);
        let event = MetalStreamBackend::flush(stream);
        let event_for_gc = event.clone();

        if let Err(e) = MetalStreamBackend::wait_event_sync(event) {
            return Box::pin(async move {
                Err(IoError::Unknown {
                    description: format!("Failed to wait for GPU: {:?}", e),
                    backtrace: cubecl_common::backtrace::BackTrace::capture(),
                })
            });
        }

        let stream = resolved.current();

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

                let buffer = resource.inner();
                let protocol_obj: &ProtocolObject<dyn MTLBuffer> = buffer.as_ref();
                let base_ptr = protocol_obj.contents().as_ptr() as *const u8;
                let contents_ptr = unsafe { base_ptr.add(offset as usize) };
                let data = unsafe { std::slice::from_raw_parts(contents_ptr, size_bytes) };

                Ok(Bytes::from_bytes_vec(data.to_vec()))
            })
            .collect();

        // Register pending and current bindings with GC.
        let mut bindings = pending;
        bindings.extend(descriptors.iter().map(|d| d.binding.memory.clone()));
        resolved.gc(GcTask::new(bindings, event_for_gc));

        Box::pin(async move { results })
    }

    fn write(
        &mut self,
        descriptors: Vec<(CopyDescriptor<'_>, Bytes)>,
        stream_id: StreamId,
    ) -> Result<(), IoError> {
        use objc2_metal::MTLBuffer;

        let mut resolved = self
            .streams
            .resolve(stream_id, descriptors.iter().map(|(d, _)| &d.binding));

        let stream = resolved.current();
        let event = MetalStreamBackend::flush(stream);
        let event_for_gc = event.clone();
        if let Err(err) = MetalStreamBackend::wait_event_sync(event) {
            return Err(IoError::Unknown {
                description: format!("Failed to wait for stream sync: {:?}", err),
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
            });
        }

        let stream = resolved.current();

        let bindings: Vec<_> = descriptors
            .iter()
            .map(|(d, _)| d.binding.memory.clone())
            .collect();

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

            let buffer = resource.inner();
            let protocol_obj: &ProtocolObject<dyn MTLBuffer> = buffer.as_ref();
            let base_ptr = protocol_obj.contents().as_ptr() as *mut u8;
            let write_ptr = unsafe { base_ptr.add(offset as usize) };

            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), write_ptr, size_bytes);
            }
        }

        resolved.gc(GcTask::new(bindings, event_for_gc));

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
        use cubecl_runtime::stream::EventStreamBackend;
        use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions};

        let mut kernel_id = kernel.id();
        kernel_id.mode(mode);

        let compiled = self
            .context
            .compile_kernel(&kernel_id, kernel, mode, self.utilities.logger.clone())
            .map_err(LaunchError::CompilationError)?;

        let dispatch_info = match count {
            CubeCount::Static(x, y, z) => DispatchInfo::Static(x, y, z),
            CubeCount::Dynamic(binding) => DispatchInfo::Dynamic(binding),
        };

        let mut resolved = self.streams.resolve(stream_id, bindings.buffers.iter());

        let stream = resolved.current();
        // Flush pending bindings to GC so memory is freed after GPU completes.
        let pending = std::mem::take(&mut stream.pending_bindings);
        if !pending.is_empty() {
            let event = MetalStreamBackend::flush(stream);
            resolved.gc(GcTask::new(pending, event));
        }

        let stream = resolved.current();

        let mut resources = Vec::with_capacity(bindings.buffers.len());
        let mut total_buffer_bytes: usize = 0;
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

            total_buffer_bytes += handle.size() as usize;

            resources.push((resource, offset));
        }

        // Handle dynamic dispatch buffer lookup before getting encoder
        let indirect_buffer_info = match &dispatch_info {
            DispatchInfo::Dynamic(binding) => {
                let handle = stream
                    .memory_management
                    .get(binding.memory.clone())
                    .expect("Handle should exist");

                let handle = match binding.offset_start {
                    Some(offset) => handle.offset_start(offset),
                    None => handle,
                };

                let offset = handle.offset();
                let resource = stream.memory_management.storage().get(&handle);
                Some((resource, offset))
            }
            _ => None,
        };

        // Get encoder and set up for dispatch
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

        let mut buffer_index = resources.len();

        if !bindings.metadata.data.is_empty() {
            let metadata_bytes: &[u8] = bytemuck::cast_slice(&bindings.metadata.data);
            if metadata_bytes.len() <= 4096 {
                use std::ptr::NonNull;
                unsafe {
                    (*encoder).setBytes_length_atIndex(
                        NonNull::new(metadata_bytes.as_ptr() as *mut _).unwrap(),
                        metadata_bytes.len(),
                        buffer_index,
                    );
                }
            } else {
                use std::ptr::NonNull;
                let metadata_buffer = unsafe {
                    (*device).newBufferWithBytes_length_options(
                        NonNull::new(metadata_bytes.as_ptr() as *mut _).unwrap(),
                        metadata_bytes.len(),
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or_else(|| LaunchError::Unknown {
                    reason: "Failed to create metadata buffer".to_string(),
                    backtrace: cubecl_common::backtrace::BackTrace::capture(),
                })?;
                unsafe {
                    (*encoder).setBuffer_offset_atIndex(Some(&metadata_buffer), 0, buffer_index);
                }
                active.temporaries.push(metadata_buffer);
            }
            buffer_index += 1;
        }

        for scalar_binding in bindings.scalars.values() {
            let scalar_bytes = scalar_binding.data();
            if scalar_bytes.len() <= 4096 {
                use std::ptr::NonNull;
                unsafe {
                    (*encoder).setBytes_length_atIndex(
                        NonNull::new(scalar_bytes.as_ptr() as *mut _).unwrap(),
                        scalar_bytes.len(),
                        buffer_index,
                    );
                }
            } else {
                use std::ptr::NonNull;
                let scalar_buffer = unsafe {
                    (*device).newBufferWithBytes_length_options(
                        NonNull::new(scalar_bytes.as_ptr() as *mut _).unwrap(),
                        scalar_bytes.len(),
                        MTLResourceOptions::StorageModeShared,
                    )
                }
                .ok_or_else(|| LaunchError::Unknown {
                    reason: "Failed to create scalar buffer".to_string(),
                    backtrace: cubecl_common::backtrace::BackTrace::capture(),
                })?;
                unsafe {
                    (*encoder).setBuffer_offset_atIndex(Some(&scalar_buffer), 0, buffer_index);
                }
                active.temporaries.push(scalar_buffer);
            }
            buffer_index += 1;
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

        stream
            .pending_bindings
            .extend(bindings.buffers.iter().map(|b| b.memory.clone()));

        Ok(())
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ExecutionError>> {
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        let fence = MetalStreamBackend::flush(resolved.current());

        Box::pin(async move { MetalStreamBackend::wait_event_sync(fence) })
    }

    fn flush(&mut self, _stream_id: StreamId) {}

    fn start_profile(&mut self, stream_id: StreamId) -> ProfilingToken {
        if let Err(err) = cubecl_common::future::block_on(self.sync(stream_id)) {
            log::warn!("{err}");
        }
        self.timestamps.start()
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<cubecl_common::profile::ProfileDuration, ProfileError> {
        if let Err(err) = cubecl_common::future::block_on(self.sync(stream_id)) {
            self.timestamps.error(err.into());
        }
        self.timestamps.stop(token)
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
