use crate::{
    MetalCompiler,
    compute::context::MetalContext,
    compute::stream::MetalStreamBackend,
    memory::{MetalBufferHandle, MetalStorage},
};
use cubecl_common::{
    bytes::Bytes,
    profile::{Duration, Instant, ProfileDuration, ProfileTicks},
};
use cubecl_core::{
    MemoryConfiguration,
    prelude::*,
    server::{
        BufferBinding, CopyDescriptor, IoError, KernelArguments, KernelResource, ProfileError,
        ProfilingToken, ServerCommunication, ServerError, ServerUtilities,
    },
};
use cubecl_environment::future::DynFut;
use cubecl_environment::stream::StreamId;
use cubecl_runtime::{
    allocator::ContiguousMemoryLayoutPolicy,
    compiler::CubeTask,
    dry_run::LaunchMode,
    logging::ServerLogger,
    memory_management::{InstallMemoryPoolsError, ManagedMemoryHandle},
    server::ComputeServer,
    storage::{ComputeStorage, ManagedResource},
    stream::{
        EventStreamBackend, ExecuteScope, FailureStore, MultiStream, ResolvedStreams, WriteScoped,
        failed_writing,
    },
    timestamp_profiler::TimestampProfiler,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLDevice};
use std::sync::Arc;

enum DispatchInfo {
    Static(u32, u32, u32),
    Dynamic(BufferBinding),
}

/// Metal compute server.
#[derive(Debug)]
pub struct MetalServer {
    context: MetalContext,
    streams: MultiStream<MetalStreamBackend>,
    pub(crate) utilities: Arc<ServerUtilities<Self>>,
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

        let mut compilation_options = cubecl_cpp::shared::CompilationOptions::default();
        // Metal honors per-op fast math via the `fast::` namespace (MSL 3+).
        compilation_options.supports_features.fast_math = true;
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
    binding: &BufferBinding,
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
    let resource = stream.memory_management.storage().get(&storage_handle)?;

    Ok((resource, offset))
}

/// Reads a pitched row-major buffer into packed bytes. `can_read_tensor` keeps the pitch at
/// the row level, so the copy collapses to 2D: a contiguous inner row, `pitch` bytes apart.
fn read_pitched(src: *const u8, shape: &[usize], strides: &[usize], elem_size: usize) -> Vec<u8> {
    let rank = shape.len();
    let total = shape.iter().product::<usize>() * elem_size;
    if rank <= 1 {
        return unsafe { std::slice::from_raw_parts(src, total) }.to_vec();
    }

    let width = shape[rank - 1] * elem_size;
    let rows = shape[..rank - 1].iter().product::<usize>();
    let pitch = strides[rank - 2] * elem_size;

    let mut out = vec![0u8; rows * width];
    for row in 0..rows {
        let src_row = unsafe { std::slice::from_raw_parts(src.add(row * pitch), width) };
        out[row * width..(row + 1) * width].copy_from_slice(src_row);
    }
    out
}

/// Writes packed bytes into a pitched row-major buffer — the inverse of [`read_pitched`].
fn write_pitched(dst: *mut u8, data: &[u8], shape: &[usize], strides: &[usize], elem_size: usize) {
    let rank = shape.len();
    if rank <= 1 {
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len()) };
        return;
    }

    let width = shape[rank - 1] * elem_size;
    let rows = shape[..rank - 1].iter().product::<usize>();
    let pitch = strides[rank - 2] * elem_size;

    for row in 0..rows {
        let src_row = &data[row * width..(row + 1) * width];
        unsafe { std::ptr::copy_nonoverlapping(src_row.as_ptr(), dst.add(row * pitch), width) };
    }
}

impl MetalServer {
    /// Mark every open profile invalid: a failure inside a profiling window
    /// invalidates the measurement. A no-op with no profile open.
    fn profile_failure(&mut self, error: &ServerError) {
        self.timestamps.failure(error);
    }
}

impl ServerCommunication for MetalServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl WriteScoped for MetalServer {
    type Streams = MultiStream<MetalStreamBackend>;

    fn write_streams(&mut self) -> &mut Self::Streams {
        &mut self.streams
    }

    fn on_failure(&mut self, _stream: StreamId, error: &ServerError) {
        // Measured per device on this backend, so the scope's stream does not
        // narrow which measurement a failure invalidates.
        self.profile_failure(error);
    }
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
            backtrace: cubecl_environment::backtrace::BackTrace::capture(),
        }
        .into())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        let cursor = resolved.cursor;
        let (stream, failures) = resolved.current_and_failures();
        let reserved = stream
            .memory_management
            .reserve(size, failures)
            .expect("Failed to reserve memory");
        stream
            .memory_management
            .bind(reserved, memory, cursor, failures)
            .expect("Failed to bind memory");
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        use objc2_metal::MTLBuffer;

        // Buffers another stream wrote are only as good as the work that wrote
        // them; see `FailureStore::ensure_written`.
        if let Err(err) = self
            .streams
            .ensure_written(descriptors.iter().map(|d| &d.handle))
        {
            return Box::pin(async move { Err(err) });
        }

        let mut resolved = self
            .streams
            .resolve(stream_id, descriptors.iter().map(|d| &d.handle));

        // Flush, wait, then read.
        let (stream, failures) = resolved.current_and_failures();
        let event = MetalStreamBackend::flush(stream, failures);

        if let Err(e) = MetalStreamBackend::wait_event_sync(event) {
            return Box::pin(async move { Err(e) });
        }

        let results: Result<Vec<_>, ServerError> = descriptors
            .iter()
            .map(|descriptor| {
                let (resource, offset) = resolve_origin_resource(&mut resolved, &descriptor.handle)
                    .map_err(ServerError::from)?;

                let buffer = resource.inner();
                let protocol_obj: &ProtocolObject<dyn MTLBuffer> = buffer.as_ref();
                let base_ptr = protocol_obj.contents().as_ptr() as *const u8;
                let src_ptr = unsafe { base_ptr.add(offset as usize) };
                let bytes = read_pitched(
                    src_ptr,
                    &descriptor.shape,
                    &descriptor.strides,
                    descriptor.elem_size,
                );

                Ok(Bytes::from_bytes_vec(bytes))
            })
            .collect();

        Box::pin(async move { results })
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        use objc2_metal::MTLBuffer;

        let mut resolved = self
            .streams
            .resolve(stream_id, descriptors.iter().map(|(d, _)| &d.handle));

        let (stream, failures) = resolved.current_and_failures();
        let event = MetalStreamBackend::flush(stream, failures);
        if let Err(err) = MetalStreamBackend::wait_event_sync(event) {
            core::mem::drop(resolved);
            // Nothing is copied, so every destination this call was given is
            // left as it was — taint them, or a read of one on another
            // logical stream finds no failure to fail on and copies stale
            // bytes.
            let mut written = self.write_set();
            written.extend(descriptors.iter().map(|(desc, _)| desc.handle.clone()));
            failed_writing(self, stream_id, written, err);
            return;
        }
        core::mem::drop(resolved);

        for (descriptor, data) in descriptors {
            // Each copy runs in its own scope over its destination: the copy
            // fills it, which is what releases an earlier failure's hold on
            // it — a caller recovers by writing from the host as much as by
            // relaunching — and a failure leaves it as it was, which is what
            // a later read of it has to fail on.
            let mut written = self.write_set();
            written.push(descriptor.handle.clone());
            ExecuteScope::over(self, stream_id, written).execute(|server| {
                let mut resolved = server
                    .streams
                    .resolve(stream_id, [&descriptor.handle].into_iter());
                let (resource, offset) = resolve_origin_resource(&mut resolved, &descriptor.handle)
                    .map_err(ServerError::Io)?;

                let buffer = resource.inner();
                let protocol_obj: &ProtocolObject<dyn MTLBuffer> = buffer.as_ref();
                let base_ptr = protocol_obj.contents().as_ptr() as *mut u8;
                let dst_ptr = unsafe { base_ptr.add(offset as usize) };

                write_pitched(
                    dst_ptr,
                    &data,
                    &descriptor.shape,
                    &descriptor.strides,
                    descriptor.elem_size,
                );
                Ok(())
            });
        }
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        stream_id: StreamId,
        launch_mode: LaunchMode,
    ) {
        use objc2_metal::{MTLBuffer, MTLComputeCommandEncoder, MTLDevice, MTLResourceOptions};

        // Compilation comes first — memoized, so a launch after the first
        // pays a map lookup — because the write scope stages what the
        // compiled kernel says it writes. A kernel that fails to compile has
        // no IR and no compiled answer, so the caller's declared IO decides:
        // only the declared outputs are left carrying the failure, never the
        // buffers the kernel was only going to read — tainting those would
        // refuse every later launch that shares them, an autotune sweep
        // above all.
        //
        // A dry run stages none either way. It was never going to write, so a
        // failure in it leaves nothing stale, and tainting its buffers would
        // fail unrelated reads of memory the run deliberately left alone.
        let kernel_id = kernel.id();
        let compiled = (|| {
            cubecl_runtime::validation::validate_cube_dim(&self.utilities.properties, &kernel_id)?;
            cubecl_runtime::validation::validate_units(&self.utilities.properties, &kernel_id)?;
            self.context.compile_kernel(
                &kernel_id,
                kernel,
                self.utilities.properties.hardware.max_shared_memory_size,
                self.utilities.logger.clone(),
            )
        })();
        let compiled = match compiled {
            Ok(compiled) => compiled,
            Err(err) => {
                if !launch_mode.is_skipped() {
                    let mut written = self.write_set();
                    written.extend(bindings.buffers_written(None).cloned());
                    failed_writing(self, stream_id, written, ServerError::Launch(err));
                } else {
                    self.profile_failure(&ServerError::Launch(err));
                }
                return;
            }
        };
        if launch_mode.is_skipped() {
            return;
        }
        let io = compiled.io.clone();

        // The scope claims what the launch writes until the body proves the
        // work enqueued, so a failure — or a panic — anywhere in it leaves a
        // read of those buffers failing on the error rather than copying
        // bytes nothing wrote. An input that already carries a failure skips
        // the launch instead, and the scope settles that too.
        let mut written = self.write_set();
        written.extend(bindings.buffers_written(io.as_deref()).cloned());
        // A dynamic count travels outside `resources`, so `buffers_read`
        // never names it — yet the dispatch reads it as its grid dimensions,
        // which is exactly the garbage-as-cube-count read the skip exists to
        // prevent.
        let count_read = match &count {
            CubeCount::Dynamic(binding) => Some(binding),
            CubeCount::Static(..) => None,
        };
        ExecuteScope::launching(
            self,
            kernel_id,
            stream_id,
            bindings.buffers_read(io.as_deref()).chain(count_read),
            written,
        )
        .execute(|server| {
            let dispatch_info = match count {
                CubeCount::Static(x, y, z) => DispatchInfo::Static(x, y, z),
                CubeCount::Dynamic(binding) => DispatchInfo::Dynamic(binding),
            };

            // Resolve every binding (including the dynamic count) so the current stream
            // waits on each binding's origin stream before dispatching.
            let mut resolved = server.streams.resolve(
                stream_id,
                bindings
                    .resources
                    .iter()
                    .map(|res| match res {
                        KernelResource::Buffer(binding) => binding,
                        KernelResource::TensorMap(_) => {
                            panic!("Tensor maps not supported on Metal")
                        }
                    })
                    .chain(match &dispatch_info {
                        DispatchInfo::Dynamic(binding) => Some(binding),
                        DispatchInfo::Static(..) => None,
                    }),
            );

            let mut resources = Vec::with_capacity(bindings.resources.len());
            let mut total_buffer_bytes: usize = 0;
            for binding in bindings.resources.iter() {
                let binding = match binding {
                    KernelResource::Buffer(binding) => binding,
                    KernelResource::TensorMap(_) => {
                        panic!("Tensor maps not supported on Metal")
                    }
                };
                let (resource, offset) = resolve_origin_resource(&mut resolved, binding)
                    .map_err(ServerError::Io)?;

                total_buffer_bytes += binding.size_in_used() as usize;

                resources.push((resource, offset));
            }

            // The indirect count buffer is read GPU-side, so it too comes from its origin stream.
            let indirect_buffer_info = match &dispatch_info {
                DispatchInfo::Dynamic(binding) => Some(
                    resolve_origin_resource(&mut resolved, binding).map_err(ServerError::Io)?,
                ),
                _ => None,
            };

            // Work is issued on the current stream; resources above came from their origins.
            let (stream, failures) = resolved.current_and_failures();

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
                        None => {
                            return Err(ServerError::Generic {
                                reason: format!(
                                    "failed to allocate a {} B Metal buffer for the kernel's \
                                     metadata",
                                    info_bytes.len()
                                ),
                                backtrace: cubecl_environment::backtrace::BackTrace::capture(),
                            });
                        }
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
                MetalStreamBackend::flush(stream, failures);
            }

            Ok(())
        });
    }

    fn sync(
        &mut self,
        handles: Vec<BufferBinding>,
        stream_id: StreamId,
    ) -> DynFut<Result<(), ServerError>> {
        // The claim check a read would have made, without the read; claims
        // are set at enqueue time, so they are already in place.
        if let Err(err) = self.streams.ensure_written(handles.iter()) {
            return Box::pin(async move { Err(err) });
        }
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        let (stream, failures) = resolved.current_and_failures();
        let fence = MetalStreamBackend::flush(stream, failures);

        Box::pin(async move { MetalStreamBackend::wait_event_sync(fence) })
    }

    fn check(
        &mut self,
        handles: Vec<BufferBinding>,
        _stream_id: StreamId,
    ) -> Result<(), ServerError> {
        self.streams.ensure_written(handles.iter())
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        // A flush reports nothing: a failure lives on the buffers the work
        // left unwritten, and a read of one of them is what surfaces it.
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        let (stream, failures) = resolved.current_and_failures();
        MetalStreamBackend::flush(stream, failures);
        Ok(())
    }

    fn start_profile(&mut self, stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        // Drain prior work so the window only contains command buffers committed from here on.
        if let Err(err) = cubecl_environment::future::block_on(self.sync(Vec::new(), stream_id)) {
            log::warn!("{err}");
        }
        // Begin collecting this window's work-bearing command buffers on the stream.
        self.streams
            .resolve(stream_id, std::iter::empty())
            .current()
            .profiling = Some(Vec::new());
        Ok(self.timestamps.start())
    }

    fn end_profile(
        &mut self,
        stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        // Flush the final encoder and wait for GPU completion so timestamps are valid.
        if let Err(err) = cubecl_environment::future::block_on(self.sync(Vec::new(), stream_id)) {
            // Drop any collected buffers so a retry can't accumulate stale work.
            self.streams
                .resolve(stream_id, std::iter::empty())
                .current()
                .profiling = None;
            self.timestamps.failure(&err);
        }

        // The collector disarms whatever the token says below: left armed, it
        // would retain every later flush's command buffer until the next
        // start_profile — and a failed profile is exactly when the token
        // errors, since a failure anywhere in the window lands in it.
        let buffers = self
            .streams
            .resolve(stream_id, std::iter::empty())
            .current()
            .profiling
            .take()
            .unwrap_or_default();

        // Clear the token (propagates any recorded profiling error). Its system-time result
        // is discarded in favor of GPU timestamps below.
        self.timestamps.stop(token)?;

        // `sync()` waits on the stream's shared event, which can be signaled slightly before
        // a command buffer reaches `Completed` status. `GPUStartTime`/`GPUEndTime` are only
        // valid once completed, so wait explicitly before reading them.
        for buffer in &buffers {
            buffer.waitUntilCompleted();
        }

        // GPU wall-time of the window: latest end minus earliest start across the
        // work-bearing command buffers. Empty window => zero, still device-timed.
        let mut start_s = f64::INFINITY;
        let mut end_s = f64::NEG_INFINITY;
        for buffer in &buffers {
            start_s = start_s.min(buffer.GPUStartTime());
            end_s = end_s.max(buffer.GPUEndTime());
        }

        let span = if buffers.is_empty() {
            Duration::ZERO
        } else if !start_s.is_finite() || !end_s.is_finite() {
            log::warn!(
                "Metal device profiling read non-finite GPU timestamps (start={start_s}, end={end_s}); reporting zero"
            );
            Duration::ZERO
        } else {
            Duration::from_secs_f64((end_s - start_s).max(0.0))
        };

        let base = Instant::now();
        let ticks = ProfileTicks::from_start_end(base, base + span);
        Ok(ProfileDuration::new_device_time(async move { ticks }))
    }

    fn get_resource(
        &mut self,
        binding: BufferBinding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<<MetalStorage as ComputeStorage>::Resource>, ServerError> {
        // The same claim check a read makes: a buffer a failed launch never
        // filled reports the failure rather than handing back a pointer to
        // whatever was there before.
        self.streams.ensure_written([&binding].into_iter())?;
        let mut resolved = self.streams.resolve(stream_id, std::iter::once(&binding));
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
    ) -> cubecl_runtime::memory_management::MemoryUsage {
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        resolved.current().memory_management.memory_usage()
    }

    fn memory_report(
        &mut self,
        stream_id: StreamId,
    ) -> cubecl_runtime::memory_management::MemoryReport {
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        resolved.current().memory_management.memory_report()
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        let (stream, failures) = resolved.current_and_failures();
        stream.memory_management.cleanup(true, failures);
    }

    fn allocation_mode(
        &mut self,
        mode: cubecl_runtime::memory_management::MemoryAllocationMode,
        stream_id: StreamId,
    ) {
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        resolved.current().memory_management.mode(mode);
    }

    fn install_memory_pools(
        &mut self,
        config: MemoryConfiguration,
        stream_id: StreamId,
    ) -> Result<(), InstallMemoryPoolsError> {
        // Streams created from now on build their GPU pools with the new
        // layout; memory is per stream, so already-created streams keep theirs.
        self.streams.backend_mut().set_gpu_pools(config.clone());
        let (_, props) = self.streams.backend_mut().gpu_pools();

        // The calling stream's pools are rebuilt in place, keeping the old
        // layout when something is still live in them.
        let mut resolved = self.streams.resolve(stream_id, std::iter::empty());
        let (stream, failures) = resolved.current_and_failures();
        stream
            .memory_management
            .install_pools(config, &props, failures)
    }
}

#[cfg(test)]
mod pitched_tests {
    use super::{read_pitched, write_pitched};

    #[test]
    fn pitched_round_trip() {
        // A [2, 3] tensor with a row pitch of 4 (one element of padding per row).
        let shape = [2usize, 3];
        let strides = [4usize, 1];
        let packed = [10u8, 20, 30, 40, 50, 60];

        let mut buffer = vec![0u8; 2 * 4];
        write_pitched(buffer.as_mut_ptr(), &packed, &shape, &strides, 1);
        // Rows land at offsets 0 and 4; the padding bytes (3, 7) stay zero.
        assert_eq!(buffer, [10, 20, 30, 0, 40, 50, 60, 0]);

        let read_back = read_pitched(buffer.as_ptr(), &shape, &strides, 1);
        assert_eq!(read_back, packed);
    }
}
