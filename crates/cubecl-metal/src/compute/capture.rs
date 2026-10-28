//! Metal's half of graph capture: a window's dispatches are recorded instead
//! of encoded, encoded once into an indirect command buffer when it closes,
//! and replayed with a single call.
//!
//! Everything around that — arming the pools, draining the drop queue,
//! pinning what the window touched, claiming what a refused recording would
//! have written — is the shared
//! [`Window`](cubecl_server::command::Window)'s.

use crate::compute::driver::{Metal, MetalBindings};
use crate::compute::pipelines::MetalCompiledKernel;
use crate::compute::stream::MetalStream;
use cubecl_server::command::GraphDriver;
use cubecl_server::server::ServerError;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice,
    MTLIndirectCommandBuffer, MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType,
    MTLIndirectComputeCommand, MTLResource, MTLResourceOptions, MTLResourceUsage, MTLSize,
};
use std::ptr::NonNull;

/// Where each recorded dispatch's metadata starts in the graph's metadata
/// buffer. Metal binds a buffer at an offset aligned this far whatever the
/// address space the kernel reads it from.
const INFO_ALIGNMENT: usize = 256;

/// A dispatch a capture window recorded instead of encoding.
pub struct RecordedDispatch {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    threadgroups: MTLSize,
    threads_per_threadgroup: MTLSize,
    bindings: MetalBindings,
}

impl RecordedDispatch {
    /// The dispatch of `kernel` over `count` threadgroups, given `bindings`.
    pub fn new(
        kernel: &MetalCompiledKernel,
        count: (u32, u32, u32),
        bindings: MetalBindings,
    ) -> Self {
        Self {
            pipeline: kernel.pipeline.clone(),
            threadgroups: MTLSize {
                width: count.0 as usize,
                height: count.1 as usize,
                depth: count.2 as usize,
            },
            threads_per_threadgroup: kernel.threads_per_threadgroup(),
            bindings,
        }
    }

    /// The buffer slots the dispatch binds: its buffers, then its metadata.
    fn bind_count(&self) -> usize {
        self.bindings.buffers.len() + usize::from(!self.bindings.info.is_empty())
    }
}

/// An instantiated graph: the indirect command buffer its recorded dispatches
/// were encoded into, and what replaying it needs.
pub struct Executable {
    /// One command per recorded dispatch, in order; `None` for a window that
    /// recorded none, whose replay does nothing.
    commands: Option<Retained<ProtocolObject<dyn MTLIndirectCommandBuffer>>>,
    dispatches: usize,
    /// The bytes the dispatches bind, counted toward the batch a replay
    /// lands in as a launch's are.
    bytes: usize,
    /// Every buffer the commands bind, each once. An encoder that executes an
    /// indirect command buffer sees none of the buffers in it, so a replay
    /// declares each one resident first; holding them also keeps every
    /// buffer the commands point at alive for as long as the graph can
    /// replay.
    resident: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// The pipelines the commands run, alive for as long as they can run.
    _pipelines: Vec<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
}

impl Executable {
    /// Encode `recording` into an indirect command buffer on `device`.
    ///
    /// Each command sets its own pipeline and buffers, and waits on the one
    /// before it: a kernel reads what the previous one wrote, as the stream
    /// it was recorded on would have had it.
    ///
    /// # Errors
    ///
    /// The device's refusal of the command buffer or of the buffer holding
    /// the dispatches' metadata.
    fn encode(
        device: &ProtocolObject<dyn MTLDevice>,
        recording: Vec<RecordedDispatch>,
    ) -> Result<Self, ServerError> {
        let dispatches = recording.len();
        let mut executable = Self {
            commands: None,
            dispatches,
            bytes: 0,
            resident: Vec::new(),
            _pipelines: Vec::new(),
        };
        if recording.is_empty() {
            return Ok(executable);
        }

        let info = Self::info_buffer(device, &recording)?;
        let descriptor = MTLIndirectCommandBufferDescriptor::new();
        descriptor.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
        descriptor.setInheritPipelineState(false);
        descriptor.setInheritBuffers(false);
        descriptor.setMaxKernelBufferBindCount(
            recording
                .iter()
                .map(RecordedDispatch::bind_count)
                .max()
                .unwrap_or(0),
        );
        // SAFETY: the descriptor describes compute commands only, and the
        // count is the number of commands encoded below.
        let commands = unsafe {
            device.newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &descriptor,
                dispatches,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or_else(|| {
            ServerError::graph_state(format!(
                "Metal refused an indirect command buffer of {dispatches} dispatches"
            ))
        })?;

        let mut info_offset = 0;
        for (index, dispatch) in recording.iter().enumerate() {
            // SAFETY: `index` is below the command count the buffer was made
            // with.
            let command = unsafe { commands.indirectComputeCommandAtIndex(index) };
            command.setComputePipelineState(&dispatch.pipeline);
            for (slot, resource) in dispatch.bindings.buffers.iter().enumerate() {
                // SAFETY: the resource is a live slice of its buffer, which
                // the executable retains below.
                unsafe {
                    command.setKernelBuffer_offset_atIndex(
                        resource.buffer(),
                        resource.offset(),
                        slot,
                    )
                };
                executable.bytes += resource.size();
                executable.resident.push(resource.retained());
            }
            if let Some(info) = &info
                && !dispatch.bindings.info.is_empty()
            {
                // SAFETY: the offset is where `info_buffer` wrote this
                // dispatch's metadata, which the executable retains below.
                unsafe {
                    command.setKernelBuffer_offset_atIndex(
                        info,
                        info_offset,
                        dispatch.bindings.buffers.len(),
                    )
                };
                info_offset += size_of_val(dispatch.bindings.info.as_slice())
                    .next_multiple_of(INFO_ALIGNMENT);
            }
            command.concurrentDispatchThreadgroups_threadsPerThreadgroup(
                dispatch.threadgroups,
                dispatch.threads_per_threadgroup,
            );
            command.setBarrier();
        }

        executable.resident.extend(info);
        executable
            .resident
            .sort_unstable_by_key(|buffer| Retained::as_ptr(buffer));
        executable
            .resident
            .dedup_by_key(|buffer| Retained::as_ptr(buffer));
        executable._pipelines = recording
            .into_iter()
            .map(|dispatch| dispatch.pipeline)
            .collect();
        executable
            ._pipelines
            .sort_unstable_by_key(|pipeline| Retained::as_ptr(pipeline));
        executable
            ._pipelines
            .dedup_by_key(|pipeline| Retained::as_ptr(pipeline));
        executable.commands = Some(commands);
        Ok(executable)
    }

    /// One buffer holding every dispatch's metadata in recording order, each
    /// at an [`INFO_ALIGNMENT`] boundary; `None` when no dispatch has any.
    ///
    /// The metadata is the graph's own: a launch hands it to the encoder by
    /// value, and an indirect command can only bind a buffer.
    fn info_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        recording: &[RecordedDispatch],
    ) -> Result<Option<Retained<ProtocolObject<dyn MTLBuffer>>>, ServerError> {
        let size: usize = recording
            .iter()
            .map(|dispatch| {
                size_of_val(dispatch.bindings.info.as_slice()).next_multiple_of(INFO_ALIGNMENT)
            })
            .sum();
        if size == 0 {
            return Ok(None);
        }
        let buffer = device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| {
                ServerError::graph_state(format!(
                    "Metal refused a {size} B buffer for the graph's metadata"
                ))
            })?;
        let base = buffer.contents().as_ptr() as *mut u8;
        let mut offset = 0;
        for dispatch in recording {
            let info: &[u8] = bytemuck::cast_slice(&dispatch.bindings.info);
            // SAFETY: the buffer is fresh, so nothing else reads or writes
            // it, and it was sized for every dispatch's aligned metadata.
            unsafe { std::ptr::copy_nonoverlapping(info.as_ptr(), base.add(offset), info.len()) };
            offset += info.len().next_multiple_of(INFO_ALIGNMENT);
        }
        Ok(Some(buffer))
    }
}

impl GraphDriver for Metal {
    type Executable = Executable;

    fn begin(_stream: &mut MetalStream) -> Result<(), ServerError> {
        // Nothing to open on the device: a stream whose capture is recording
        // records its dispatches rather than encoding them.
        Ok(())
    }

    fn instantiate(
        stream: &mut MetalStream,
        doomed: Option<ServerError>,
    ) -> Result<Executable, ServerError> {
        let recording = core::mem::take(&mut stream.recording);
        if let Some(doomed) = doomed {
            return Err(doomed);
        }
        let device = stream.queue().device();
        Executable::encode(&device, recording)
    }

    fn upload(_exec: &Executable, _stream: &mut MetalStream) {
        // Nothing to stage: the commands were encoded into shared memory.
    }

    fn replay(exec: &Executable, stream: &mut MetalStream) -> Result<(), ServerError> {
        stream.reject_while_recording("replay")?;
        let Some(commands) = &exec.commands else {
            return Ok(());
        };
        let resident: Vec<NonNull<ProtocolObject<dyn MTLResource>>> = exec
            .resident
            .iter()
            .map(|buffer| NonNull::from(ProtocolObject::<dyn MTLResource>::from_ref(&**buffer)))
            .collect();

        let mut queue = stream.queue();
        let encoder = queue.batch().compute();
        // SAFETY: every pointer is a buffer the executable retains, and the
        // count is the array's length.
        unsafe {
            encoder.useResources_count_usage(
                NonNull::new(resident.as_ptr() as *mut _).expect("a vector is never null"),
                resident.len(),
                MTLResourceUsage::Read | MTLResourceUsage::Write,
            );
            encoder.executeCommandsInBuffer_withRange(commands, NSRange::new(0, exec.dispatches));
        }
        queue.encoded(exec.dispatches, exec.bytes);
        Ok(())
    }
}
