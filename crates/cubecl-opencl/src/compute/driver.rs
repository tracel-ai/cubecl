//! What the shared [`Command`](cubecl_server::command::Command) cannot do itself: `OpenCL`'s device
//! calls, and where an `OpenCL` stream keeps the state they move in step with.
//!
//! Host memory is plain memory, which every `OpenCL` copy takes as it is.

use crate::compute::artifact::OpenClKernel;
use crate::compute::cl::{self, ClError, Event};
use crate::compute::stream::{Marker, OpenClStream, OpenClStreamBackend, Queue};
use crate::memory::{OpenClResource, OpenClStorage};
use cubecl_common::bytes::Bytes;
use cubecl_core::server::{MetadataBindingInfo, ResourceLimitError};
use cubecl_environment::backtrace::BackTrace;
use cubecl_server::command::{CopyLayout, DeviceStream, Driver};
use cubecl_server::id::KernelId;
use cubecl_server::memory_management::drop_queue::PendingDropQueue;
use cubecl_server::memory_management::{ManagedMemoryBinding, MemoryManagement};
use cubecl_server::metadata_cache::MetadataInfoCache;
use cubecl_server::server::{Handle, IoError, LaunchError};
use cubecl_server::storage::{
    BytesResource, BytesStorage, PinnedMemoryAllocController, PinnedMemoryResource,
};
use cubecl_server::stream::StreamCapture;
use opencl_sys::{
    CL_INVALID_WORK_GROUP_SIZE, CL_MEM_COPY_HOST_PTR, CL_MEM_READ_ONLY, cl_context, cl_mem,
};
use std::ffi::c_void;

impl DeviceStream for OpenClStream {
    type Fence = Marker;
    type DeviceStorage = OpenClStorage;
    type HostStorage = BytesStorage;
    type Signal = Queue;

    fn device_memory(&mut self) -> &mut MemoryManagement<OpenClStorage> {
        &mut self.memory_management
    }

    fn host_memory(&mut self) -> &mut MemoryManagement<BytesStorage> {
        &mut self.host_memory
    }

    fn drop_queue(&mut self) -> &mut PendingDropQueue<Marker> {
        &mut self.drop_queue
    }

    fn capturing(&mut self) -> &mut StreamCapture {
        &mut self.capturing
    }

    fn info_cache(&mut self) -> &mut MetadataInfoCache<Handle> {
        &mut self.info_cache
    }

    fn signal(&self) -> Queue {
        self.queue
    }

    fn fence(queue: Queue) -> Marker {
        Marker::on(queue.raw)
    }
}

/// The context kernels are launched in, with what refusing a launch reports.
#[derive(Debug)]
pub(crate) struct OpenClContext {
    pub(crate) context: cl_context,
    pub(crate) max_units: u32,
}

/// What an `OpenCL` launch is given: a pointer per buffer, then the launch's metadata; and, when a
/// profiling window is open on its stream, takes back the event of its run.
pub(crate) struct OpenClLaunch {
    pub(crate) buffers: Vec<OpenClResource>,
    pub(crate) info: MetadataBindingInfo,
    pub(crate) timed: bool,
    pub(crate) event: Option<Event>,
}

/// `OpenCL`, as the shared command sees it.
pub(crate) struct OpenCl;

impl Driver for OpenCl {
    type Backend = OpenClStreamBackend;
    type Stream = OpenClStream;
    type Context = OpenClContext;
    type LaunchArgs = OpenClLaunch;
    type Loaded = OpenClKernel;

    unsafe fn pinned_bytes(
        binding: ManagedMemoryBinding,
        resource: BytesResource,
        size: usize,
    ) -> Bytes {
        // Plain host memory serves as pinned: every copy takes it.
        let (ptr, len) = resource.get_write_ptr_and_length();
        let resource = PinnedMemoryResource { ptr, size: len };
        let controller = Box::new(PinnedMemoryAllocController::init(binding, resource));
        // SAFETY: the caller guarantees `binding` names initialized memory of at least `size` bytes.
        unsafe { Bytes::from_controller(controller, size) }
    }

    unsafe fn copy_to_host(
        resource: &OpenClResource,
        layout: &CopyLayout<'_>,
        bytes: &mut Bytes,
        stream: &OpenClStream,
    ) -> Result<(), IoError> {
        // SAFETY: as the function's.
        unsafe {
            host_copy(
                stream,
                resource,
                layout,
                bytes.as_mut_ptr(),
                bytes.len(),
                false,
            )
        }
    }

    unsafe fn copy_to_device(
        resource: &OpenClResource,
        layout: &CopyLayout<'_>,
        data: &[u8],
        stream: &OpenClStream,
    ) -> Result<(), IoError> {
        // SAFETY: as the function's; nothing writes the host bytes copied to the device.
        unsafe {
            host_copy(
                stream,
                resource,
                layout,
                data.as_ptr().cast_mut(),
                data.len(),
                true,
            )
        }
    }

    unsafe fn copy_on_device(
        source: &OpenClResource,
        target: &OpenClResource,
        queue: Queue,
    ) -> Result<(), IoError> {
        let copy = (
            target.at(0),
            source.at(0).cast_const(),
            source.size as usize,
        );
        // SAFETY: two live slices of the same size, untouched until the queue's next fence.
        unsafe { queue.copies([copy]) }?;
        Ok(queue.barrier()?)
    }

    fn launch(
        ctx: &mut OpenClContext,
        stream: &mut OpenClStream,
        _id: &KernelId,
        kernel: &OpenClKernel,
        count: (u32, u32, u32),
        args: &mut OpenClLaunch,
    ) -> Result<(), LaunchError> {
        let raw = kernel.kernel.raw();
        let mut index = 0;
        for buffer in &args.buffers {
            // SAFETY: the kernel is live and the pointer lies in this context's device memory.
            unsafe {
                stream
                    .queue
                    .memory
                    .set_arg(raw, index, buffer.at(0).cast_const())
            }?;
            index += 1;
        }
        // The dynamic metadata as a buffer of its own, released once the launch is enqueued; the
        // static part by value.
        let info = &args.info;
        let static_len = info.dynamic_metadata_offset.min(info.data.len());
        let metadata = match kernel.has_dynamic_meta {
            true => {
                let tail = bytemuck::cast_slice(&info.data[static_len..]);
                let metadata = Metadata::new(ctx.context, tail)?;
                // SAFETY: the kernel and the buffer are live, in the same context.
                unsafe { cl::set_arg(raw, index, &metadata.0) }?;
                index += 1;
                Some(metadata)
            }
            false => None,
        };
        if kernel.has_info {
            let bytes: &[u8] = bytemuck::cast_slice(&info.data[..static_len]);
            // SAFETY: the kernel's info struct, by value: the static part of the info is its layout.
            unsafe { cl::set_arg(raw, index, bytes) }?;
        }
        let dim = kernel.cube_dim;
        let groups = [count.0 as usize, count.1 as usize, count.2 as usize];
        let local = [dim.x as usize, dim.y as usize, dim.z as usize];
        // SAFETY: every argument is set, to memory that outlives the run.
        args.event = unsafe { cl::launch(stream.queue.raw, raw, groups, local, args.timed) }
            .map_err(|err| launch_error(err, dim.num_elems(), ctx.max_units))?;
        drop(metadata);
        Ok(stream.enqueued()?)
    }
}

/// Enqueues the copy `layout` describes between `resource` and the `len` host bytes at `host`, a
/// contiguous run at a time, checked to fill the host bytes and lie in the resource.
///
/// # Safety
///
/// `host` is valid for `len` bytes, and both buffers outlive the copy.
unsafe fn host_copy(
    stream: &OpenClStream,
    resource: &OpenClResource,
    layout: &CopyLayout<'_>,
    host: *mut u8,
    len: usize,
    to_device: bool,
) -> Result<(), IoError> {
    // Rows of `width` bytes, `stride` apart in the resource and one after the other on the host.
    let total = layout.shape.iter().product::<usize>() * layout.elem_size;
    let (rows, width, stride) = layout.pitch.map_or((1, total, total), |pitch| {
        (pitch.height, pitch.width_bytes, pitch.stride_bytes)
    });
    let end = rows.saturating_sub(1) * stride + width;
    if rows * width != len || end as u64 > resource.size {
        return Err(IoError::Unknown {
            description: format!(
                "a copy of {} bytes between {len} host bytes and a {}-byte binding",
                rows * width,
                resource.size
            ),
            backtrace: BackTrace::capture(),
        });
    }
    let copies = (0..rows).map(|row| {
        let device = resource.at(row * stride);
        let host = host.wrapping_add(row * width).cast::<c_void>();
        match to_device {
            true => (device, host.cast_const(), width),
            false => (host, device.cast_const(), width),
        }
    });
    // SAFETY: the runs lie in both buffers, which outlive the copy.
    unsafe { stream.queue.copies(copies) }?;
    Ok(stream.enqueued()?)
}

/// A refused enqueue as a launch error: a work-group size the build took but the device refuses
/// at enqueue (`PoCL`) is too many units.
fn launch_error(err: ClError, requested: u32, max: u32) -> LaunchError {
    match err.code {
        Some(CL_INVALID_WORK_GROUP_SIZE) => {
            LaunchError::TooManyResources(ResourceLimitError::Units {
                requested,
                max,
                backtrace: BackTrace::capture(),
            })
        }
        _ => err.into(),
    }
}

/// The dynamic metadata of one launch: a read-only buffer holding a copy of it, released once the launch is
/// enqueued (the kernel keeps it alive until it has run).
struct Metadata(cl_mem);

impl Metadata {
    fn new(context: cl_context, data: &[u8]) -> Result<Self, ClError> {
        // An empty tail (a kernel whose tensors all have rank 0) still binds a buffer.
        let padded = [0u8; 4];
        let data = if data.is_empty() { &padded[..] } else { data };
        // SAFETY: `data` is valid for its length; the buffer copies it at creation.
        unsafe {
            cl3::memory::create_buffer(
                context,
                CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                data.len(),
                data.as_ptr().cast_mut().cast(),
            )
        }
        .map(Self)
        .map_err(ClError::of("clCreateBuffer"))
    }
}

impl Drop for Metadata {
    fn drop(&mut self) {
        // The reference this value owns; the runtime keeps the buffer while a kernel uses it.
        cl::api().clReleaseMemObject(self.0);
    }
}
