//! What the shared [`Command`](cubecl_runtime::command::Command) cannot do
//! itself: HIP's four device calls, and where a HIP stream keeps the state
//! they move in step with.

use crate::compute::context::HipContext;
use crate::compute::fence::Fence;
use crate::compute::gpu::GpuResource;
use crate::compute::storage::cpu::PinnedMemoryStorage;
use crate::compute::storage::gpu::GpuStorage;
use crate::compute::stream::{HipStreamBackend, Stream};
use cubecl_common::bytes::Bytes;
use cubecl_hip_sys::{
    hipMemcpyKind_hipMemcpyDeviceToHost, hipMemcpyKind_hipMemcpyHostToDevice, ihipStream_t,
};
use cubecl_runtime::command::{CopyLayout, DeviceStream, Driver};
use cubecl_runtime::driver::checked;
use cubecl_runtime::id::KernelId;
use cubecl_runtime::memory_management::drop_queue::PendingDropQueue;
use cubecl_runtime::memory_management::{ManagedMemoryBinding, MemoryManagement};
use cubecl_runtime::metadata_cache::MetadataInfoCache;
use cubecl_runtime::server::{Handle, IoError, LaunchError};
use cubecl_runtime::storage::PinnedMemoryAllocController;
use cubecl_runtime::stream::StreamCapture;

impl DeviceStream for Stream {
    type Fence = Fence;
    type DeviceStorage = GpuStorage;
    type HostStorage = PinnedMemoryStorage;
    type Signal = *mut ihipStream_t;

    fn device_memory(&mut self) -> &mut MemoryManagement<GpuStorage> {
        &mut self.memory_management_gpu
    }

    fn host_memory(&mut self) -> &mut MemoryManagement<PinnedMemoryStorage> {
        &mut self.memory_management_cpu
    }

    fn drop_queue(&mut self) -> &mut PendingDropQueue<Fence> {
        &mut self.drop_queue
    }

    fn capturing(&mut self) -> &mut StreamCapture {
        &mut self.capturing
    }

    fn info_cache(&mut self) -> &mut MetadataInfoCache<Handle> {
        &mut self.info_cache
    }

    fn signal(&self) -> Self::Signal {
        self.sys
    }

    fn fence(signal: Self::Signal) -> Fence {
        Fence::new(signal)
    }
}

/// HIP, as the shared command sees it.
pub(crate) struct Hip;

impl Driver for Hip {
    type Backend = HipStreamBackend;
    type Stream = Stream;
    type Context = HipContext;
    type LaunchArgs = [GpuResource];

    unsafe fn pinned_bytes(
        binding: ManagedMemoryBinding,
        resource: <PinnedMemoryStorage as cubecl_runtime::storage::ComputeStorage>::Resource,
        size: usize,
    ) -> Bytes {
        let controller =
            alloc::boxed::Box::new(PinnedMemoryAllocController::init(binding, resource));
        // SAFETY: the caller guarantees `binding` names initialized memory of
        // at least `size` bytes.
        unsafe { Bytes::from_controller(controller, size) }
    }

    unsafe fn copy_to_host(
        resource: &GpuResource,
        layout: &CopyLayout<'_>,
        bytes: &mut Bytes,
        stream: &Stream,
    ) -> Result<(), IoError> {
        let Some(pitch) = layout.pitch else {
            // SAFETY: the source is contiguous, so one linear copy of exactly
            // the bytes the destination was sized for.
            let status = unsafe {
                cubecl_hip_sys::hipMemcpyDtoHAsync(
                    bytes.as_mut_ptr() as *mut _,
                    resource.ptr,
                    bytes.len(),
                    stream.sys,
                )
            };
            return Ok(checked("hipMemcpyDtoHAsync", status)?);
        };

        // SAFETY: the source is pitched. The 2D copy respects the stride of the
        // second-to-last dimension; a flat copy would scramble the rows, so a
        // failure is an error rather than a fallback.
        let status = unsafe {
            cubecl_hip_sys::hipMemcpy2DAsync(
                bytes.as_mut_ptr() as *mut _,
                pitch.width_bytes,
                resource.ptr,
                pitch.stride_bytes,
                pitch.width_bytes,
                pitch.height,
                hipMemcpyKind_hipMemcpyDeviceToHost,
                stream.sys,
            )
        };
        checked("hipMemcpy2DAsync", status).map_err(|err| IoError::Unknown {
            description: alloc::format!(
                "{err}; copying to the host from shape {:?}, strides {:?}, elem_size {}, \
                 spitch {}, width {}, height {}",
                layout.shape,
                layout.strides,
                layout.elem_size,
                pitch.stride_bytes,
                pitch.width_bytes,
                pitch.height
            ),
            backtrace: cubecl_environment::backtrace::BackTrace::capture(),
        })?;
        Ok(())
    }

    unsafe fn copy_to_device(
        resource: &GpuResource,
        layout: &CopyLayout<'_>,
        data: &[u8],
        stream: &Stream,
    ) -> Result<(), IoError> {
        let ptr = data as *const _ as *mut _;

        let Some(pitch) = layout.pitch else {
            if resource.size < data.len() as u64 {
                return Err(IoError::Unknown {
                    description: alloc::format!(
                        "write of {} bytes exceeds the target buffer of {} bytes",
                        data.len(),
                        resource.size
                    ),
                    backtrace: cubecl_environment::backtrace::BackTrace::capture(),
                });
            }
            // SAFETY: the destination is contiguous, the bound check above
            // covers it, and `ptr` points to `data.len()` valid host bytes.
            let status = unsafe {
                cubecl_hip_sys::hipMemcpyHtoDAsync(resource.ptr, ptr, data.len(), stream.sys)
            };
            return Ok(checked("hipMemcpyHtoDAsync", status)?);
        };

        // SAFETY: the destination is pitched. The 2D copy lays the rows out at
        // the stride the allocation was sized for.
        let status = unsafe {
            cubecl_hip_sys::hipMemcpy2DAsync(
                resource.ptr,
                pitch.stride_bytes,
                ptr,
                pitch.width_bytes,
                pitch.width_bytes,
                pitch.height,
                hipMemcpyKind_hipMemcpyHostToDevice,
                stream.sys,
            )
        };
        checked("hipMemcpy2DAsync", status).map_err(|err| IoError::Unknown {
            description: alloc::format!(
                "{err}; copying to the device from shape {:?}, strides {:?}, elem_size {}, \
                 dpitch {}, width {}, height {}, resource size {}",
                layout.shape,
                layout.strides,
                layout.elem_size,
                pitch.stride_bytes,
                pitch.width_bytes,
                pitch.height,
                resource.size
            ),
            backtrace: cubecl_environment::backtrace::BackTrace::capture(),
        })?;
        Ok(())
    }

    fn launch(
        ctx: &mut HipContext,
        stream: &mut Stream,
        kernel: KernelId,
        count: (u32, u32, u32),
        args: &mut [GpuResource],
    ) -> Result<(), LaunchError> {
        ctx.execute_task(stream, kernel, count, args)
    }
}
