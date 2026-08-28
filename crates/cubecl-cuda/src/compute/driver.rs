//! What the shared [`Command`](cubecl_runtime::command::Command) cannot do
//! itself: CUDA's four device calls, and where a CUDA stream keeps the state
//! they move in step with.

use crate::compute::context::CudaContext;
use crate::compute::storage::cpu::PinnedMemoryStorage;
use crate::compute::storage::gpu::{GpuResource, GpuStorage};
use crate::compute::stream::{CudaStreamBackend, Stream};
use crate::compute::sync::Fence;
use cubecl_common::bytes::Bytes;
use cubecl_environment::backtrace::BackTrace;
use cubecl_runtime::command::{CopyLayout, DeviceStream, Driver};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::memory_management::drop_queue::PendingDropQueue;
use cubecl_runtime::memory_management::{ManagedMemoryBinding, MemoryManagement};
use cubecl_runtime::metadata_cache::MetadataInfoCache;
use cubecl_runtime::server::{Handle, IoError, LaunchError};
use cubecl_runtime::storage::{ComputeStorage, PinnedMemoryAllocController};
use cubecl_runtime::stream::StreamCapture;
use cudarc::driver::sys::{CUDA_MEMCPY2D_st, CUmemorytype, CUstream_st, cuMemcpy2DAsync_v2};
use std::ffi::c_void;

impl DeviceStream for Stream {
    type Fence = Fence;
    type DeviceStorage = GpuStorage;
    type HostStorage = PinnedMemoryStorage;
    type Signal = *mut CUstream_st;

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

/// CUDA, as the shared command sees it.
pub(crate) struct Cuda;

impl Driver for Cuda {
    type Backend = CudaStreamBackend;
    type Stream = Stream;
    type Context = CudaContext;
    /// The driver takes an array of pointers to the arguments, so a tensor-map
    /// descriptor sits in it beside a buffer's device pointer.
    type LaunchArgs = [*mut c_void];

    unsafe fn pinned_bytes(
        binding: ManagedMemoryBinding,
        resource: <PinnedMemoryStorage as ComputeStorage>::Resource,
        size: usize,
    ) -> Bytes {
        let controller = Box::new(PinnedMemoryAllocController::init(binding, resource));
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
            return unsafe {
                cudarc::driver::result::memcpy_dtoh_async(bytes, resource.ptr, stream.sys)
                    .map_err(|err| copy_failed("memcpy_dtoh_async", err, layout))
            };
        };

        let copy = CUDA_MEMCPY2D_st {
            srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
            srcDevice: resource.ptr,
            srcPitch: pitch.stride_bytes,
            dstMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
            dstHost: bytes.as_mut_ptr() as *mut c_void,
            dstPitch: pitch.width_bytes,
            WidthInBytes: pitch.width_bytes,
            Height: pitch.height,
            srcXInBytes: Default::default(),
            srcY: Default::default(),
            srcHost: Default::default(),
            srcArray: Default::default(),
            dstXInBytes: Default::default(),
            dstY: Default::default(),
            dstDevice: Default::default(),
            dstArray: Default::default(),
        };
        // SAFETY: the descriptor is fully initialized from a validated layout,
        // and both sides point at live allocations of the size it names.
        unsafe {
            cuMemcpy2DAsync_v2(&copy, stream.sys)
                .result()
                .map_err(|err| copy_failed("cuMemcpy2DAsync_v2", err, layout))
        }
    }

    unsafe fn copy_to_device(
        resource: &GpuResource,
        layout: &CopyLayout<'_>,
        data: &[u8],
        stream: &Stream,
    ) -> Result<(), IoError> {
        let Some(pitch) = layout.pitch else {
            // The one write the taint bookkeeping cannot record: an oversized
            // copy corrupts whatever pool slice sits past the target, memory
            // no failure ever claimed. Same check as the HIP twin.
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
            // covers it, and `data` is a live host slice the caller keeps
            // alive until the stream is synchronized.
            return unsafe {
                cudarc::driver::result::memcpy_htod_async(resource.ptr, data, stream.sys)
                    .map_err(|err| copy_failed("memcpy_htod_async", err, layout))
            };
        };

        let copy = CUDA_MEMCPY2D_st {
            srcMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
            srcHost: data.as_ptr() as *const c_void,
            srcPitch: pitch.width_bytes,
            dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
            dstDevice: resource.ptr,
            dstPitch: pitch.stride_bytes,
            WidthInBytes: pitch.width_bytes,
            Height: pitch.height,
            srcXInBytes: Default::default(),
            srcY: Default::default(),
            srcDevice: Default::default(),
            srcArray: Default::default(),
            dstXInBytes: Default::default(),
            dstY: Default::default(),
            dstHost: Default::default(),
            dstArray: Default::default(),
        };
        // SAFETY: the descriptor is fully initialized from a validated layout,
        // and both sides point at live allocations of the size it names.
        unsafe {
            cuMemcpy2DAsync_v2(&copy, stream.sys)
                .result()
                .map_err(|err| copy_failed("cuMemcpy2DAsync_v2", err, layout))
        }
    }

    fn launch(
        ctx: &mut CudaContext,
        stream: &mut Stream,
        kernel: KernelId,
        count: (u32, u32, u32),
        args: &mut [*mut c_void],
    ) -> Result<(), LaunchError> {
        ctx.execute_task(stream, kernel, count, args)
    }
}

/// A driver copy that failed, named alongside the layout it was given.
///
/// The geometry is what makes one of these diagnosable: a refusal on a shape
/// the driver will not take reads very differently from one on a shape it
/// should have.
fn copy_failed(op: &str, err: impl core::fmt::Display, layout: &CopyLayout<'_>) -> IoError {
    IoError::Unknown {
        description: format!(
            "CUDA {op} failed: {err}; shape {:?}, strides {:?}, elem_size {}, pitch {:?}",
            layout.shape, layout.strides, layout.elem_size, layout.pitch
        ),
        backtrace: BackTrace::capture(),
    }
}
