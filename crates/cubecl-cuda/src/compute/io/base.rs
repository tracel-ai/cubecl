use super::controller::PinnedMemoryManagedAllocController;
use crate::compute::{CudaContext, MB, valid_strides};
use cubecl_common::bytes::Bytes;
use cubecl_core::server::{CopyDescriptor, IoError};
use cubecl_runtime::memory_management::MemoryHandle;
use cudarc::driver::sys::{CUDA_MEMCPY2D_st, CUmemorytype, cuMemcpy2DAsync_v2};
use std::{ffi::c_void, ops::DerefMut};

/// Registers multiple lazy buffer copies to [Bytes], potentially using pinned memory.
///
/// # Arguments
///
/// * `ctx` - The CUDA context for managing memory and streams.
/// * `descriptors` - A vector of copy descriptors specifying the source data.
///
/// # Returns
///
/// A [Result] containing a vector of [Bytes] with the copied data, or an [IoError] if any copy fails.
pub fn register_copies_to_bytes(
    ctx: &mut CudaContext,
    descriptors: Vec<CopyDescriptor<'_>>,
) -> Result<Vec<Bytes>, IoError> {
    let mut result = Vec::with_capacity(descriptors.len());

    for descriptor in descriptors {
        result.push(register_copy_to_bytes(ctx, descriptor, false)?);
    }

    Ok(result)
}

/// Registers a single lazy buffer copy to [Bytes], potentially using pinned memory.
///
/// # Arguments
///
/// * `ctx` - The CUDA context for managing memory and streams.
/// * `descriptor` - The copy descriptor specifying the source data.
/// * `marked_pinned` - Whether to force the use of pinned memory for the copy.
///
/// # Returns
///
/// A [Result] containing the copied data as [Bytes], or an [IoError] if the copy fails.
pub fn register_copy_to_bytes(
    ctx: &mut CudaContext,
    descriptor: CopyDescriptor<'_>,
    marked_pinned: bool,
) -> Result<Bytes, IoError> {
    let CopyDescriptor {
        binding,
        shape,
        strides,
        elem_size,
    } = descriptor;

    if !valid_strides(shape, strides) {
        return Err(IoError::UnsupportedStrides);
    }

    let num_bytes = shape.iter().product::<usize>() * elem_size;
    let resource = ctx
        .memory_management_gpu
        .get_resource(binding.memory, binding.offset_start, binding.offset_end)
        .ok_or(IoError::InvalidHandle)?;

    let mut bytes = bytes_from_managed_pinned_memory(ctx, num_bytes, marked_pinned)
        .unwrap_or_else(|| Bytes::from_bytes_vec(vec![0; num_bytes]));

    let rank = shape.len();
    if rank <= 1 {
        unsafe {
            cudarc::driver::result::memcpy_dtoh_async(bytes.deref_mut(), resource.ptr, ctx.stream)
                .map_err(|e| IoError::Unknown(format!("CUDA memcpy failed: {}", e)))?;
        }
        return Ok(bytes);
    }

    let dim_x = shape[rank - 1];
    let width_bytes = dim_x * elem_size;
    let dim_y: usize = shape.iter().rev().skip(1).product();
    let pitch = strides[rank - 2] * elem_size;
    let slice = bytes.deref_mut();

    let cpy = CUDA_MEMCPY2D_st {
        srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
        srcDevice: resource.ptr,
        srcPitch: pitch,
        dstMemoryType: CUmemorytype::CU_MEMORYTYPE_HOST,
        dstHost: slice.as_mut_ptr() as *mut c_void,
        dstPitch: width_bytes,
        WidthInBytes: width_bytes,
        Height: dim_y,
        ..Default::default()
    };

    unsafe {
        cuMemcpy2DAsync_v2(&cpy, ctx.stream)
            .result()
            .map_err(|e| IoError::Unknown(format!("CUDA 2D memcpy failed: {}", e)))?;
    }

    Ok(bytes)
}

/// Creates a [Bytes] instance from pinned memory, if suitable for the given size.
///
/// For small data transfers (<= 100 MB) or when explicitly marked as pinned, this function
/// uses pinned memory to optimize performance. For larger transfers, it falls back to regular memory.
///
/// # Arguments
///
/// * `ctx` - The CUDA context for managing memory.
/// * `num_bytes` - The number of bytes to allocate.
/// * `marked_pinned` - Whether to force the use of pinned memory.
///
/// # Returns
///
/// An [Option] containing a [Bytes] instance if pinned memory is used, or [None] if regular memory should be used instead.
fn bytes_from_managed_pinned_memory(
    ctx: &mut CudaContext,
    num_bytes: usize,
    marked_pinned: bool,
) -> Option<Bytes> {
    // Use pinned memory for small transfers (<= 100 MB) or when explicitly marked.
    if !marked_pinned && num_bytes > 100 * MB {
        return None;
    }

    let handle = ctx.memory_management_cpu.reserve(num_bytes as u64).ok()?;
    let binding = handle.binding();
    let resource = ctx
        .memory_management_cpu
        .get_resource(binding.clone(), None, None)
        .ok_or(IoError::InvalidHandle)
        .ok()?;

    let (controller, alloc) = PinnedMemoryManagedAllocController::init(binding, resource);

    Some(unsafe { Bytes::from_raw_parts(alloc, num_bytes, Box::new(controller)) })
}
