use super::controller::PinnedMemoryManagedAllocController;
use crate::compute::{CudaContext, MB, valid_strides};
use cubecl_common::bytes::Bytes;
use cubecl_core::server::{CopyDescriptor, IoError};
use cubecl_runtime::memory_management::MemoryHandle;
use cudarc::driver::sys::{CUDA_MEMCPY2D_st, CUmemorytype, cuMemcpy2DAsync_v2};
use std::{ffi::c_void, ops::DerefMut};

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

/// Register a lazy buffer copy potentially using pinned memory.
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
                .unwrap();
        };
        return Ok(bytes);
    }

    let dim_x = shape[rank - 1];
    let width_bytes = dim_x * elem_size;
    let dim_y: usize = shape.iter().rev().skip(1).product();
    let pitch = strides[rank - 2] * elem_size;
    let slice: &mut [u8] = bytes.deref_mut();

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
        cuMemcpy2DAsync_v2(&cpy, ctx.stream).result().unwrap();
    };

    Ok(bytes)
}

fn bytes_from_managed_pinned_memory(
    ctx: &mut CudaContext,
    num_bytes: usize,
    marked_pinned: bool,
) -> Option<Bytes> {
    // If not marked as pinned memory, we still use pinned memory for small data transfer to
    // speedup sync caused by small reads.
    if !marked_pinned && num_bytes > 100 * MB {
        return None;
    }

    let handle = ctx.memory_management_cpu.reserve(num_bytes as u64).ok()?;
    let binding = handle.binding();
    let resource = ctx
        .memory_management_cpu
        .get_resource(binding.clone(), None, None)?;

    let (controler, alloc) = PinnedMemoryManagedAllocController::init(binding, resource).ok()?;

    Some(unsafe { Bytes::from_raw_parts(alloc, num_bytes as usize, Box::new(controler)) })
}
