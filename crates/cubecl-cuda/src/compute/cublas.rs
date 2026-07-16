use cubecl_common::backtrace::BackTrace;
use cubecl_core::{
    ir::{ElemType, FloatKind},
    server::{GemmDescriptor, GemmMatrix, ServerError},
};
use cudarc::cublas::{result, sys};
use std::ffi::c_void;

use super::storage::gpu::GpuResource;

#[derive(Default)]
pub(crate) struct CublasState {
    handle: Option<sys::cublasHandle_t>,
    stream: Option<cudarc::driver::sys::CUstream>,
}

impl core::fmt::Debug for CublasState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CublasState")
            .field("initialized", &self.handle.is_some())
            .finish()
    }
}

impl CublasState {
    pub(crate) fn launch(
        &mut self,
        descriptor: &GemmDescriptor,
        lhs: &GpuResource,
        rhs: &GpuResource,
        out: &GpuResource,
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<(), ServerError> {
        if descriptor.elem != ElemType::Float(FloatKind::BF16) {
            return Err(validation_error("cuBLAS GEMM currently supports only BF16"));
        }
        validate(descriptor, lhs, rhs, out)?;

        if descriptor.m == 0
            || descriptor.n == 0
            || descriptor.k == 0
            || descriptor.batch_count == 0
        {
            return Ok(());
        }

        let handle = match self.handle {
            Some(handle) => handle,
            None => {
                let handle = result::create_handle().map_err(cublas_error)?;
                self.handle = Some(handle);
                handle
            }
        };
        if self.stream != Some(stream) {
            // SAFETY: both handles are owned by this CUDA server. Changing the
            // cuBLAS stream only changes where subsequent work is enqueued.
            unsafe { result::set_stream(handle, stream.cast()) }.map_err(cublas_error)?;
            self.stream = Some(stream);
        }

        let alpha = 1.0f32;
        let beta = 0.0f32;
        let op_rhs = operation(&descriptor.rhs);
        let op_lhs = operation(&descriptor.lhs);

        // cuBLAS is column-major. Swapping the row-major operands computes
        // C^T = rhs^T @ lhs^T without copies. The call is asynchronous on the
        // exact stream resolved by CubeCL for these bindings.
        unsafe {
            result::gemm_strided_batched_ex(
                handle,
                op_rhs,
                op_lhs,
                descriptor.n as i32,
                descriptor.m as i32,
                descriptor.k as i32,
                (&alpha as *const f32).cast::<c_void>(),
                rhs.ptr as *const c_void,
                sys::cudaDataType::CUDA_R_16BF,
                descriptor.rhs.leading_dimension as i32,
                descriptor.rhs.batch_stride as i64,
                lhs.ptr as *const c_void,
                sys::cudaDataType::CUDA_R_16BF,
                descriptor.lhs.leading_dimension as i32,
                descriptor.lhs.batch_stride as i64,
                (&beta as *const f32).cast::<c_void>(),
                out.ptr as *mut c_void,
                sys::cudaDataType::CUDA_R_16BF,
                descriptor.out.leading_dimension as i32,
                descriptor.out.batch_stride as i64,
                descriptor.batch_count as i32,
                sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP,
            )
        }
        .map_err(cublas_error)?;

        Ok(())
    }

    pub(crate) fn destroy(&mut self) {
        if let Some(handle) = self.handle.take() {
            // SAFETY: this state uniquely owns the handle and destroys it once.
            if let Err(err) = unsafe { result::destroy_handle(handle) } {
                log::warn!("Unable to destroy cuBLAS handle: {err}");
            }
        }
    }
}

fn operation(matrix: &GemmMatrix) -> sys::cublasOperation_t {
    if matrix.transposed {
        sys::cublasOperation_t::CUBLAS_OP_T
    } else {
        sys::cublasOperation_t::CUBLAS_OP_N
    }
}

fn validate(
    descriptor: &GemmDescriptor,
    lhs: &GpuResource,
    rhs: &GpuResource,
    out: &GpuResource,
) -> Result<(), ServerError> {
    let max_i32 = i32::MAX as u32;
    if descriptor.m > max_i32
        || descriptor.n > max_i32
        || descriptor.k > max_i32
        || descriptor.batch_count > max_i32
    {
        return Err(validation_error(
            "GEMM dimensions exceed the cuBLAS i32 API",
        ));
    }
    if descriptor.out.transposed {
        return Err(validation_error("GEMM output must be row-major"));
    }

    validate_matrix(
        "lhs",
        &descriptor.lhs,
        descriptor.m,
        descriptor.k,
        descriptor.batch_count,
        lhs,
        true,
    )?;
    validate_matrix(
        "rhs",
        &descriptor.rhs,
        descriptor.k,
        descriptor.n,
        descriptor.batch_count,
        rhs,
        true,
    )?;
    validate_matrix(
        "out",
        &descriptor.out,
        descriptor.m,
        descriptor.n,
        descriptor.batch_count,
        out,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_matrix(
    name: &str,
    matrix: &GemmMatrix,
    rows: u32,
    cols: u32,
    batches: u32,
    resource: &GpuResource,
    allow_broadcast: bool,
) -> Result<(), ServerError> {
    let required_ld = if matrix.transposed { rows } else { cols };
    if matrix.leading_dimension < required_ld || matrix.leading_dimension > i32::MAX as u32 {
        return Err(validation_error(&format!(
            "{name} leading dimension {} is invalid for logical shape [{rows}, {cols}]",
            matrix.leading_dimension
        )));
    }
    if matrix.batch_stride > i64::MAX as u64 {
        return Err(validation_error(&format!(
            "{name} batch stride exceeds the cuBLAS i64 API"
        )));
    }
    let matrix_elems = if rows == 0 || cols == 0 {
        0
    } else if matrix.transposed {
        (cols as u64 - 1) * matrix.leading_dimension as u64 + rows as u64
    } else {
        (rows as u64 - 1) * matrix.leading_dimension as u64 + cols as u64
    };
    if !allow_broadcast && batches > 1 && matrix.batch_stride < matrix_elems {
        return Err(validation_error("GEMM output batches may not overlap"));
    }
    let batch_offset = if batches <= 1 || matrix.batch_stride == 0 {
        0
    } else {
        (batches as u64 - 1)
            .checked_mul(matrix.batch_stride)
            .ok_or_else(|| validation_error("GEMM batch stride overflow"))?
    };
    let required_bytes = batch_offset
        .checked_add(matrix_elems)
        .and_then(|elements| elements.checked_mul(2))
        .ok_or_else(|| validation_error("GEMM buffer size overflow"))?;
    if required_bytes > resource.size {
        return Err(validation_error(&format!(
            "{name} requires {required_bytes} bytes but its binding contains {}",
            resource.size
        )));
    }
    Ok(())
}

fn validation_error(message: &str) -> ServerError {
    ServerError::Validation {
        message: message.into(),
        backtrace: BackTrace::capture(),
    }
}

fn cublas_error(error: result::CublasError) -> ServerError {
    ServerError::Generic {
        reason: format!("cuBLAS GEMM failed: {error}"),
        backtrace: BackTrace::capture(),
    }
}
