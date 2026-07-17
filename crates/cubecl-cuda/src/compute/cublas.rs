use std::collections::HashMap;
use std::ffi::c_void;

use cubecl_common::backtrace::BackTrace;
use cubecl_core::{
    ir::{ElemType, FloatKind},
    server::{GemmDescriptor, GemmMatrix, ServerError},
};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublaslt::{result as lt, sys};

use super::storage::gpu::GpuResource;

/// Workspace given to every cublasLt matmul. 32 MiB matches the size the
/// cuBLAS documentation recommends for Ampere+ so the heuristic can pick
/// split-K and other workspace-hungry algorithms.
const WORKSPACE_BYTES: usize = 32 * 1024 * 1024;

/// A cached execution plan: the descriptor/layout objects plus the algorithm
/// the cublasLt heuristic selected for one GEMM shape.
struct MatmulPlan {
    desc: sys::cublasLtMatmulDesc_t,
    a_layout: sys::cublasLtMatrixLayout_t,
    b_layout: sys::cublasLtMatrixLayout_t,
    d_layout: sys::cublasLtMatrixLayout_t,
    algo: sys::cublasLtMatmulAlgo_t,
}

#[derive(PartialEq, Eq, Hash)]
struct MatrixKey {
    leading_dimension: u32,
    batch_stride: u64,
    transposed: bool,
}

#[derive(PartialEq, Eq, Hash)]
struct PlanKey {
    m: u32,
    n: u32,
    k: u32,
    batch_count: u32,
    lhs: MatrixKey,
    rhs: MatrixKey,
    out: MatrixKey,
}

impl PlanKey {
    fn new(descriptor: &GemmDescriptor) -> Self {
        let matrix = |m: &GemmMatrix| MatrixKey {
            leading_dimension: m.leading_dimension,
            batch_stride: m.batch_stride,
            transposed: m.transposed,
        };
        Self {
            m: descriptor.m,
            n: descriptor.n,
            k: descriptor.k,
            batch_count: descriptor.batch_count,
            lhs: matrix(&descriptor.lhs),
            rhs: matrix(&descriptor.rhs),
            out: matrix(&descriptor.out),
        }
    }
}

#[derive(Default)]
pub(crate) struct CublasState {
    handle: Option<sys::cublasLtHandle_t>,
    /// One workspace per CUDA stream: concurrent matmuls on different
    /// streams must not share scratch memory.
    workspaces: HashMap<usize, cudarc::driver::sys::CUdeviceptr>,
    plans: HashMap<PlanKey, MatmulPlan>,
}

impl core::fmt::Debug for CublasState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CublasState")
            .field("initialized", &self.handle.is_some())
            .field("plans", &self.plans.len())
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
        if descriptor.m == 0 || descriptor.n == 0 || descriptor.batch_count == 0 {
            return Ok(());
        }
        if descriptor.k == 0 {
            return Err(validation_error(
                "cuBLAS GEMM does not initialize a nonempty zero-K output",
            ));
        }
        validate(descriptor, lhs, rhs, out)?;

        let handle = match self.handle {
            Some(handle) => handle,
            None => {
                let handle = lt::create_handle().map_err(cublas_error)?;
                self.handle = Some(handle);
                handle
            }
        };
        let workspace = match self.workspaces.get(&(stream as usize)) {
            Some(workspace) => *workspace,
            None => {
                // SAFETY: the server made its CUDA context current before
                // resolving the streams for this launch.
                let workspace = unsafe { cudarc::driver::result::malloc_sync(WORKSPACE_BYTES) }
                    .map_err(|err| ServerError::Generic {
                        reason: format!("cublasLt workspace allocation failed: {err}"),
                        backtrace: BackTrace::capture(),
                    })?;
                self.workspaces.insert(stream as usize, workspace);
                workspace
            }
        };

        let key = PlanKey::new(descriptor);
        if !self.plans.contains_key(&key) {
            let plan = build_plan(handle, descriptor)?;
            self.plans.insert(PlanKey::new(descriptor), plan);
        }
        let plan = self
            .plans
            .get(&key)
            .expect("cublasLt plan was just inserted");

        let alpha = 1.0f32;
        let beta = 0.0f32;

        // cuBLAS is column-major. Swapping the row-major operands computes
        // D^T = rhs^T @ lhs^T without copies, so A carries `rhs` and B
        // carries `lhs`. The call is asynchronous on the given stream.
        unsafe {
            lt::matmul(
                handle,
                plan.desc,
                (&alpha as *const f32).cast::<c_void>(),
                (&beta as *const f32).cast::<c_void>(),
                rhs.ptr as *const c_void,
                plan.a_layout,
                lhs.ptr as *const c_void,
                plan.b_layout,
                out.ptr as *const c_void,
                plan.d_layout,
                out.ptr as *mut c_void,
                plan.d_layout,
                &plan.algo,
                workspace as *mut c_void,
                WORKSPACE_BYTES,
                stream.cast(),
            )
        }
        .map_err(cublas_error)?;

        Ok(())
    }

    pub(crate) fn destroy(&mut self) {
        for (_, plan) in self.plans.drain() {
            // SAFETY: each plan uniquely owns its descriptor objects and is
            // destroyed exactly once.
            unsafe {
                let _ = lt::destroy_matmul_desc(plan.desc);
                let _ = lt::destroy_matrix_layout(plan.a_layout);
                let _ = lt::destroy_matrix_layout(plan.b_layout);
                let _ = lt::destroy_matrix_layout(plan.d_layout);
            }
        }
        for (_, workspace) in self.workspaces.drain() {
            // SAFETY: the workspace was allocated by this state on the
            // server's context and freed exactly once.
            if let Err(err) = unsafe { cudarc::driver::result::free_sync(workspace) } {
                log::warn!("Unable to free cublasLt workspace: {err}");
            }
        }
        if let Some(handle) = self.handle.take() {
            // SAFETY: this state uniquely owns the handle and destroys it once.
            if let Err(err) = unsafe { lt::destroy_handle(handle) } {
                log::warn!("Unable to destroy cublasLt handle: {err}");
            }
        }
    }
}

/// Build the descriptor, layouts, and heuristic-selected algorithm for one
/// GEMM shape. All dimensions below are in cuBLAS column-major terms, i.e.
/// the row-major operands are swapped: `m_lt = n`, `n_lt = m`, `A = rhs`,
/// `B = lhs`.
fn build_plan(
    handle: sys::cublasLtHandle_t,
    descriptor: &GemmDescriptor,
) -> Result<MatmulPlan, ServerError> {
    let m_lt = descriptor.n as u64;
    let n_lt = descriptor.m as u64;
    let k_lt = descriptor.k as u64;
    let op_a = operation(&descriptor.rhs);
    let op_b = operation(&descriptor.lhs);

    let desc = lt::create_matmul_desc(
        sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        sys::cudaDataType::CUDA_R_32F,
    )
    .map_err(cublas_error)?;
    let destroy_desc = || {
        // SAFETY: created above, not yet owned by a plan.
        unsafe {
            let _ = lt::destroy_matmul_desc(desc);
        }
    };
    for (attr, op) in [
        (
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            op_a,
        ),
        (
            sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB,
            op_b,
        ),
    ] {
        let value = op as i32;
        // SAFETY: `desc` is live and the attribute is an i32 by contract.
        if let Err(err) = unsafe {
            lt::set_matmul_desc_attribute(
                desc,
                attr,
                (&value as *const i32).cast::<c_void>(),
                core::mem::size_of::<i32>(),
            )
        } {
            destroy_desc();
            return Err(cublas_error(err));
        }
    }

    // Stored (pre-transpose) dimensions of each column-major operand.
    let a_dims = if matches!(op_a, cublasOperation_t::CUBLAS_OP_N) {
        (m_lt, k_lt)
    } else {
        (k_lt, m_lt)
    };
    let b_dims = if matches!(op_b, cublasOperation_t::CUBLAS_OP_N) {
        (k_lt, n_lt)
    } else {
        (n_lt, k_lt)
    };
    let layout = |rows: u64, cols: u64, matrix: &GemmMatrix| -> Result<_, ServerError> {
        let layout = lt::create_matrix_layout(
            sys::cudaDataType::CUDA_R_16BF,
            rows,
            cols,
            matrix.leading_dimension as i64,
        )
        .map_err(cublas_error)?;
        let batch_count = descriptor.batch_count as i32;
        let batch_stride = matrix.batch_stride as i64;
        // SAFETY: `layout` is live; both attributes take the given widths.
        let result = unsafe {
            lt::set_matrix_layout_attribute(
                layout,
                sys::cublasLtMatrixLayoutAttribute_t::CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                (&batch_count as *const i32).cast::<c_void>(),
                core::mem::size_of::<i32>(),
            )
            .and_then(|_| {
                lt::set_matrix_layout_attribute(
                    layout,
                    sys::cublasLtMatrixLayoutAttribute_t::
                        CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
                    (&batch_stride as *const i64).cast::<c_void>(),
                    core::mem::size_of::<i64>(),
                )
            })
        };
        if let Err(err) = result {
            // SAFETY: created above, not yet owned by a plan.
            unsafe {
                let _ = lt::destroy_matrix_layout(layout);
            }
            return Err(cublas_error(err));
        }
        Ok(layout)
    };

    let a_layout = match layout(a_dims.0, a_dims.1, &descriptor.rhs) {
        Ok(layout) => layout,
        Err(err) => {
            destroy_desc();
            return Err(err);
        }
    };
    let b_layout = match layout(b_dims.0, b_dims.1, &descriptor.lhs) {
        Ok(layout) => layout,
        Err(err) => {
            destroy_desc();
            // SAFETY: created above, not yet owned by a plan.
            unsafe {
                let _ = lt::destroy_matrix_layout(a_layout);
            }
            return Err(err);
        }
    };
    let d_layout = match layout(m_lt, n_lt, &descriptor.out) {
        Ok(layout) => layout,
        Err(err) => {
            destroy_desc();
            // SAFETY: created above, not yet owned by a plan.
            unsafe {
                let _ = lt::destroy_matrix_layout(a_layout);
                let _ = lt::destroy_matrix_layout(b_layout);
            }
            return Err(err);
        }
    };
    let cleanup = || {
        destroy_desc();
        // SAFETY: created above, not yet owned by a plan.
        unsafe {
            let _ = lt::destroy_matrix_layout(a_layout);
            let _ = lt::destroy_matrix_layout(b_layout);
            let _ = lt::destroy_matrix_layout(d_layout);
        }
    };

    let pref = match lt::create_matmul_pref() {
        Ok(pref) => pref,
        Err(err) => {
            cleanup();
            return Err(cublas_error(err));
        }
    };
    let workspace_bytes = WORKSPACE_BYTES as u64;
    // SAFETY: `pref` is live and the attribute is a u64 by contract.
    if let Err(err) = unsafe {
        lt::set_matmul_pref_attribute(
            pref,
            sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&workspace_bytes as *const u64).cast::<c_void>(),
            core::mem::size_of::<u64>(),
        )
    } {
        // SAFETY: created above.
        unsafe {
            let _ = lt::destroy_matmul_pref(pref);
        }
        cleanup();
        return Err(cublas_error(err));
    }

    // SAFETY: every descriptor is live; the C layout equals the D layout
    // because beta is always zero.
    let heuristic = unsafe {
        lt::get_matmul_algo_heuristic(handle, desc, a_layout, b_layout, d_layout, d_layout, pref)
    };
    // SAFETY: created above.
    unsafe {
        let _ = lt::destroy_matmul_pref(pref);
    }
    let heuristic = match heuristic {
        Ok(heuristic) => heuristic,
        Err(_) => {
            cleanup();
            return Err(validation_error(
                "cublasLt has no algorithm for this GEMM shape",
            ));
        }
    };

    Ok(MatmulPlan {
        desc,
        a_layout,
        b_layout,
        d_layout,
        algo: heuristic.algo,
    })
}

fn operation(matrix: &GemmMatrix) -> cublasOperation_t {
    if matrix.transposed {
        cublasOperation_t::CUBLAS_OP_T
    } else {
        cublasOperation_t::CUBLAS_OP_N
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
    if resources_overlap(out, lhs) || resources_overlap(out, rhs) {
        return Err(validation_error("GEMM output may not overlap either input"));
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

fn resources_overlap(lhs: &GpuResource, rhs: &GpuResource) -> bool {
    if lhs.size == 0 || rhs.size == 0 {
        return false;
    }
    let lhs_end = lhs.ptr.saturating_add(lhs.size);
    let rhs_end = rhs.ptr.saturating_add(rhs.size);
    lhs.ptr < rhs_end && rhs.ptr < lhs_end
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

fn cublas_error(error: lt::CublasError) -> ServerError {
    ServerError::Generic {
        reason: format!("cuBLAS error: {error:?}"),
        backtrace: BackTrace::capture(),
    }
}
