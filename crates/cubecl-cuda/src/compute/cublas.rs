use std::collections::HashMap;
use std::ffi::c_void;

use cubecl_common::backtrace::BackTrace;
#[cfg(cuda_12050)]
use cubecl_core::server::GroupedGemmDescriptor;
use cubecl_core::{
    ir::{ElemType, FloatKind},
    server::{GemmDescriptor, GemmMatrix, ServerError},
};
use cudarc::cublas::sys::cublasOperation_t;
#[cfg(cuda_12050)]
use cudarc::cublas::{result as blas, sys as blas_sys};
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

/// Pinned-host and device storage for one in-flight grouped-GEMM pointer list.
///
/// cuBLAS consumes the matrix-pointer arrays on the device. The completion
/// event prevents either side of this staging pair from being overwritten
/// while an earlier grouped launch is still reading it.
#[cfg(cuda_12050)]
struct GroupedPointerStaging {
    host: *mut std::ffi::c_void,
    device: cudarc::driver::sys::CUdeviceptr,
    event: cudarc::driver::sys::CUevent,
    capacity: usize,
    in_flight: bool,
    /// Captured graph nodes retain both staging addresses for replay, so a
    /// captured slot must remain immutable until the server is destroyed.
    captured: bool,
}

#[cfg(cuda_12050)]
impl GroupedPointerStaging {
    fn new(capacity: usize) -> Result<Self, ServerError> {
        let bytes = capacity
            .checked_mul(core::mem::size_of::<u64>())
            .ok_or_else(|| validation_error("grouped GEMM pointer staging size overflow"))?;
        // SAFETY: both allocations are owned by the returned staging slot and
        // released exactly once in `destroy`.
        let host = unsafe {
            cudarc::driver::result::malloc_host(
                bytes,
                cudarc::driver::sys::CU_MEMHOSTALLOC_WRITECOMBINED,
            )
        }
        .map_err(cuda_error)?;
        let device = match unsafe { cudarc::driver::result::malloc_sync(bytes) } {
            Ok(device) => device,
            Err(err) => {
                // SAFETY: `host` was allocated immediately above and has not
                // been exposed or freed.
                unsafe {
                    let _ = cudarc::driver::result::free_host(host);
                }
                return Err(cuda_error(err));
            }
        };
        let event = match cudarc::driver::result::event::create(
            cudarc::driver::sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING,
        ) {
            Ok(event) => event,
            Err(err) => {
                // SAFETY: both allocations are uniquely owned here.
                unsafe {
                    let _ = cudarc::driver::result::free_sync(device);
                    let _ = cudarc::driver::result::free_host(host);
                }
                return Err(cuda_error(err));
            }
        };
        Ok(Self {
            host,
            device,
            event,
            capacity,
            in_flight: false,
            captured: false,
        })
    }

    fn available(&self) -> bool {
        !self.captured
            && (!self.in_flight
            // SAFETY: `event` remains live for this slot's entire lifetime.
            || unsafe { cudarc::driver::result::event::query(self.event) }.is_ok())
    }

    fn destroy(self) {
        // The server is shutting down. Waiting here protects the pinned host
        // buffer if a final pointer upload is still in flight.
        if self.in_flight {
            // SAFETY: `event` is live and was recorded after the last grouped
            // launch using this slot.
            let _ = unsafe { cudarc::driver::result::event::synchronize(self.event) };
        }
        // SAFETY: all three resources are uniquely owned by this slot.
        unsafe {
            let _ = cudarc::driver::result::event::destroy(self.event);
            let _ = cudarc::driver::result::free_sync(self.device);
            let _ = cudarc::driver::result::free_host(self.host);
        }
    }
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
    #[cfg(cuda_12050)]
    grouped_handle: Option<blas_sys::cublasHandle_t>,
    /// One workspace per CUDA stream: concurrent matmuls on different
    /// streams must not share scratch memory.
    workspaces: HashMap<usize, cudarc::driver::sys::CUdeviceptr>,
    plans: HashMap<PlanKey, MatmulPlan>,
    #[cfg(cuda_12050)]
    grouped_staging: Vec<GroupedPointerStaging>,
}

impl core::fmt::Debug for CublasState {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut debug = f.debug_struct("CublasState");
        debug.field("initialized", &self.handle.is_some());
        #[cfg(cuda_12050)]
        debug.field("grouped_initialized", &self.grouped_handle.is_some());
        debug.field("plans", &self.plans.len());
        #[cfg(cuda_12050)]
        debug.field("grouped_staging", &self.grouped_staging.len());
        debug.finish()
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

    #[cfg(cuda_12050)]
    pub(crate) fn launch_grouped(
        &mut self,
        descriptor: &GroupedGemmDescriptor,
        lhs: &[GpuResource],
        rhs: &[GpuResource],
        out: &[GpuResource],
        stream: cudarc::driver::sys::CUstream,
    ) -> Result<(), ServerError> {
        let Some(elem) = descriptor.groups.first().map(|group| group.elem) else {
            return Ok(());
        };
        if elem != ElemType::Float(FloatKind::BF16) {
            return Err(validation_error(
                "cuBLAS grouped GEMM currently supports only BF16",
            ));
        }
        if descriptor.groups.len() != lhs.len()
            || descriptor.groups.len() != rhs.len()
            || descriptor.groups.len() != out.len()
        {
            return Err(validation_error(
                "grouped GEMM descriptor/resource count mismatch",
            ));
        }

        let mut active = Vec::with_capacity(descriptor.groups.len());
        let mut pointer_count = 0usize;
        for (index, group) in descriptor.groups.iter().enumerate() {
            if group.elem != elem {
                return Err(validation_error(
                    "every grouped GEMM entry must use the descriptor element type",
                ));
            }
            if group.m == 0 || group.n == 0 || group.batch_count == 0 {
                continue;
            }
            if group.k == 0 {
                return Err(validation_error(
                    "cuBLAS grouped GEMM does not initialize a nonempty zero-K output",
                ));
            }
            validate(group, &lhs[index], &rhs[index], &out[index])?;
            pointer_count = pointer_count
                .checked_add(group.batch_count as usize)
                .ok_or_else(|| validation_error("grouped GEMM batch count overflow"))?;
            active.push(index);
        }
        if active.is_empty() {
            return Ok(());
        }
        for (position, &index) in active.iter().enumerate() {
            for &other in &active[position + 1..] {
                if resources_overlap(&out[index], &out[other])
                    || resources_overlap(&out[index], &lhs[other])
                    || resources_overlap(&out[index], &rhs[other])
                    || resources_overlap(&out[other], &lhs[index])
                    || resources_overlap(&out[other], &rhs[index])
                {
                    return Err(validation_error(
                        "grouped GEMM outputs may not overlap another group",
                    ));
                }
            }
        }
        let group_count: i32 = active
            .len()
            .try_into()
            .map_err(|_| validation_error("grouped GEMM has more than i32::MAX groups"))?;
        let staging_values = pointer_count
            .checked_mul(3)
            .ok_or_else(|| validation_error("grouped GEMM pointer count overflow"))?;

        let slot_index = match self
            .grouped_staging
            .iter()
            .position(|slot| slot.capacity >= staging_values && slot.available())
        {
            Some(index) => index,
            None => {
                let capacity = staging_values
                    .checked_next_power_of_two()
                    .ok_or_else(|| validation_error("grouped GEMM staging capacity overflow"))?;
                self.grouped_staging
                    .push(GroupedPointerStaging::new(capacity)?);
                self.grouped_staging.len() - 1
            }
        };
        let slot = &mut self.grouped_staging[slot_index];
        // CUDA graphs replay the captured host-to-device copy, so both the
        // pinned source and device destination addresses must remain stable.
        // Retaining a tiny slot per captured grouped launch provides that
        // lifetime without imposing synchronization on ordinary launches.
        // SAFETY: `stream` is the live execution stream resolved by the server.
        slot.captured = !matches!(
            unsafe { cudarc::driver::result::stream::is_capturing(stream) }.map_err(cuda_error)?,
            cudarc::driver::sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        );
        // SAFETY: the pinned allocation contains `capacity` u64 values and the
        // selected slot is no longer in use by a previous launch.
        let host =
            unsafe { core::slice::from_raw_parts_mut(slot.host.cast::<u64>(), slot.capacity) };

        let mut trans_a = Vec::with_capacity(active.len());
        let mut trans_b = Vec::with_capacity(active.len());
        let mut m = Vec::with_capacity(active.len());
        let mut n = Vec::with_capacity(active.len());
        let mut k = Vec::with_capacity(active.len());
        let mut lda = Vec::with_capacity(active.len());
        let mut ldb = Vec::with_capacity(active.len());
        let mut ldc = Vec::with_capacity(active.len());
        let mut group_sizes = Vec::with_capacity(active.len());
        let alpha = vec![1.0_f32; active.len()];
        let beta = vec![0.0_f32; active.len()];
        let elem_size = core::mem::size_of::<half::bf16>() as u64;
        let (a_values, rest) = host[..staging_values].split_at_mut(pointer_count);
        let (b_values, c_values) = rest.split_at_mut(pointer_count);
        let mut pointer = 0;

        for &index in &active {
            let group = &descriptor.groups[index];
            // cuBLAS is column-major. Swapping the row-major operands computes
            // D^T = rhs^T @ lhs^T without materialization.
            trans_a.push(operation(&group.rhs));
            trans_b.push(operation(&group.lhs));
            m.push(group.n as i32);
            n.push(group.m as i32);
            k.push(group.k as i32);
            lda.push(group.rhs.leading_dimension as i32);
            ldb.push(group.lhs.leading_dimension as i32);
            ldc.push(group.out.leading_dimension as i32);
            group_sizes.push(group.batch_count as i32);
            for batch in 0..group.batch_count as u64 {
                a_values[pointer] = batch_pointer(&rhs[index], &group.rhs, batch, elem_size)?;
                b_values[pointer] = batch_pointer(&lhs[index], &group.lhs, batch, elem_size)?;
                c_values[pointer] = batch_pointer(&out[index], &group.out, batch, elem_size)?;
                pointer += 1;
            }
        }
        debug_assert_eq!(pointer, pointer_count);

        // SAFETY: the host slice is pinned and remains untouched until the
        // completion event recorded below. The device allocation is large
        // enough for exactly `staging_values` pointers.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(slot.device, &host[..staging_values], stream)
        }
        .map_err(cuda_error)?;

        let handle = match self.grouped_handle {
            Some(handle) => handle,
            None => {
                let handle = blas::create_handle().map_err(grouped_cublas_error)?;
                self.grouped_handle = Some(handle);
                handle
            }
        };
        // SAFETY: the handle and CubeCL stream are live for the server.
        unsafe { blas::set_stream(handle, stream.cast()) }.map_err(grouped_cublas_error)?;
        let a_device = slot.device as *const *const c_void;
        let b_device = (slot.device + (pointer_count * core::mem::size_of::<u64>()) as u64)
            as *const *const c_void;
        let c_device = (slot.device + (2 * pointer_count * core::mem::size_of::<u64>()) as u64)
            as *const *mut c_void;
        // SAFETY: dimensions and leading dimensions were validated; all
        // pointer arrays reside in the staging device allocation and refer to
        // live CubeCL resources ordered on `stream`.
        let launch = unsafe {
            blas_sys::cublasGemmGroupedBatchedEx(
                handle,
                trans_a.as_ptr(),
                trans_b.as_ptr(),
                m.as_ptr(),
                n.as_ptr(),
                k.as_ptr(),
                alpha.as_ptr().cast::<c_void>(),
                a_device,
                blas_sys::cudaDataType_t::CUDA_R_16BF,
                lda.as_ptr(),
                b_device,
                blas_sys::cudaDataType_t::CUDA_R_16BF,
                ldb.as_ptr(),
                beta.as_ptr().cast::<c_void>(),
                c_device,
                blas_sys::cudaDataType_t::CUDA_R_16BF,
                ldc.as_ptr(),
                group_count,
                group_sizes.as_ptr(),
                blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            )
            .result()
        };
        // Record even when cuBLAS rejects the launch: the pointer upload is
        // already queued and must complete before the slot is reused.
        // SAFETY: `event` and `stream` are live.
        unsafe { cudarc::driver::result::event::record(slot.event, stream) }.map_err(cuda_error)?;
        slot.in_flight = true;
        launch.map_err(grouped_cublas_error)?;
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
        #[cfg(cuda_12050)]
        for staging in self.grouped_staging.drain(..) {
            staging.destroy();
        }
        if let Some(handle) = self.handle.take() {
            // SAFETY: this state uniquely owns the handle and destroys it once.
            if let Err(err) = unsafe { lt::destroy_handle(handle) } {
                log::warn!("Unable to destroy cublasLt handle: {err}");
            }
        }
        #[cfg(cuda_12050)]
        if let Some(handle) = self.grouped_handle.take() {
            // SAFETY: this state uniquely owns the handle and destroys it once.
            if let Err(err) = unsafe { blas::destroy_handle(handle) } {
                log::warn!("Unable to destroy cuBLAS grouped-GEMM handle: {err}");
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

#[cfg(cuda_12050)]
fn batch_pointer(
    resource: &GpuResource,
    matrix: &GemmMatrix,
    batch: u64,
    elem_size: u64,
) -> Result<u64, ServerError> {
    batch
        .checked_mul(matrix.batch_stride)
        .and_then(|offset| offset.checked_mul(elem_size))
        .and_then(|offset| resource.ptr.checked_add(offset))
        .ok_or_else(|| validation_error("grouped GEMM batch pointer overflow"))
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

#[cfg(cuda_12050)]
fn grouped_cublas_error(error: blas::CublasError) -> ServerError {
    ServerError::Generic {
        reason: format!("cuBLAS grouped GEMM error: {error:?}"),
        backtrace: BackTrace::capture(),
    }
}

#[cfg(cuda_12050)]
fn cuda_error(error: cudarc::driver::DriverError) -> ServerError {
    ServerError::Generic {
        reason: format!("CUDA grouped GEMM staging error: {error:?}"),
        backtrace: BackTrace::capture(),
    }
}
