use crate::components::stage::TilingLayout;
use crate::components::{InvalidConfigError, MatmulProblem};
use crate::components::{
    MatmulElems, MatrixLayout,
    stage::{StageMemoryConfig, SwizzleMode},
};
use crate::components::{MatmulPrecision, global::memory::GlobalIterator};
use crate::components::{StageIdent, global::stride_align_bits};
use crate::components::{global::GlobalReaderConfig, stage::StageConfig};
use crate::components::{global::SharedGlobalMatmulConfig, stage::StageFamily};
use cubecl_core::ir::{BarrierLevel, OpaqueType, SemanticType};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

#[cube]
/// A loading job represents a sequence of loading tasks.
/// Each task is the smallest unit of loading work:
/// one unit at one iteration, operating at a specific point within a read view.
/// The job holds shared information reused across read views and iterations.
/// By calling execute_task at strategic moments, one can hope to speed up the matmul.
pub trait LoadingJob<EG: Numeric, ES: Numeric, TL: TilingLayout, S: SyncStrategy>:
    CubeType + Clone
{
    type Stage: StageFamily;

    /// Execute the `task_id`th loading task
    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut <Self::Stage as StageFamily>::Stage<ES, TL>,
        barrier: &mut S::Barrier,
        #[comptime] config: GlobalReaderConfig,
    );

    /// Get the number of tasks
    fn task_count(this: &Self) -> comptime_type!(u32);
}

/// A synchronization strategy determines the type of synchronization object, how to create it and
/// how to synchronize on it.
/// The sync strategy must match the one on both the LHS and RHS loading strategy.
#[cube]
pub trait SyncStrategy {
    type Barrier: CubeType + Clone;
    fn create_barrier() -> Self::Barrier;
    fn sync<MP: MatmulPrecision, S: StageConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] config: SharedGlobalMatmulConfig<S>,
    );
}

/// Allows to verify configs are valid for a reader
pub trait LoadingValidation {
    /// Verify that configs are valid for a reader, otherwise return an error stating why
    fn check<R: Runtime>(
        client: &ComputeClient<R>,
        problem: &MatmulProblem,
        config: &GlobalReaderConfig,
        dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError>;
}

/// Validates if [async barrier instructions](SemanticType::Barrier) is available on the current
/// device.
pub fn validate_async_barrier<R: Runtime>(
    client: &ComputeClient<R>,
) -> Result<(), InvalidConfigError> {
    if !client
        .properties()
        .features
        .supports_type(OpaqueType::Barrier(BarrierLevel::Cube))
    {
        return Err(Box::new(
            "Async barrier instructions are not available on the current device",
        ));
    }

    Ok(())
}

/// Validates if [async copy instructions](copy_async) is available on the current
/// device.
pub fn validate_async_copy<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    dtypes: &MatmulElems,
    config: &GlobalReaderConfig,
) -> Result<(), InvalidConfigError> {
    if !client.properties().features.copy_async {
        return Err(Box::new(
            "Async copy instructions are not available on the current device",
        ));
    }

    let dtype_global = dtypes.global(config.stage_ident.into());
    let dtype_stage = dtypes.stage(config.stage_ident.into());

    if dtype_global.size() != dtype_stage.size() {
        return Err(Box::new(
            "Async copy requires stage and global types to be the same",
        ));
    }

    if dtype_global.quantized && !dtype_stage.quantized {
        return Err(Box::new(
            "Async copy doesn't support dequantizing on global read",
        ));
    }

    if stride_align_bits(problem, dtypes, config.stage_ident.into()) < 4 {
        return Err(Box::new(
            "Async copy requires strides to be aligned to 16 bytes",
        ));
    }

    Ok(())
}

/// Validates if swizzling is disabled, for loaders that can't support it.
pub fn validate_noswizzle(config: StageMemoryConfig) -> Result<(), InvalidConfigError> {
    if config.swizzle != SwizzleMode::None {
        return Err(Box::new("This loader doesn't support swizzling"));
    }

    Ok(())
}

/// Validates if swizzling is valid with the line size, for sync readers that read in terms of full
/// lines
pub fn validate_swizzle_atom_size(
    config: StageMemoryConfig,
    ident: StageIdent,
    dtypes: &MatmulElems,
) -> Result<(), InvalidConfigError> {
    if config.swizzle == SwizzleMode::None {
        return Ok(());
    }

    let line_bytes = dtypes.stage(ident.into()).size() * config.line_size as usize;
    if line_bytes > config.swizzle.atom_size() {
        return Err(Box::new("Load atom can't be larger than swizzle atom"));
    }

    Ok(())
}

/// Validates if [tensor memory accelerator features](SemanticType::TensorMap) are available on the current
/// device.
pub fn validate_tma<R: Runtime>(
    client: &ComputeClient<R>,
    problem: &MatmulProblem,
    config: &GlobalReaderConfig,
    dtypes: &MatmulElems,
) -> Result<(), InvalidConfigError> {
    if !client
        .properties()
        .features
        .supports_type(SemanticType::TensorMap)
    {
        return Err(Box::new(
            "Tensor memory accelerator features are not available on the current device",
        ));
    }

    let dtype_global = dtypes.global(config.stage_ident.into());
    let dtype_stage = dtypes.stage(config.stage_ident.into());

    if dtype_global.size() != dtype_stage.size() {
        return Err(Box::new(
            "TMA requires stage and global types to be the same",
        ));
    }

    if dtype_global.quantized && !dtype_stage.quantized {
        return Err(Box::new("TMA doesn't support dequantizing on global read"));
    }

    if stride_align_bits(problem, dtypes, config.stage_ident.into()) < 4 {
        return Err(Box::new("TMA requires strides to be aligned to 16 bytes"));
    }

    if matches!(config.smem_config.swizzle, SwizzleMode::None) {
        return Ok(());
    }

    let row_size = match config.smem_config.matrix_layout {
        MatrixLayout::RowMajor => config.smem_config.elements_per_stage_along_col(),
        MatrixLayout::ColMajor => config.smem_config.elements_per_stage_along_row(),
    };
    let row_bytes = row_size * dtypes.global(config.stage_ident.into()).size() as u32;

    // Slightly tighter than the actual requirements, but simple enough and is always followed by
    // selection. Getting illegal memory access if this isn't followed for some reason.
    if row_bytes as usize != config.smem_config.swizzle.span_size() {
        return Err(Box::new("Swizzling size must be equal to row size for TMA"));
    }

    Ok(())
}

/// Dummy trait implementation
pub struct NoLoadingValidation {}
impl LoadingValidation for NoLoadingValidation {
    fn check<R: Runtime>(
        _client: &ComputeClient<R>,
        _problem: &MatmulProblem,
        _config: &GlobalReaderConfig,
        _dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Controls bounds checking for reader operations.
///
/// This **does not** disable tensor read bounds checks.
/// It only affects checks for whether the reader loads more data than allowed
/// at each global matmul iteration.
pub enum ReaderMode {
    /// Enforces compile-time validation of balanced workloads across units.
    /// Restricts valid combinations of tile shape, count, and line size.
    Strict,
    /// Inserts runtime checks only when an out-of-bounds access will occur.
    /// May reduce performance if workloads are imbalanced.
    #[default]
    Relaxed,
}
