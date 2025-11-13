use crate::components::{InvalidConfigError, MatmulIdent};
use crate::components::{
    MatmulElems, MatrixLayout,
    stage::{StageMemoryConfig, SwizzleMode, TilingLayout},
};
use crate::components::{MatmulPrecision, global::memory::GlobalIterator};
use crate::components::{global::GlobalConfig, stage::StageFamily};
use cubecl_core::ir::SemanticType;
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl};

#[cube]
/// A loading job represents a sequence of loading tasks.
/// Each task is the smallest unit of loading work:
/// one unit at one iteration, operating at a specific point within a read view.
/// The job holds shared information reused across read views and iterations.
/// By calling execute_task at strategic moments, one can hope to speed up the matmul.
pub trait LoadingJob<EG: Numeric, ES: Numeric, TL: TilingLayout, S: SyncStrategy>:
    CubeType + Copy + Clone
{
    type Stage: StageFamily;

    /// Execute the `task_id`th loading task
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut <Self::Stage as StageFamily>::Stage<ES, TL>,
        barrier: &mut S::Barrier,
        #[comptime] config: G,
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
    fn sync<MP: MatmulPrecision, G: GlobalConfig>(
        barrier: &mut Self::Barrier,
        #[comptime] config: G,
    );
}

/// Allows to verify configs are valid for a reader
pub trait LoadingValidation {
    /// Verify that configs are valid for a reader, otherwise return an error stating why
    fn check<C: GlobalConfig, R: Runtime>(
        client: &ComputeClient<R::Server>,
        config: &C,
        ident: MatmulIdent,
        dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError>;
}

/// Validates if [async barrier instructions](SemanticType::Barrier) is available on the current
/// device.
pub fn validate_async_barrier<R: Runtime>(
    client: &ComputeClient<R::Server>,
) -> Result<(), InvalidConfigError> {
    if !client
        .properties()
        .features
        .supports_type(SemanticType::Barrier)
    {
        return Err(Box::new(
            "Async barrier instructions are not available on the current device",
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

/// Validates if [tensor memory accelerator features](SemanticType::TensorMap) are available on the current
/// device.
pub fn validate_tma<R: Runtime>(
    client: &ComputeClient<R::Server>,
    config: StageMemoryConfig,
    ident: MatmulIdent,
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

    if dtypes.global(ident).size() != dtypes.stage(ident).size() {
        return Err(Box::new(
            "TMA requires stage and global types to be the same",
        ));
    }

    let row_size = match config.matrix_layout {
        MatrixLayout::RowMajor => config.elements_in_stage_col(),
        MatrixLayout::ColMajor => config.elements_in_stage_row(),
    };
    let row_bytes = row_size * dtypes.global(ident).size() as u32;

    let swizzle_span = match config.swizzle {
        SwizzleMode::None => return Ok(()),
        SwizzleMode::B32 => 32,
        SwizzleMode::B64 => 64,
        SwizzleMode::B128 => 128,
    };

    // Slightly tighter than the actual requirements, but simple enough and is always followed by
    // selection. Getting illegal memory access if this isn't followed for some reason.
    if row_bytes != swizzle_span {
        return Err(Box::new("Swizzling size must be equal to row size for TMA"));
    }

    Ok(())
}

/// Dummy trait implementation
pub struct NoLoadingValidation {}
impl LoadingValidation for NoLoadingValidation {
    fn check<C: GlobalConfig, R: Runtime>(
        _client: &ComputeClient<R::Server>,
        _config: &C,
        _ident: MatmulIdent,
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
