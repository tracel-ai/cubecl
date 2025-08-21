use crate::components::global::memory::TensorReader;
use crate::components::global::{CopyMechanism, GlobalConfig};
use crate::components::stage::{StageMemory, TilingLayout};
use crate::components::{InputPrecision, InvalidConfigError, MatmulIdent};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
/// A loading job represents a sequence of loading tasks.
/// Each task is the smallest unit of loading work:
/// one unit at one iteration, operating at a specific point within a read view.
/// The job holds shared information reused across read views and iterations.
/// By calling execute_task at strategic moments, one can hope to speed up the matmul.
pub trait LoadingJob<IP: InputPrecision, TL: TilingLayout>: CubeType + Copy + Clone {
    /// Execute the `task_id`th loading task
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        #[comptime] task_id: u32,
        tensor_reader: &TensorReader<IP::Global>,
        stage_memory: &mut StageMemory<IP::Stage, TL>,
        #[comptime] config: G,
    );

    /// Get the number of tasks
    fn task_count(this: &Self) -> comptime_type!(u32);
}

#[cube]
/// A loading job represents a sequence of loading tasks.
/// Each task is the smallest unit of loading work:
/// one unit at one iteration, operating at a specific point within a read view.
/// The job holds shared information reused across read views and iterations.
/// By calling execute_task at strategic moments, one can hope to speed up the matmul.
pub trait AsyncLoadingJob<IP: InputPrecision, TL: TilingLayout>: CubeType + Copy + Clone {
    /// Execute the `task_id`th loading task
    fn execute_task<CM: CopyMechanism<IP::Stage>, G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<IP::Global>,
        stage_memory: &mut StageMemory<IP::Stage, TL>,
        mechanism: &CM,
        #[comptime] config: G,
    );

    /// Get the number of tasks
    fn task_count(this: &Self) -> comptime_type!(u32);
}

/// Allows to verify configs are valid for a loader
pub trait LoadingValidation {
    /// Verify that configs are valid for a loader, otherwise return an error stating why
    fn check<C: GlobalConfig>(config: &C, ident: MatmulIdent) -> Result<(), InvalidConfigError>;
}

/// Dummy trait implementation
pub struct NoLoadingValidation {}
impl LoadingValidation for NoLoadingValidation {
    fn check<C: GlobalConfig>(_config: &C, _ident: MatmulIdent) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[derive(Default, Copy, Clone, Debug, Hash, PartialEq, Eq)]
/// Controls bounds checking for loader operations.
///
/// This **does not** disable tensor read bounds checks.
/// It only affects checks for whether the loader loads more data than allowed
/// at each global matmul iteration.
pub enum LoaderMode {
    /// Enforces compile-time validation of balanced workloads across units.
    /// Restricts valid combinations of tile shape, count, and line size.
    Strict,
    /// Inserts runtime checks only when an out-of-bounds access will occur.
    /// May reduce performance if workloads are imbalanced.
    #[default]
    Relaxed,
}
