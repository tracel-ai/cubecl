use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CopyMechanism, GlobalConfig, Quantization};
use crate::matmul::components::stage::{StageMemory, TilingLayout};
use crate::matmul::components::{Ident, InvalidConfigError, MatmulPrecision};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;

#[cube]
/// A loading job represents a group of loading tasks.
/// Each task is the smallest unit of loading work:
/// one thread at one iteration, operating at a specific point within a read view.
/// The job holds shared information reused across read views and iterations.
/// By calling execute_task at strategic moments, one can hope to speed up the matmul.
pub trait LoadingJob<MP: MatmulPrecision, TL: TilingLayout>: CubeType + Copy + Clone {
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage_memory: &mut StageMemory<MP::ES, TL>,
        quantization: &CubeOption<Quantization<MP>>,
        #[comptime] config: G,
    );

    fn task_count(this: &Self) -> comptime_type!(u32);
}

#[cube]
pub trait AsyncLoadingJob<MP: MatmulPrecision, TL: TilingLayout>: CubeType + Copy + Clone {
    fn execute_task<CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage_memory: &mut StageMemory<MP::ES, TL>,
        mechanism: &CM,
        #[comptime] config: G,
    );

    fn task_count(this: &Self) -> comptime_type!(u32);
}

pub trait LoadingValidation {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError>;
}

/// Trait for selecting loader bounds check strategy.
pub trait LoaderCheck: Clone + Copy + Sync + Send + 'static {
    fn to_level() -> LoaderCheckLevel;
}

/// Marker type for strict compile-time enforcement (no runtime check).
#[derive(Clone, Copy, Debug)]
pub struct ComptimeCheck;
impl LoaderCheck for ComptimeCheck {
    fn to_level() -> LoaderCheckLevel {
        LoaderCheckLevel::Comptime
    }
}

/// Marker type for fallback to runtime check if needed.
#[derive(Clone, Copy, Debug)]
pub struct RuntimeCheck;
impl LoaderCheck for RuntimeCheck {
    fn to_level() -> LoaderCheckLevel {
        LoaderCheckLevel::Runtime
    }
}

#[derive(Clone, Eq, PartialEq, Copy)]
/// Controls bounds checking for loader operations.
///
/// This **does not** disable tensor read bounds checks.
/// It only affects checks for whether the loader loads more data than allowed
/// at each global matmul iteration.
pub enum LoaderCheckLevel {
    /// Panics at compile time if an out-of-bounds access will occur.
    /// No runtime checks are added.
    Comptime,
    /// Adds a runtime check if and only if an out-of-bounds access will occur.
    Runtime,
}
