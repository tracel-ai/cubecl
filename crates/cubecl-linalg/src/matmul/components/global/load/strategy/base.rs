use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CopyMechanism, GlobalConfig};
use crate::matmul::components::stage::{Stage, TilingLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
fn adsf() -> comptime_type!(f32) {
    4.5f32
}

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
        stage: &mut Stage<MP::ES, TL>,
        #[comptime] config: G,
    );

    fn len(this: &Self) -> comptime_type!(u32);
}

#[cube]
pub trait AsyncLoadingJob<MP: MatmulPrecision, TL: TilingLayout>: CubeType + Copy + Clone {
    fn execute_task<CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, TL>,
        mechanism: &CM,
        #[comptime] config: G,
    );

    fn len(this: &Self) -> comptime_type!(u32);
}
