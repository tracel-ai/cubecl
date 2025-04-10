use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CopyMechanism, GlobalConfig};
use crate::matmul::components::stage::{Stage, TilingLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
/// A loading job represents a group of loading tasks.
/// Each task is the smallest unit of loading work:
/// one thread at one iteration, operating at a specific point within a read view.
/// The job holds shared information reused across read views and iterations.
/// By calling execute_task at strategic moments, one can hope to speed up the matmul.
pub trait LoadingJob<MP: MatmulPrecision, TL: TilingLayout>: CubeType + Copy + Clone {
    type LoadingJobConfig: LoadingJobConfig<MP, TL, Self>;

    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, TL>,
        #[comptime] config: G,
    );
}

#[cube]
pub trait AsyncLoadingJob<MP: MatmulPrecision, TL: TilingLayout>: CubeType + Copy + Clone {
    type LoadingJobConfig: AsyncLoadingJobConfig<MP, TL, Self>;

    fn execute_task<CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, TL>,
        mechanism: &CM,
        #[comptime] config: G,
    );
}

pub trait LoadingJobConfig<MP: MatmulPrecision, TL: TilingLayout, LJ: LoadingJob<MP, TL>> {
    fn len(job: &LJ) -> u32;

    fn __expand_len(
        context: &mut cubecl::prelude::Scope,
        job: <LJ as cubecl::prelude::CubeType>::ExpandType,
    ) -> u32;
}

pub trait AsyncLoadingJobConfig<MP: MatmulPrecision, TL: TilingLayout, LJ: AsyncLoadingJob<MP, TL>>
{
    fn len(job: &LJ) -> u32;

    fn __expand_len(
        context: &mut cubecl::prelude::Scope,
        job: <LJ as cubecl::prelude::CubeType>::ExpandType,
    ) -> u32;
}

pub type JobConfig<MP, TL, Job> = <Job as LoadingJob<MP, TL>>::LoadingJobConfig;
pub type AsyncJobConfig<MP, TL, Job> = <Job as AsyncLoadingJob<MP, TL>>::LoadingJobConfig;
