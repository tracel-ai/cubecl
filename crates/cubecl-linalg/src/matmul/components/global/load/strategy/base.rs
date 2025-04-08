use crate::matmul::components::global::load::{
    AsyncBufferLoadingStrategy, AsyncFullLoadingStrategy, SyncBufferLoadingStrategy,
    SyncFullLoadingStrategy,
};
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CopyMechanism, GlobalConfig, Quantization};
use crate::matmul::components::stage::Stage;
use crate::matmul::components::{InputIdent, MatmulPrecision};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;

#[cube]
/// A loading job represents a group of loading tasks.
/// Each task is the smallest unit of loading work:
/// one thread at one iteration, operating at a specific point within a read view.
/// The job holds shared information reused across read views and iterations.
/// By calling execute_task at strategic moments, one can hope to speed up the matmul.
pub trait LoadingJob<MP: MatmulPrecision>: CubeType + Copy + Clone {
    type LoadingJobConfig: LoadingJobConfig<MP, Self>;

    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        read_view: &TensorReader<MP::EI>,
        #[comptime] config: G,
    );
}

pub trait LoadingJobConfig<MP: MatmulPrecision, LJ: LoadingJob<MP>> {
    fn len(job: &LJ) -> u32;

    fn __expand_len(
        context: &mut cubecl::prelude::Scope,
        job: <LJ as cubecl::prelude::CubeType>::ExpandType,
    ) -> u32;
}

type JobConfig<MP: MatmulPrecision, Job> = <Job as LoadingJob<MP>>::LoadingJobConfig;

#[cube]
pub(crate) fn default_sync_full_load<
    LS: SyncFullLoadingStrategy,
    MP: MatmulPrecision,
    G: GlobalConfig,
>(
    read_view: &TensorReader<MP::EI>,
    stage: Stage<MP::ES, LS::TilingLayout>,
    quantization: CubeOption<Quantization<MP>>,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    let mut job = LS::job::<MP, G>(stage, quantization, input_ident, config);

    let len = JobConfig::<MP, LS::Job<MP>>::len(&job);
    for task_id in 0..len {
        LS::Job::execute_task::<G>(&mut job, task_id, read_view, config);
    }
}

#[cube]
pub(crate) fn default_sync_buffer_load<
    LS: SyncBufferLoadingStrategy,
    MP: MatmulPrecision,
    G: GlobalConfig,
>(
    read_view: &TensorReader<MP::EI>,
    stage: Stage<MP::ES, LS::TilingLayout>,
    quantization: CubeOption<Quantization<MP>>,
    #[comptime] buffer_index: u32,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    let mut job = LS::job::<MP, G>(stage, quantization, buffer_index, input_ident, config);

    let len = JobConfig::<MP, LS::Job<MP>>::len(&job);
    for task_id in 0..len {
        LS::Job::execute_task::<G>(&mut job, task_id, read_view, config);
    }
}

#[cube]
pub(crate) fn default_async_full_load<
    LS: AsyncFullLoadingStrategy,
    MP: MatmulPrecision,
    G: GlobalConfig,
    CM: CopyMechanism<MP::ES>,
>(
    read_view: &TensorReader<MP::EI>,
    stage: Stage<MP::ES, LS::TilingLayout>,
    mechanism: CM,
    quantization: CubeOption<Quantization<MP>>,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    let mut job = LS::job::<MP, CM, G>(stage, mechanism, quantization, input_ident, config);

    let len = JobConfig::<MP, LS::Job<MP, CM>>::len(&job);
    for task_id in 0..len {
        LS::Job::execute_task::<G>(&mut job, task_id, read_view, config);
    }
}

#[cube]
pub(crate) fn default_async_buffer_load<
    LS: AsyncBufferLoadingStrategy,
    MP: MatmulPrecision,
    G: GlobalConfig,
    CM: CopyMechanism<MP::ES>,
>(
    read_view: &TensorReader<MP::EI>,
    stage: Stage<MP::ES, LS::TilingLayout>,
    mechanism: CM,
    quantization: CubeOption<Quantization<MP>>,
    #[comptime] buffer_index: u32,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    let mut job = LS::job::<MP, CM, G>(
        stage,
        mechanism,
        quantization,
        buffer_index,
        input_ident,
        config,
    );

    let len = JobConfig::<MP, LS::Job<MP, CM>>::len(&job);
    for task_id in 0..len {
        LS::Job::execute_task::<G>(&mut job, task_id, read_view, config);
    }
}
