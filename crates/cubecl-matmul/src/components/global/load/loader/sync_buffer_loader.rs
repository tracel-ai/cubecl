use std::marker::PhantomData;

use super::StageIdent;
use super::TaskCounter;
use crate::components::InputIdent;
use crate::components::MatmulPrecision;
use crate::components::global::GlobalConfig;
use crate::components::global::Quantization;
use crate::components::global::load::LoadingJob;
use crate::components::global::load::LoadingValidation;
use crate::components::global::multi_stage::Job;
use crate::components::global::multi_stage::JobExecutor;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::global::global_memory::TensorReader;
use crate::components::stage::PartialStageToTileReader;
use crate::components::stage::StageMemory;
use crate::components::stage::TilingLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, CubeOptionExpand};

#[cube]
/// A strategy for synchronously loading a buffer (partial stage), either eagerly or as a deferred job.
pub trait SyncBufferLoadingStrategy:
    'static + Send + Sync + Clone + LoadingValidation + LoadMaxRoundPlaneCount
{
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;

    /// The [LoadingJob] for this strategy.
    type Job<MP: MatmulPrecision>: LoadingJob<MP, Self::TilingLayout>;

    /// Returns the job with preliminary calculations done.
    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] buffer_index: u32,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP>;
}

#[derive(Clone, CubeType)]
pub struct SyncBufferLoader<MP: MatmulPrecision, G: GlobalConfig, L: SyncBufferLoadingStrategy> {
    tensor_reader: TensorReader<MP::EI>,
    stage_memory: StageMemory<MP::ES, L::TilingLayout>,
    loading_job: CubeOption<(L::Job<MP>, L::Job<MP>)>,
    quantization: CubeOption<Quantization<MP>>,
    #[cube(comptime)]
    input_ident: InputIdent,
    #[cube(comptime)]
    _config: PhantomData<G>,
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncBufferLoadingStrategy>
    SyncBufferLoader<MP, G, L>
{
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self {
        let stage_memory =
            StageMemory::new::<G::StageConfig>(2u32, input_ident.as_ident(), config.stage_config());
        let tensor_reader = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        let loading_job = match config.precompute_job() {
            true => CubeOption::new_Some((
                L::new_job::<MP, G>(0u32, input_ident, config),
                L::new_job::<MP, G>(1u32, input_ident, config),
            )),
            false => CubeOption::new_None(),
        };

        SyncBufferLoader::<MP, G, L> {
            tensor_reader,
            stage_memory,
            loading_job,
            quantization,
            input_ident,
            _config: PhantomData::<G>,
        }
    }

    pub fn reader(
        this: &Self,
        #[comptime] stage_ident: StageIdent,
    ) -> PartialStageToTileReader<MP::ES, L::TilingLayout> {
        PartialStageToTileReader::new(this.stage_memory, stage_ident, this.input_ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_reader.update_view(k_offset, this.input_ident);
    }

    pub fn fill_stage(this: &mut Self, #[comptime] stage_ident: StageIdent, #[comptime] config: G) {
        let mut loading_job = match this.loading_job {
            CubeOption::Some(job) => match stage_ident {
                StageIdent::A => job.0,
                StageIdent::B => job.1,
            },
            CubeOption::None => match stage_ident {
                StageIdent::A => L::new_job::<MP, G>(0u32, this.input_ident, config),
                StageIdent::B => L::new_job::<MP, G>(1u32, this.input_ident, config),
            },
        };

        let len = L::Job::task_count(&loading_job);

        let mut task_id = comptime![0u32];

        #[allow(clippy::explicit_counter_loop)]
        #[unroll]
        for _ in 0..len {
            L::Job::<MP>::execute_task::<G>(
                &mut loading_job,
                task_id,
                &this.tensor_reader,
                &mut this.stage_memory,
                &this.quantization,
                config,
            );
            comptime![task_id += 1];
        }
    }
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncBufferLoadingStrategy> JobExecutor<G>
    for SyncBufferLoader<MP, G, L>
{
    type Job = SyncBufferLoaderJob<MP, L>;

    fn create_job(
        this: &Self,
        #[comptime] stage_ident: StageIdent,
        #[comptime] config: G,
    ) -> Self::Job {
        let loading = match this.loading_job {
            CubeOption::Some(job) => match stage_ident {
                StageIdent::A => job.0,
                StageIdent::B => job.1,
            },
            CubeOption::None => match stage_ident {
                StageIdent::A => L::new_job::<MP, G>(0u32, this.input_ident, config),
                StageIdent::B => L::new_job::<MP, G>(1u32, this.input_ident, config),
            },
        };

        let num_tasks = L::Job::task_count(&loading);

        SyncBufferLoaderJob::<MP, L> {
            loading,
            num_tasks,
            current: ComptimeCell::new(TaskCounter { counter: 0u32 }),
        }
    }

    fn execute_task(this: &mut Self, job: &mut SyncBufferLoaderJob<MP, L>, #[comptime] config: G) {
        let task_id = job.current.read().counter;

        L::Job::<MP>::execute_task::<G>(
            &mut job.loading,
            task_id,
            &this.tensor_reader,
            &mut this.stage_memory,
            &this.quantization,
            config,
        );

        job.current.store(TaskCounter {
            counter: comptime!(task_id + 1u32),
        });
    }

    fn execute_all_remaining_tasks(this: &mut Self, job: &mut Self::Job, #[comptime] config: G) {
        let task_counter = job.current.read().counter;

        let mut task_id = comptime![task_counter];

        #[allow(clippy::explicit_counter_loop)]
        #[unroll]
        for _ in task_counter..job.num_tasks {
            L::Job::<MP>::execute_task::<G>(
                &mut job.loading,
                task_id,
                &this.tensor_reader,
                &mut this.stage_memory,
                &this.quantization,
                config,
            );
            comptime![task_id += 1];
        }

        job.current.store(TaskCounter {
            counter: comptime!(job.num_tasks),
        });
    }

    fn execute_whole_job(
        this: &mut Self,
        #[comptime] stage_ident: StageIdent,
        #[comptime] config: G,
    ) {
        Self::execute_all_remaining_tasks(
            this,
            &mut Self::create_job(this, stage_ident, config),
            config,
        );
    }
}

#[derive(CubeType)]
pub struct SyncBufferLoaderJob<MP: MatmulPrecision, L: SyncBufferLoadingStrategy> {
    loading: L::Job<MP>,
    #[cube(comptime)]
    pub num_tasks: u32,
    pub current: ComptimeCell<TaskCounter>,
}

#[cube]
impl<MP: MatmulPrecision, L: SyncBufferLoadingStrategy> Job for SyncBufferLoaderJob<MP, L> {
    fn current(this: &Self) -> comptime_type!(u32) {
        this.current.read().counter
    }

    fn num_tasks(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}
