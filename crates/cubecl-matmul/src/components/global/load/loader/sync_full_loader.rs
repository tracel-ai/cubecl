use std::marker::PhantomData;

use crate::components::global::GlobalConfig;
use crate::components::global::Quantization;
use crate::components::global::global_memory::TensorReader;
use crate::components::global::load::LoadingJob;
use crate::components::global::load::LoadingValidation;
use crate::components::global::load::StageIdent;
use crate::components::global::load::TaskCounter;
use crate::components::global::multi_stage::Job;
use crate::components::global::multi_stage::JobExecutor;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::stage::FullStageToTileReader;
use crate::components::stage::StageMemory;
use crate::components::stage::TilingLayout;
use crate::components::{InputIdent, MatmulPrecision};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, CubeOptionExpand};

#[cube]
/// A strategy for synchronously loading a full stage memory.
pub trait SyncFullLoadingStrategy:
    'static + Send + Sync + Clone + LoadingValidation + LoadMaxRoundPlaneCount
{
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;

    /// The [LoadingJob] for this strategy.
    type Job<MP: MatmulPrecision>: LoadingJob<MP, Self::TilingLayout>;

    /// Returns the job with preliminary calculations done.
    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP>;
}

#[derive(Clone, CubeType)]
/// Loads the entire stage memory using synchronous data movement operations.
///
/// A complete load is referred to as a `Job`, which is divided into `Tasks`—
/// each Task represents a single data transfer for a specific unit
pub struct SyncFullLoader<MP: MatmulPrecision, G: GlobalConfig, L: SyncFullLoadingStrategy> {
    tensor_reader: TensorReader<MP::EI>,
    stage_memory: StageMemory<MP::ES, L::TilingLayout>,
    loading_job: CubeOption<L::Job<MP>>,
    quantization: CubeOption<Quantization<MP>>,
    #[cube(comptime)]
    input_ident: InputIdent,
    #[cube(comptime)]
    _phantom: PhantomData<(G, L)>,
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncFullLoadingStrategy> SyncFullLoader<MP, G, L> {
    /// Create a new SyncFullLoader
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
            StageMemory::new::<G::StageConfig>(1u32, input_ident.as_ident(), config.stage_config());
        let tensor_reader = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        let loading_job = match config.precompute_job() {
            true => CubeOption::new_Some(L::new_job::<MP, G>(input_ident, config)),
            false => CubeOption::new_None(),
        };

        SyncFullLoader::<MP, G, L> {
            tensor_reader,
            stage_memory,
            loading_job,
            quantization,
            input_ident,
            _phantom: PhantomData::<(G, L)>,
        }
    }

    /// Give a reader to the loaded stage memory.
    pub fn reader(this: &Self) -> FullStageToTileReader<MP::ES, L::TilingLayout> {
        FullStageToTileReader::new(this.stage_memory, this.input_ident)
    }

    /// Advance the view over global memory along the k dimension by a specified offset, `k_offset`.
    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_reader.update_view(k_offset, this.input_ident);
    }

    /// Accomplish the entire job of filling the stage memory
    pub fn fill_stage(this: &mut Self, #[comptime] config: G) {
        let mut loading_job = match this.loading_job {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<MP, G>(this.input_ident, config),
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
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncFullLoadingStrategy> JobExecutor<G>
    for SyncFullLoader<MP, G, L>
{
    type JobIterator = SyncFullLoaderJobIterator<MP, L>;

    fn create_job_iterator(
        this: &Self,
        #[comptime] _stage_ident: StageIdent,
        #[comptime] config: G,
    ) -> Self::JobIterator {
        let job = match this.loading_job {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<MP, G>(this.input_ident, config),
        };

        let num_tasks = L::Job::task_count(&job);

        SyncFullLoaderJobIterator::<MP, L> {
            job,
            num_tasks,
            current: ComptimeCell::new(TaskCounter { counter: 0u32 }),
        }
    }

    fn execute_task(
        this: &mut Self,
        job_iterator: &mut SyncFullLoaderJobIterator<MP, L>,
        #[comptime] config: G,
    ) {
        let task_id = job_iterator.current.read().counter;

        L::Job::<MP>::execute_task::<G>(
            &mut job_iterator.job,
            task_id,
            &this.tensor_reader,
            &mut this.stage_memory,
            &this.quantization,
            config,
        );

        job_iterator.current.store(TaskCounter {
            counter: comptime!(task_id + 1u32),
        });
    }

    fn execute_all_remaining_tasks(
        this: &mut Self,
        job_iterator: &mut Self::JobIterator,
        #[comptime] config: G,
    ) {
        let task_counter = job_iterator.current.read().counter;

        let mut task_id = comptime![task_counter];

        #[allow(clippy::explicit_counter_loop)]
        #[unroll]
        for _ in task_counter..job_iterator.num_tasks {
            L::Job::<MP>::execute_task::<G>(
                &mut job_iterator.job,
                task_id,
                &this.tensor_reader,
                &mut this.stage_memory,
                &this.quantization,
                config,
            );
            comptime![task_id += 1];
        }

        job_iterator.current.store(TaskCounter {
            counter: comptime!(job_iterator.num_tasks),
        });
    }

    fn execute_whole_job(
        this: &mut Self,
        #[comptime] stage_ident: StageIdent,
        #[comptime] config: G,
    ) {
        Self::execute_all_remaining_tasks(
            this,
            &mut Self::create_job_iterator(this, stage_ident, config),
            config,
        );
    }
}

#[derive(CubeType)]
/// A comptime iterator over a job for sync full loader
pub struct SyncFullLoaderJobIterator<MP: MatmulPrecision, L: SyncFullLoadingStrategy> {
    job: L::Job<MP>,
    #[cube(comptime)]
    pub num_tasks: u32,
    pub current: ComptimeCell<TaskCounter>,
}

#[cube]
impl<MP: MatmulPrecision, L: SyncFullLoadingStrategy> Job for SyncFullLoaderJobIterator<MP, L> {
    fn current(this: &Self) -> comptime_type!(u32) {
        this.current.read().counter
    }

    fn num_tasks(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}
