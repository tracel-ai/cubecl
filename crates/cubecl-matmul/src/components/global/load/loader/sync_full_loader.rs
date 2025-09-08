use std::marker::PhantomData;

use crate::components::InputPrecision;
use crate::components::MatmulIdent;
use crate::components::global::GlobalConfig;
use crate::components::global::load::LoadingJob;
use crate::components::global::load::LoadingValidation;
use crate::components::global::load::StageBuffer;
use crate::components::global::load::TaskCounter;
use crate::components::global::memory::TensorReader;
use crate::components::global::multi_stage::JobExecutor;
use crate::components::global::multi_stage::JobIterator;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::stage::FullStageToTileReader;
use crate::components::stage::StageMemory;
use crate::components::stage::TilingLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords3d},
};

#[cube]
/// A strategy for synchronously loading a full stage memory.
pub trait SyncFullLoadingStrategy:
    'static + Send + Sync + Clone + LoadingValidation + LoadMaxRoundPlaneCount
{
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;

    /// The [LoadingJob] for this strategy.
    type Job<IP: InputPrecision>: LoadingJob<IP, Self::TilingLayout>;

    /// Returns the job with preliminary calculations done.
    fn new_job<IP: InputPrecision, G: GlobalConfig>(
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self::Job<IP>;
}

#[derive(Clone, CubeType)]
/// Loads the entire stage memory using synchronous data movement operations.
///
/// A complete load is referred to as a `Job`, which is divided into `Tasks`—
/// each Task represents a single data transfer for a specific unit
pub struct SyncFullLoader<IP: InputPrecision, G: GlobalConfig, L: SyncFullLoadingStrategy> {
    tensor_reader: TensorReader<IP::Global>,
    stage_memory: StageMemory<IP::Stage, L::TilingLayout>,
    loading_job: CubeOption<L::Job<IP>>,
    #[cube(comptime)]
    ident: MatmulIdent,
    #[cube(comptime)]
    _phantom: PhantomData<(G, L)>,
}

#[cube]
impl<IP: InputPrecision, G: GlobalConfig, L: SyncFullLoadingStrategy> SyncFullLoader<IP, G, L> {
    /// Create a new SyncFullLoader
    pub fn new(
        tensor: View<Line<IP::Global>, Coords3d>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self {
        let stage_memory = StageMemory::new::<G::StageMemoryConfig>(
            1u32,
            comptime!(ident.into_stage()),
            config.stage_memory_config(),
        );
        let tensor_reader = TensorReader::new(tensor, (batch_offset, x_offset, y_offset));

        let loading_job = match config.precompute_job() {
            true => CubeOption::new_Some(L::new_job::<IP, G>(ident, config)),
            false => CubeOption::new_None(),
        };

        SyncFullLoader::<IP, G, L> {
            tensor_reader,
            stage_memory,
            loading_job,
            ident,
            _phantom: PhantomData::<(G, L)>,
        }
    }

    /// Give a reader to the loaded stage memory.
    pub fn reader(this: &Self) -> FullStageToTileReader<IP::Stage, L::TilingLayout> {
        FullStageToTileReader::new(this.stage_memory, comptime!(this.ident.into_stage()))
    }

    /// Advance the view over global memory along the k dimension by a specified offset, `k_offset`.
    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_reader.update_view(k_offset, this.ident);
    }

    /// Accomplish the entire job of filling the stage memory
    pub fn fill_stage(this: &mut Self, #[comptime] config: G) {
        let mut loading_job = match this.loading_job {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<IP, G>(this.ident, config),
        };

        let len = L::Job::task_count(&loading_job);
        let mut task_id = comptime![0u32];

        #[allow(clippy::explicit_counter_loop)]
        #[unroll]
        for _ in 0..len {
            L::Job::<IP>::execute_task::<G>(
                &mut loading_job,
                task_id,
                &this.tensor_reader,
                &mut this.stage_memory,
                config,
            );
            comptime![task_id += 1];
        }
    }
}

#[cube]
impl<IP: InputPrecision, G: GlobalConfig, L: SyncFullLoadingStrategy> JobExecutor<G>
    for SyncFullLoader<IP, G, L>
{
    type JobIterator = SyncFullLoaderJobIterator<IP, L>;

    fn create_job_iterator(
        this: &Self,
        #[comptime] _stage_buffer: StageBuffer,
        #[comptime] config: G,
    ) -> Self::JobIterator {
        let job = match this.loading_job {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<IP, G>(this.ident, config),
        };

        let num_tasks = L::Job::task_count(&job);

        SyncFullLoaderJobIterator::<IP, L> {
            job,
            num_tasks,
            current: ComptimeCell::new(TaskCounter { counter: 0u32 }),
        }
    }

    fn execute_task(
        this: &mut Self,
        job_iterator: &mut SyncFullLoaderJobIterator<IP, L>,
        #[comptime] config: G,
    ) {
        let task_id = job_iterator.current.read().counter;

        L::Job::<IP>::execute_task::<G>(
            &mut job_iterator.job,
            task_id,
            &this.tensor_reader,
            &mut this.stage_memory,
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
            L::Job::<IP>::execute_task::<G>(
                &mut job_iterator.job,
                task_id,
                &this.tensor_reader,
                &mut this.stage_memory,
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
        #[comptime] stage_buffer: StageBuffer,
        #[comptime] config: G,
    ) {
        Self::execute_all_remaining_tasks(
            this,
            &mut Self::create_job_iterator(this, stage_buffer, config),
            config,
        );
    }
}

#[derive(CubeType)]
/// A comptime iterator over a job for sync full loader
pub struct SyncFullLoaderJobIterator<IP: InputPrecision, L: SyncFullLoadingStrategy> {
    job: L::Job<IP>,
    #[cube(comptime)]
    pub num_tasks: u32,
    pub current: ComptimeCell<TaskCounter>,
}

#[cube]
impl<IP: InputPrecision, L: SyncFullLoadingStrategy> JobIterator
    for SyncFullLoaderJobIterator<IP, L>
{
    fn current(this: &Self) -> comptime_type!(u32) {
        this.current.read().counter
    }

    fn num_tasks(this: &Self) -> comptime_type!(u32) {
        this.num_tasks
    }
}
