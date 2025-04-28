use std::marker::PhantomData;

use super::BufferId;
use crate::matmul::components::InputIdent;
use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::GlobalConfig;
use crate::matmul::components::global::LoadingValidation;
use crate::matmul::components::global::Quantization;
use crate::matmul::components::global::load::LoadingJob;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::BufferReader;
use crate::matmul::components::stage::StageMemory;
use crate::matmul::components::stage::TilingLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

#[cube]
/// A strategy for synchronously loading a buffer (partial stage), either eagerly or as a deferred job.
pub trait SyncBufferLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
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
    stage: StageMemory<MP::ES, L::TilingLayout>,
    loading_job: CubeOption<L::Job<MP>>,
    quantization: CubeOption<Quantization<MP>>,
    #[cube(comptime)]
    buffer_id: BufferId,
    #[cube(comptime)]
    input_ident: InputIdent,
    #[cube(comptime)]
    _phantom: PhantomData<G>,
}

#[cube]
impl<MP: MatmulPrecision, G: GlobalConfig, L: SyncBufferLoadingStrategy>
    SyncBufferLoader<MP, G, L>
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tensor_reader: TensorReader<MP::EI>,
        stage: StageMemory<MP::ES, L::TilingLayout>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] buffer_id: BufferId,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self {
        let loading_job = match config.precompute_job() {
            true => CubeOption::new_Some(L::new_job::<MP, G>(
                comptime!(buffer_id.to_index()),
                input_ident,
                config,
            )),
            false => CubeOption::new_None(),
        };

        SyncBufferLoader::<MP, G, L> {
            tensor_reader,
            stage,
            loading_job,
            quantization,
            buffer_id,
            input_ident,
            _phantom: PhantomData::<G>,
        }
    }

    pub fn reader(this: &Self) -> BufferReader<MP::ES, L::TilingLayout> {
        BufferReader::new(this.stage, this.buffer_id, this.input_ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_reader.update_view(k_offset, this.input_ident);
    }

    pub fn fill_stage(this: &mut Self, #[comptime] config: G) {
        let mut loading_job = match this.loading_job {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<MP, G>(
                comptime!(this.buffer_id.to_index()),
                this.input_ident,
                config,
            ),
        };

        let len = L::Job::task_count(&loading_job);
        for task_id in 0..len {
            L::Job::<MP>::execute_task::<G>(
                &mut loading_job,
                task_id,
                &this.tensor_reader,
                &mut this.stage,
                &this.quantization,
                config,
            );
        }
    }

    pub fn create_job(this: &Self, #[comptime] config: G) -> SyncBufferLoaderJob<MP, L> {
        let loading_job = match this.loading_job {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<MP, G>(
                comptime!(this.buffer_id.to_index()),
                this.input_ident,
                config,
            ),
        };

        let num_tasks = L::Job::task_count(&loading_job);

        SyncBufferLoaderJob::<MP, L> {
            loading_job,
            num_tasks,
            current: ComptimeCell::new(TaskCounter { counter: 0u32 }),
        }
    }

    pub fn execute_task(
        this: &mut Self,
        job: &mut SyncBufferLoaderJob<MP, L>,
        #[comptime] config: G,
    ) {
        let task_id = job.current.read().counter;

        L::Job::<MP>::execute_task::<G>(
            &mut job.loading_job,
            task_id,
            &this.tensor_reader,
            &mut this.stage,
            &this.quantization,
            config,
        );

        job.current.store(TaskCounter {
            counter: comptime!(task_id + 1u32),
        });
    }
}

#[derive(CubeType)]
pub struct SyncBufferLoaderJob<MP: MatmulPrecision, L: SyncBufferLoadingStrategy> {
    loading_job: L::Job<MP>,
    #[cube(comptime)]
    pub num_tasks: u32,
    pub current: ComptimeCell<TaskCounter>,
}

#[derive(CubeType, Clone)]
pub struct TaskCounter {
    #[cube(comptime)]
    pub counter: u32,
}
