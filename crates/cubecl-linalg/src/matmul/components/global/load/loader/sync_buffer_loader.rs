use std::marker::PhantomData;

use super::BufferId;
use super::TaskCounter;
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
use cubecl_std::tensor::r#virtual::VirtualTensor;
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
        let stage =
            StageMemory::new::<G::SmmConfig>(2u32, input_ident.as_ident(), config.to_smm_config());
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
            stage,
            loading_job,
            quantization,
            input_ident,
            _config: PhantomData::<G>,
        }
    }

    pub fn reader(
        this: &Self,
        #[comptime] buffer_id: BufferId,
    ) -> BufferReader<MP::ES, L::TilingLayout> {
        BufferReader::new(this.stage, buffer_id, this.input_ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_reader.update_view(k_offset, this.input_ident);
    }

    pub fn fill_stage(this: &mut Self, #[comptime] buffer_id: BufferId, #[comptime] config: G) {
        let mut loading_job = match this.loading_job {
            CubeOption::Some(job) => match buffer_id {
                BufferId::A => job.0,
                BufferId::B => job.1,
            },
            CubeOption::None => match buffer_id {
                BufferId::A => L::new_job::<MP, G>(0u32, this.input_ident, config),
                BufferId::B => L::new_job::<MP, G>(1u32, this.input_ident, config),
            },
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

    pub fn create_job(
        this: &Self,
        #[comptime] buffer_id: BufferId,
        #[comptime] config: G,
    ) -> SyncBufferLoaderJob<MP, L> {
        let loading = match this.loading_job {
            CubeOption::Some(job) => match buffer_id {
                BufferId::A => job.0,
                BufferId::B => job.1,
            },
            CubeOption::None => match buffer_id {
                BufferId::A => L::new_job::<MP, G>(0u32, this.input_ident, config),
                BufferId::B => L::new_job::<MP, G>(1u32, this.input_ident, config),
            },
        };

        let num_tasks = L::Job::task_count(&loading);

        SyncBufferLoaderJob::<MP, L> {
            loading,
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
            &mut job.loading,
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
    loading: L::Job<MP>,
    #[cube(comptime)]
    pub num_tasks: u32,
    pub current: ComptimeCell<TaskCounter>,
}
