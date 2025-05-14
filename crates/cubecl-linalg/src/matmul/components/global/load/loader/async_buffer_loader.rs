use super::BufferId;
use crate::matmul::components::global::base::GlobalConfig;
use crate::matmul::components::global::load::{AsyncLoadingJob, LoadingValidation};
use crate::matmul::components::global::multi_stage::double_buffering::DoubleBufferingGlobalConfig;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CopyMechanism, Quantization};
use crate::matmul::components::stage::BufferReader;
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::stage::{self, StageMemory};
use crate::matmul::components::{InputIdent, MatmulPrecision};
use core::marker::PhantomData;
use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::BarrierLevel;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, CubeOptionExpand};

#[cube]
/// A strategy for asynchronously loading a buffer (partial stage), either eagerly or as a deferred job.
pub trait AsyncBufferLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;

    /// The [LoadingJob] for this strategy.
    type Job<MP: MatmulPrecision>: AsyncLoadingJob<MP, Self::TilingLayout>;

    /// Returns the job with preliminary calculations done.
    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] buffer_index: u32,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP>;

    /// The barrier level at which the copy mechanism works
    fn barrier_level() -> BarrierLevel;
}

#[derive(CubeType)]
pub struct AsyncBufferLoader<
    MP: MatmulPrecision,
    S: stage::StageConfig,
    CM: CopyMechanism<MP::ES>,
    L: AsyncBufferLoadingStrategy,
> {
    tensor_reader: TensorReader<MP::EI>,
    stage_memory: StageMemory<MP::ES, L::TilingLayout>,
    loading_job: CubeOption<(L::Job<MP>, L::Job<MP>)>,
    #[cube(comptime)]
    input_ident: InputIdent,
    #[cube(comptime)]
    _phantom: PhantomData<(S, CM)>,
}

#[cube]
impl<
    MP: MatmulPrecision,
    S: stage::StageConfig,
    CM: CopyMechanism<MP::ES>,
    L: AsyncBufferLoadingStrategy,
> AsyncBufferLoader<MP, S, CM, L>
{
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: DoubleBufferingGlobalConfig<S>,
    ) -> Self {
        let stage_memory =
            StageMemory::new::<S>(2u32, input_ident.as_ident(), config.to_smm_config());
        let tensor_reader = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }

        let loading_job = match config.precompute_job() {
            true => CubeOption::new_Some((
                L::new_job::<MP, DoubleBufferingGlobalConfig<S>>(0u32, input_ident, config),
                L::new_job::<MP, DoubleBufferingGlobalConfig<S>>(1u32, input_ident, config),
            )),
            false => CubeOption::new_None(),
        };

        AsyncBufferLoader::<MP, S, CM, L> {
            tensor_reader,
            stage_memory,
            loading_job,
            input_ident,
            _phantom: PhantomData::<(S, CM)>,
        }
    }

    pub fn reader(
        this: &Self,
        #[comptime] buffer_id: BufferId,
    ) -> BufferReader<MP::ES, L::TilingLayout> {
        BufferReader::new(this.stage_memory, buffer_id, this.input_ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_reader.update_view(k_offset, this.input_ident);
    }

    pub fn fill_stage(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] buffer_id: BufferId,
        #[comptime] config: DoubleBufferingGlobalConfig<S>,
    ) {
        let mut loading_job = match this.loading_job {
            CubeOption::Some(job) => match buffer_id {
                BufferId::A => job.0,
                BufferId::B => job.1,
            },
            CubeOption::None => match buffer_id {
                BufferId::A => {
                    L::new_job::<MP, DoubleBufferingGlobalConfig<S>>(0u32, this.input_ident, config)
                }
                BufferId::B => {
                    L::new_job::<MP, DoubleBufferingGlobalConfig<S>>(1u32, this.input_ident, config)
                }
            },
        };

        let len = L::Job::task_count(&loading_job);
        for task_id in 0..len {
            L::Job::<MP>::execute_task::<CM, DoubleBufferingGlobalConfig<S>>(
                &mut loading_job,
                task_id,
                &this.tensor_reader,
                &mut this.stage_memory,
                mechanism,
                config,
            );
        }
    }

    pub fn clear_stage(
        this: &mut Self,
        #[comptime] buffer_id: BufferId,
        #[comptime] config: DoubleBufferingGlobalConfig<S>,
    ) {
        this.stage_memory
            .clear_buffer::<S>(buffer_id, this.input_ident, config.to_smm_config())
    }
}
