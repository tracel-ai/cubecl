use super::BufferId;
use crate::matmul::components::global::base::GlobalConfig;
use crate::matmul::components::global::load::strategy::AsyncLoadingJobConfig;
use crate::matmul::components::global::load::{AsyncJobConfig, AsyncLoadingJob};
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{
    CommonGlobalConfig, CopyMechanism, LoadingValidation, Quantization,
};
use crate::matmul::components::stage::BufferReader;
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{InputIdent, MatmulPrecision};
use core::marker::PhantomData;
use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::BarrierLevel;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[cube]
/// A strategy for asynchronously loading a buffer (partial stage), either eagerly or as a deferred job.
pub trait AsyncBufferLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;

    /// The [LoadingJob] for this strategy.
    type Job<MP: MatmulPrecision>: AsyncLoadingJob<MP, Self::TilingLayout>;

    /// Returns the job with preliminary calculations done.
    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        quantization: CubeOption<Quantization<MP>>,
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
    stage: Stage<MP::ES, L::TilingLayout>,
    loading_job_a: L::Job<MP>,
    loading_job_b: L::Job<MP>,
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
        #[comptime] config: CommonGlobalConfig<S>,
    ) -> Self {
        let stage = Stage::new::<S>(input_ident.as_ident(), config.to_smm_config());
        let tensor_reader = TensorReader::new(tensor, x_offset, y_offset, batch_offset);
        let loading_job_a =
            L::new_job::<MP, CommonGlobalConfig<S>>(quantization, 0u32, input_ident, config);
        let loading_job_b =
            L::new_job::<MP, CommonGlobalConfig<S>>(quantization, 1u32, input_ident, config);

        AsyncBufferLoader::<MP, S, CM, L> {
            tensor_reader,
            stage,
            loading_job_a,
            loading_job_b,
            input_ident,
            _phantom: PhantomData::<(S, CM)>,
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

    pub fn fill_stage(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] buffer_id: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        let mut loading_job = match buffer_id {
            BufferId::A => this.loading_job_a,
            BufferId::B => this.loading_job_b,
        };

        let len = AsyncJobConfig::<MP, L::TilingLayout, L::Job<MP>>::len(&loading_job);
        for task_id in 0..len {
            L::Job::<MP>::execute_task::<CM, CommonGlobalConfig<S>>(
                &mut loading_job,
                task_id,
                &this.tensor_reader,
                &mut this.stage,
                mechanism,
                config,
            );
        }
    }

    pub fn clear_stage(
        this: &mut Self,
        #[comptime] buffer_id: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        this.stage
            .clear_buffer::<S>(buffer_id, this.input_ident, config.to_smm_config())
    }
}
