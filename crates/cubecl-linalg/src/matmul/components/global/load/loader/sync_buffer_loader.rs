use std::marker::PhantomData;

use super::BufferId;
use crate::matmul::components::InputIdent;
use crate::matmul::components::MatmulPrecision;
use crate::matmul::components::global::GlobalConfig;
use crate::matmul::components::global::LoadingValidation;
use crate::matmul::components::global::Quantization;
use crate::matmul::components::global::load::JobConfig;
use crate::matmul::components::global::load::LoadingJob;
use crate::matmul::components::global::load::strategy::LoadingJobConfig;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::stage::Stage;
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::stage::single_buffer::BufferReader;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[cube]
/// A strategy for synchronously loading a buffer (partial stage), either eagerly or as a deferred job.
pub trait SyncBufferLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;

    /// The [LoadingJob] for this strategy.
    type Job<MP: MatmulPrecision>: LoadingJob<MP, Self::TilingLayout>;

    /// Returns the job with preliminary calculations done.
    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] buffer_index: u32,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP>;
}

#[derive(Clone, CubeType)]
pub struct SyncBufferLoader<MP: MatmulPrecision, G: GlobalConfig, L: SyncBufferLoadingStrategy> {
    tensor_reader: TensorReader<MP::EI>,
    stage: Stage<MP::ES, L::TilingLayout>,
    loading_job_a: L::Job<MP>,
    loading_job_b: L::Job<MP>,
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
        let stage = Stage::new::<G::SmmConfig>(input_ident.as_ident(), config.to_smm_config());
        let tensor_reader = TensorReader::new(tensor, x_offset, y_offset, batch_offset);
        let loading_job_a = L::new_job::<MP, G>(quantization, 0u32, input_ident, config);
        let loading_job_b = L::new_job::<MP, G>(quantization, 1u32, input_ident, config);

        SyncBufferLoader::<MP, G, L> {
            tensor_reader,
            stage,
            loading_job_a,
            loading_job_b,
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
        let mut loading_job = match buffer_id {
            BufferId::A => this.loading_job_a,
            BufferId::B => this.loading_job_b,
        };

        let len = JobConfig::<MP, L::TilingLayout, L::Job<MP>>::len(&loading_job);
        for task_id in 0..len {
            L::Job::<MP>::execute_task::<G>(
                &mut loading_job,
                task_id,
                &this.tensor_reader,
                &mut this.stage,
                config,
            );
        }
    }
}
