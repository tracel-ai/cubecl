use super::BufferId;
use crate::matmul::components::global::base::GlobalConfig;
use crate::matmul::components::global::load::LoadingJob;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{
    CommonGlobalConfig, CopyMechanism, LoadingValidation, Quantization,
};
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::stage::single_buffer::BufferReader;
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
    type Job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>>: LoadingJob<MP>;

    /// Immediately load the stage for the buffer identified by buffer_index.
    fn load_buffer<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        tensor_reader: &TensorReader<MP::EI>,
        stage: Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] buffer_index: u32,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    );

    /// Returns the job with preliminary calculations done.
    fn job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        stage: Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] buffer_index: u32,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP, CM>;

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
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    pub quantization: CubeOption<Quantization<MP>>,
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
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }
        let stage = Stage::new::<S>(input_ident.as_ident(), config.to_smm_config());
        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        AsyncBufferLoader::<MP, S, CM, L> {
            tensor_view,
            stage,
            quantization,
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
        this.tensor_view.update_view(k_offset, this.input_ident);
    }

    pub fn fill_stage(
        this: &mut Self,
        mechanism: CM,
        #[comptime] buffer: BufferId,
        #[comptime] config: CommonGlobalConfig<S>,
    ) {
        L::load_buffer::<MP, CM, CommonGlobalConfig<S>>(
            &this.tensor_view,
            this.stage,
            mechanism,
            this.quantization,
            buffer.to_index(),
            this.input_ident,
            config,
        );
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
