use std::marker::PhantomData;

use crate::matmul::components::global::load::LoadingJob;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{CopyMechanism, GlobalConfig, LoadingValidation};
use crate::matmul::components::global::{Quantization, single_stage};
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::stage::multi_buffer::FullReader;
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{Ident, InputIdent, MatmulPrecision, global};
use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::BarrierLevel;
use cubecl_core::prelude::*;
use cubecl_std::CubeOption;
use cubecl_std::tensor::r#virtual::VirtualTensor;

#[cube]
/// A strategy for fully and asynchronously loading a stage, either eagerly or as a deferred job.
pub trait AsyncFullLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;

    /// The [LoadingJob] for this strategy
    type Job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>>: LoadingJob<MP>;

    /// Loads the entire stage immediately from the tensor reader.
    fn load_full<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        read_view: &TensorReader<MP::EI>,
        stage: Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    );

    /// Returns the job with preliminary calculations done.
    fn job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        stage: Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP, CM>;

    /// The barrier level at which the copy mechanism works
    fn barrier_level() -> BarrierLevel;
}

#[derive(CubeType)]
pub struct AsyncLoader<
    MP: MatmulPrecision,
    CM: CopyMechanism<MP::ES>,
    S: stage::StageConfig,
    L: AsyncFullLoadingStrategy,
> {
    pub tensor_view: TensorReader<MP::EI>,
    pub stage: Stage<MP::ES, L::TilingLayout>,
    pub quantization: CubeOption<Quantization<MP>>,
    #[cube(comptime)]
    ident: InputIdent,
    #[cube(comptime)]
    _phantom: PhantomData<(S, L, CM)>,
}

#[cube]
impl<
    MP: MatmulPrecision,
    CM: CopyMechanism<MP::ES>,
    S: stage::StageConfig,
    L: AsyncFullLoadingStrategy,
> AsyncLoader<MP, CM, S, L>
{
    pub fn new<G: global::GlobalConfig>(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    ) -> Self {
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }

        let mut stage = Stage::new::<G::SmmConfig>(ident.as_ident(), config.to_smm_config());

        match ident {
            InputIdent::Lhs =>
            {
                #[allow(clippy::collapsible_if)]
                if config.check_row_bounds(ident) {
                    if x_offset
                        > tensor.shape(tensor.rank() - 2)
                            - config.tiling_dimensions(Ident::Lhs).total_row()
                    {
                        stage.clear::<G::SmmConfig>(ident, config.to_smm_config());
                    }
                }
            }
            InputIdent::Rhs =>
            {
                #[allow(clippy::collapsible_if)]
                if config.check_col_bounds(ident) {
                    if y_offset
                        > tensor.shape(tensor.rank() - 1)
                            - config.tiling_dimensions(Ident::Rhs).total_col()
                    {
                        stage.clear::<G::SmmConfig>(ident, config.to_smm_config());
                    }
                }
            }
        }

        let tensor_view = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        AsyncLoader::<MP, CM, S, L> {
            tensor_view,
            stage,
            quantization,
            ident,
            _phantom: PhantomData::<(S, L, CM)>,
        }
    }

    pub fn fill_stage(this: &mut Self, mechanism: CM, #[comptime] config: single_stage::Config<S>) {
        L::load_full::<MP, CM, single_stage::Config<S>>(
            &this.tensor_view,
            this.stage,
            mechanism,
            this.quantization,
            this.ident,
            config,
        );
    }

    pub fn clear_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        this.stage.clear::<S>(this.ident, config.to_smm_config())
    }

    pub fn reader(this: &Self) -> FullReader<MP::ES, L::TilingLayout> {
        FullReader::new(this.stage, this.ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_view.update_view(k_offset, this.ident);
    }
}
