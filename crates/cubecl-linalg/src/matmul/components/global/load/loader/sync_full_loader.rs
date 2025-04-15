use std::marker::PhantomData;

use crate::matmul::components::global::Quantization;
use crate::matmul::components::global::load::LoadingJob;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{self};
use crate::matmul::components::global::{GlobalConfig, LoadingValidation, single_stage};
use crate::matmul::components::stage::FullReader;
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::stage::{self, Stage};
use crate::matmul::components::{InputIdent, MatmulPrecision};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, CubeOptionExpand};

#[cube]
/// A strategy for fully and synchronously loading a stage.
pub trait SyncFullLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
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

#[derive(CubeType)]
pub struct SyncFullLoader<MP: MatmulPrecision, S: stage::StageConfig, L: SyncFullLoadingStrategy> {
    tensor_reader: TensorReader<MP::EI>,
    stage: Stage<MP::ES, L::TilingLayout>,
    loading_job: CubeOption<L::Job<MP>>,
    quantization: CubeOption<Quantization<MP>>,
    #[cube(comptime)]
    input_ident: InputIdent,
    #[cube(comptime)]
    _phantom: PhantomData<(S, L)>,
}

#[cube]
impl<MP: MatmulPrecision, S: stage::StageConfig, L: SyncFullLoadingStrategy>
    SyncFullLoader<MP, S, L>
{
    pub fn new<G: global::GlobalConfig>(
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

        let loading_job = match config.precompute_job() {
            true => CubeOption::new_Some(L::new_job::<MP, G>(input_ident, config)),
            false => CubeOption::new_None(),
        };

        SyncFullLoader::<MP, S, L> {
            tensor_reader,
            stage,
            loading_job,
            quantization,
            input_ident,
            _phantom: PhantomData::<(S, L)>,
        }
    }

    pub fn reader(this: &Self) -> FullReader<MP::ES, L::TilingLayout> {
        FullReader::new(this.stage, this.input_ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_reader.update_view(k_offset, this.input_ident);
    }

    pub fn fill_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        let mut loading_job = match this.loading_job {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<MP, single_stage::Config<S>>(this.input_ident, config),
        };

        let len = L::Job::task_count(&loading_job);
        for task_id in 0..len {
            L::Job::<MP>::execute_task::<single_stage::Config<S>>(
                &mut loading_job,
                task_id,
                &this.tensor_reader,
                &mut this.stage,
                &this.quantization,
                config,
            );
        }
    }
}
