use std::marker::PhantomData;

use crate::components::global::load::{AsyncLoadingJob, LoadingValidation};
use crate::components::global::tensor_view::TensorReader;
use crate::components::global::{CopyMechanism, GlobalConfig};
use crate::components::global::{Quantization, single_stage};
use crate::components::stage::FullStageToTileReader;
use crate::components::stage::TilingLayout;
use crate::components::stage::{self, StageMemory};
use crate::components::{InputIdent, MatmulPrecision, global};
use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::BarrierLevel;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, CubeOptionExpand};

#[cube]
/// A strategy for fully and asynchronously loading a stage.
pub trait AsyncFullLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;

    /// The [LoadingJob] for this strategy.
    type Job<MP: MatmulPrecision>: AsyncLoadingJob<MP, Self::TilingLayout>;

    /// Returns the job with preliminary calculations done.
    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP>;

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
    tensor_reader: TensorReader<MP::EI>,
    stage_memory: StageMemory<MP::ES, L::TilingLayout>,
    loading_job: CubeOption<L::Job<MP>>,
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

        let mut stage_memory =
            StageMemory::new::<G::StageConfig>(1u32, ident.as_ident(), config.stage_config());
        let tensor_reader = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        let loading_job = match config.precompute_job() {
            true => CubeOption::new_Some(L::new_job::<MP, G>(ident, config)),
            false => CubeOption::new_None(),
        };

        match ident {
            InputIdent::Lhs =>
            {
                #[allow(clippy::collapsible_if)]
                if config.check_row_bounds(ident) {
                    if tensor_reader.x_offset.read()
                        > tensor_reader.shape_x - config.tiling_scheme().elements_in_stage_m()
                    {
                        stage_memory.clear::<G>(ident, config);
                    }
                }
            }
            InputIdent::Rhs =>
            {
                #[allow(clippy::collapsible_if)]
                if config.check_col_bounds(ident) {
                    if tensor_reader.y_offset.read()
                        > tensor_reader.shape_y - config.tiling_scheme().elements_in_stage_n()
                    {
                        stage_memory.clear::<G>(ident, config);
                    }
                }
            }
        }

        AsyncLoader::<MP, CM, S, L> {
            tensor_reader,
            stage_memory,
            loading_job,
            ident,
            _phantom: PhantomData::<(S, L, CM)>,
        }
    }

    pub fn fill_stage(
        this: &mut Self,
        mechanism: &CM,
        #[comptime] config: single_stage::Config<S>,
    ) {
        let mut loading_job = match this.loading_job {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<MP, single_stage::Config<S>>(this.ident, config),
        };

        let len = L::Job::task_count(&loading_job);
        for task_id in 0..len {
            L::Job::<MP>::execute_task::<CM, single_stage::Config<S>>(
                &mut loading_job,
                task_id,
                &this.tensor_reader,
                &mut this.stage_memory,
                mechanism,
                config,
            );
        }
    }

    pub fn clear_stage(this: &mut Self, #[comptime] config: single_stage::Config<S>) {
        this.stage_memory
            .clear::<single_stage::Config<S>>(this.ident, config)
    }

    pub fn reader(this: &Self) -> FullStageToTileReader<MP::ES, L::TilingLayout> {
        FullStageToTileReader::new(this.stage_memory, this.ident)
    }

    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_reader.update_view(k_offset, this.ident);
    }
}
