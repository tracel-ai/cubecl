use std::marker::PhantomData;

use crate::components::global::Quantization;
use crate::components::global::global_memory::TensorReader;
use crate::components::global::load::{AsyncLoadingJob, LoadingValidation};
use crate::components::global::{CopyMechanism, GlobalConfig};
use crate::components::stage::FullStageToTileReader;
use crate::components::stage::TilingLayout;
use crate::components::stage::{self, StageMemory};
use crate::components::{MatmulIdent, MatmulPrecision, StageIdent};
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
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP>;

    /// The barrier level at which the copy mechanism works
    fn barrier_level() -> BarrierLevel;
}

#[derive(CubeType)]
/// Loads the entire stage memory using asynchronous data movement operations.
///
/// A complete load is referred to as a `Job`, which is divided into `Tasks`—
/// each Task represents a single data transfer for a specific unit
pub struct AsyncFullLoader<
    MP: MatmulPrecision,
    CM: CopyMechanism<MP::ES>,
    S: stage::StageConfig,
    L: AsyncFullLoadingStrategy,
    G: GlobalConfig,
> {
    tensor_reader: TensorReader<MP::EI>,
    stage_memory: StageMemory<MP::ES, L::TilingLayout>,
    loading_job: CubeOption<L::Job<MP>>,
    #[cube(comptime)]
    ident: MatmulIdent,
    #[cube(comptime)]
    _phantom: PhantomData<(S, L, CM, G)>,
}

#[cube]
impl<
    MP: MatmulPrecision,
    CM: CopyMechanism<MP::ES>,
    S: stage::StageConfig,
    L: AsyncFullLoadingStrategy,
    G: GlobalConfig,
> AsyncFullLoader<MP, CM, S, L, G>
{
    /// Create a new AsyncFullLoader
    pub fn new(
        tensor: VirtualTensor<MP::EI>,
        x_offset: u32,
        y_offset: u32,
        batch_offset: u32,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self {
        comptime! {
            if quantization.is_some() {
                todo!();
            }
        }
        let stage_ident = comptime!(StageIdent::from_matmul(ident));

        let mut stage_memory =
            StageMemory::new::<G::StageConfig>(1u32, stage_ident, config.stage_config());
        let tensor_reader = TensorReader::new(tensor, x_offset, y_offset, batch_offset);

        let loading_job = match config.precompute_job() {
            true => CubeOption::new_Some(L::new_job::<MP, G>(ident, config)),
            false => CubeOption::new_None(),
        };

        match ident {
            MatmulIdent::Lhs =>
            {
                #[allow(clippy::collapsible_if)]
                if config.check_row_bounds(ident) {
                    if tensor_reader.x_offset.read()
                        > tensor_reader.shape_x - config.tiling_scheme().elements_in_stage_m()
                    {
                        stage_memory.clear_all::<G>(stage_ident, config);
                    }
                }
            }
            MatmulIdent::Rhs =>
            {
                #[allow(clippy::collapsible_if)]
                if config.check_col_bounds(ident) {
                    if tensor_reader.y_offset.read()
                        > tensor_reader.shape_y - config.tiling_scheme().elements_in_stage_n()
                    {
                        stage_memory.clear_all::<G>(stage_ident, config);
                    }
                }
            }
            MatmulIdent::Out => comptime!(unreachable!()),
        }

        AsyncFullLoader::<MP, CM, S, L, G> {
            tensor_reader,
            stage_memory,
            loading_job,
            ident,
            _phantom: PhantomData,
        }
    }

    /// Accomplish the entire job of filling the stage memory
    pub fn fill_stage(this: &mut Self, mechanism: &CM, #[comptime] config: G) {
        let mut loading_job = match this.loading_job {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<MP, G>(this.ident, config),
        };

        let len = L::Job::task_count(&loading_job);
        for task_id in 0..len {
            L::Job::<MP>::execute_task::<CM, G>(
                &mut loading_job,
                task_id,
                &this.tensor_reader,
                &mut this.stage_memory,
                mechanism,
                config,
            );
        }
    }

    /// Zero out the stage memory
    pub fn clear_stage(this: &mut Self, #[comptime] config: G) {
        this.stage_memory
            .clear_all::<G>(comptime!(StageIdent::from_matmul(this.ident)), config)
    }

    /// Give a reader to the loaded stage memory.
    pub fn reader(this: &Self) -> FullStageToTileReader<MP::ES, L::TilingLayout> {
        FullStageToTileReader::new(
            this.stage_memory,
            comptime!(StageIdent::from_matmul(this.ident)),
        )
    }

    /// Advance the view over global memory along the k dimension by a specified offset, `k_offset`.
    pub fn advance_view(this: &mut Self, k_offset: u32) {
        this.tensor_reader.update_view(k_offset, this.ident);
    }
}
