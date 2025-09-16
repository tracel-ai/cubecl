use std::marker::PhantomData;

use crate::components::global::load::{AsyncLoadingJob, LoadingValidation};
use crate::components::global::memory::TensorReader;
use crate::components::global::{CopyMechanism, GlobalConfig};
use crate::components::stage::FullStageReader;
use crate::components::stage::TilingLayout;
use crate::components::stage::{self, StageMemory};
use crate::components::{InputPrecision, MatmulIdent};
use cubecl_core as cubecl;
use cubecl_core::prelude::barrier::BarrierLevel;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption, CubeOptionExpand,
    tensor::{View, layout::Coords3d},
};

#[cube]
/// A strategy for fully and asynchronously loading a stage.
pub trait AsyncFullLoadingStrategy: 'static + Send + Sync + Clone + LoadingValidation {
    /// The layout describing how data is tiled across the stage.
    type TilingLayout: TilingLayout;

    /// The [LoadingJob] for this strategy.
    type Job<IP: InputPrecision>: AsyncLoadingJob<IP, Self::TilingLayout>;

    /// Returns the job with preliminary calculations done.
    fn new_job<IP: InputPrecision, G: GlobalConfig>(
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self::Job<IP>;

    /// The barrier level at which the copy mechanism works
    fn barrier_level() -> BarrierLevel;
}

#[derive(CubeType)]
/// Loads the entire stage memory using asynchronous data movement operations.
///
/// A complete load is referred to as a `Job`, which is divided into `Tasks`—
/// each Task represents a single data transfer for a specific unit
pub struct AsyncFullStageLoader<
    IP: InputPrecision,
    CM: CopyMechanism,
    S: stage::StageConfig,
    L: AsyncFullLoadingStrategy,
    G: GlobalConfig,
> {
    tensor_reader: TensorReader<IP::Global>,
    stage_memory: StageMemory<IP::Stage, L::TilingLayout>,
    loading_job: CubeOption<L::Job<IP>>,
    #[cube(comptime)]
    ident: MatmulIdent,
    #[cube(comptime)]
    _phantom: PhantomData<(S, L, CM, G)>,
}

#[cube]
impl<
    IP: InputPrecision,
    CM: CopyMechanism,
    S: stage::StageConfig,
    L: AsyncFullLoadingStrategy,
    G: GlobalConfig,
> AsyncFullStageLoader<IP, CM, S, L, G>
{
    /// Create a new AsyncFullLoader
    pub fn new(
        view: View<Line<IP::Global>, Coords3d>,
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self {
        let mut stage_memory = StageMemory::new::<G::StageMemoryConfig>(
            1u32,
            comptime!(ident.into_stage()),
            config.stage_memory_config(),
        );
        let (_, shape_x, shape_y) = view.shape();
        let tensor_reader = TensorReader::new(view);

        let loading_job = match config.precompute_job() {
            true => CubeOption::new_Some(L::new_job::<IP, G>(ident, config)),
            false => CubeOption::new_None(),
        };

        match ident {
            MatmulIdent::Lhs =>
            {
                #[allow(clippy::collapsible_if)]
                if config.check_row_bounds(ident) {
                    if tensor_reader.row_offset.read()
                        > shape_x - config.tiling_scheme().elements_in_stage_m()
                    {
                        stage_memory.clear_all::<G>(ident, config);
                    }
                }
            }
            MatmulIdent::Rhs =>
            {
                #[allow(clippy::collapsible_if)]
                if config.check_col_bounds(ident) {
                    if tensor_reader.col_offset.read()
                        > shape_y - config.tiling_scheme().elements_in_stage_n()
                    {
                        stage_memory.clear_all::<G>(ident, config);
                    }
                }
            }
            MatmulIdent::Out => comptime!(unreachable!()),
        }

        AsyncFullStageLoader::<IP, CM, S, L, G> {
            tensor_reader,
            stage_memory,
            loading_job,
            ident,
            _phantom: PhantomData,
        }
    }

    /// Accomplish the entire job of loading data into the stage memory
    pub fn load_stage(&mut self, mechanism: &CM, #[comptime] config: G) {
        let mut loading_job = match self.loading_job {
            CubeOption::Some(loading_job) => loading_job,
            CubeOption::None => L::new_job::<IP, G>(self.ident, config),
        };

        let len = L::Job::task_count(&loading_job);
        for task_id in 0..len {
            L::Job::<IP>::execute_task::<CM, G>(
                &mut loading_job,
                task_id,
                &self.tensor_reader,
                &mut self.stage_memory,
                mechanism,
                config,
            );
        }
    }

    /// Zero out the stage memory
    pub fn clear_stage(&mut self, #[comptime] config: G) {
        self.stage_memory.clear_all::<G>(self.ident, config)
    }

    /// Give a reader to the loaded stage memory.
    pub fn reader(&self) -> FullStageReader<IP::Stage, L::TilingLayout> {
        FullStageReader::new(self.stage_memory, comptime!(self.ident.into_stage()))
    }

    /// Advance the view over global memory along the k dimension by a specified offset, `k_offset`.
    pub fn advance_view(&mut self, k_offset: u32) {
        self.tensor_reader
            .update_view(k_offset, comptime!(self.ident.view_direction()));
    }
}
