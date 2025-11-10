use std::marker::PhantomData;

use crate::components::global::read::{PartialLoadingStrategy, tiled::TiledLayout};
use crate::components::global::{GlobalConfig, RoleRule};
use crate::components::global::{multi_stage::LoadMaxRoundPlaneCount, read::sync::Synchronous};
use crate::components::stage::{ContiguousTilingLayout, StridedStage, TilingOrder};
use crate::components::{InvalidConfigError, MatmulIdent, MatrixPrecision, TilingScheme};
use crate::components::{global::memory::GlobalIterator, stage::TilingValidation};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{LoadingJob, LoadingValidation, ReaderMode};

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the stage using all planes.
/// Unit with pos X loads lines with indices X, X + NUM_UNITS, X + 2 * NUM_UNITS, ...
pub struct SyncPartialCyclicLoading<T: TilingOrder> {
    #[cube(comptime)]
    _phantom: PhantomData<T>,
}

impl<TO: TilingOrder> LoadingValidation for SyncPartialCyclicLoading<TO> {
    fn check<C: GlobalConfig, R: Runtime>(
        _client: &ComputeClient<R::Server>,
        config: &C,
        ident: MatmulIdent,
    ) -> Result<(), InvalidConfigError> {
        if let ReaderMode::Strict = config.reader_mode() {
            let line_size = config.global_line_size(ident);
            let num_lines_per_tile = config.tiling_scheme().elements_in_tile(ident) / line_size;
            let num_tiles_in_stage = config.tiling_scheme().tiles_in_stage(ident);
            let total_num_lines = num_tiles_in_stage * num_lines_per_tile;

            let total_units = config.plane_dim() * config.num_loading_planes(ident);
            let jump_length = total_units * line_size;
            let num_tasks_per_unit = total_num_lines.div_ceil(total_units);

            let max_id = total_units - 1;
            let max_task_id = num_tasks_per_unit - 1;
            let max_position_base = max_id * line_size;
            let max_position = max_position_base + max_task_id * jump_length;
            let num_stage_elements = config.tiling_scheme().elements_in_stage(ident);

            if max_position > num_stage_elements {
                return Err(Box::new(
                    "Too many data will be loaded, resulting in out-of-bounds",
                ));
            }
        }

        ContiguousTilingLayout::<TO>::check(config.global_memory_config(ident))?;

        Ok(())
    }
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for SyncPartialCyclicLoading<TO> {
    fn max_round_plane_count(
        tiling_scheme: &TilingScheme,
        ident: MatmulIdent,
        line_size: u8,
        plane_dim: u32,
    ) -> u32 {
        let num_lines_per_tile = tiling_scheme.elements_in_tile(ident) / line_size as u32;
        let num_tiles_in_stage = tiling_scheme.tiles_in_stage(ident);
        let total_num_lines = num_tiles_in_stage * num_lines_per_tile;
        total_num_lines.div_ceil(plane_dim)
    }
}

#[cube]
impl<TO: TilingOrder> PartialLoadingStrategy for SyncPartialCyclicLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = Synchronous;
    type Job<IP: MatrixPrecision> = SyncPartialCyclicJob;

    fn new_job<IP: MatrixPrecision, G: GlobalConfig>(
        #[comptime] stage_index: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] line_size: u32,
        #[comptime] config: G,
    ) -> SyncPartialCyclicJob {
        let num_stage_elements = config.tiling_scheme().elements_in_stage(ident);

        let tile_size = config.tiling_scheme().elements_in_tile(ident);
        let tile_count_row = config.tiling_scheme().tiles_in_stage_row(ident);
        let tile_count_col = config.tiling_scheme().tiles_in_stage_col(ident);

        let num_lines_per_tile = tile_size / line_size;
        let total_units = config.plane_dim() * config.num_loading_planes(ident);

        let num_tiles_in_stage = tile_count_row * tile_count_col;
        let total_num_lines = num_tiles_in_stage * num_lines_per_tile;
        let balanced_workload = total_num_lines.is_multiple_of(total_units);
        let num_tasks_per_unit = total_num_lines.div_ceil(total_units);
        let jump_length = total_units * line_size;

        let plane_id = RoleRule::new(config.role_rule_config())
            .load_index(ident, config.specialized_loading_sides());
        let unit_id = plane_id * config.plane_dim() + UNIT_POS_X;
        let unit_position_base = unit_id * line_size;

        SyncPartialCyclicJob {
            unit_position_base,
            num_tasks_per_unit,
            stage_index,
            jump_length,
            num_lines_per_tile,
            ident,
            balanced_workload,
            num_stage_elements,
            reader_mode: comptime!(config.reader_mode()),
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct SyncPartialCyclicJob {
    unit_position_base: u32,

    #[cube(comptime)]
    num_tasks_per_unit: u32,
    #[cube(comptime)]
    stage_index: u32,
    #[cube(comptime)]
    jump_length: u32,
    #[cube(comptime)]
    num_lines_per_tile: u32,
    #[cube(comptime)]
    ident: MatmulIdent,
    #[cube(comptime)]
    balanced_workload: bool,
    #[cube(comptime)]
    num_stage_elements: u32,
    #[cube(comptime)]
    reader_mode: ReaderMode,
}

#[cube]
impl<IP: MatrixPrecision, TO: TilingOrder> LoadingJob<IP, ContiguousTilingLayout<TO>, Synchronous>
    for SyncPartialCyclicJob
{
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<IP::Global>>,
        stage: &mut StridedStage<IP::Stage, ContiguousTilingLayout<TO>>,
        _barrier: &mut (),
        #[comptime] config: G,
    ) {
        let unit_position = this.unit_position_base + task_id * this.jump_length;
        let mut stage = stage.with_buffer_index(this.stage_index);

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.reader_mode == ReaderMode::Strict || this.balanced_workload) {
            load_and_store_line::<IP, TO, G>(this, unit_position, global_iter, &mut stage, config);
        } else {
            if unit_position < this.num_stage_elements {
                load_and_store_line::<IP, TO, G>(
                    this,
                    unit_position,
                    global_iter,
                    &mut stage,
                    config,
                );
            }
        }
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        this.num_tasks_per_unit
    }
}

#[cube]
pub(crate) fn load_and_store_line<IP: MatrixPrecision, TO: TilingOrder, G: GlobalConfig>(
    job: &SyncPartialCyclicJob,
    unit_position: u32,
    global_iter: &GlobalIterator<Line<IP::Global>>,
    stage: &mut StridedStage<IP::Stage, ContiguousTilingLayout<TO>>,
    #[comptime] config: G,
) {
    let layout = TiledLayout::new(comptime!(config.global_memory_config(job.ident)));
    let view = global_iter.view().view(layout);

    let (tile_size, tile_count_row, tile_count_col) = comptime! {
        (
            config.tiling_scheme().elements_in_tile(job.ident),
            config.tiling_scheme().tiles_in_stage_row(job.ident),
            config.tiling_scheme().tiles_in_stage_col(job.ident),
        )
    };
    let line_size = view.line_size();

    let tile_index = unit_position / tile_size;
    let pos_within_tile = unit_position % tile_size;

    let (tile_x_within_stage, tile_y_within_stage) = TO::to_row_col(
        tile_index,
        tile_count_row,
        tile_count_col,
        comptime!(config.stage_memory_config(job.ident)),
    );

    let tile = match comptime!(job.ident) {
        MatmulIdent::Lhs => (
            tile_x_within_stage,
            job.stage_index * tile_count_col + tile_y_within_stage,
        ),
        MatmulIdent::Rhs => (
            job.stage_index * tile_count_row + tile_x_within_stage,
            tile_y_within_stage,
        ),
        MatmulIdent::Out => comptime!(unreachable!()),
    };

    let line_read = view.read_checked((tile, pos_within_tile));

    let tile_start = tile_index * job.num_lines_per_tile;
    let tile_end = tile_start + job.num_lines_per_tile;
    let mut tile_slice = stage
        .as_slice_mut(line_size)
        .slice_mut(tile_start, tile_end);

    tile_slice[pos_within_tile / line_size] = Line::cast_from(line_read);
}
