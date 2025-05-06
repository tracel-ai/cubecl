use crate::matmul::components::global::Quantization;
use crate::matmul::components::global::load::SyncFullLoadingStrategy;
use crate::matmul::components::stage::OrderedTilingOrder;
use crate::matmul::components::{
    FormattedConfigError, Ident, InputIdent, InvalidConfigError, MatmulPrecision,
};
use crate::matmul::components::{
    global::{GlobalConfig, LoadingValidation, tensor_view::TensorReader},
    stage::{ContiguousTilingLayout, StageMemory, TilingOrder},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::LoadingJob;

#[derive(CubeType, Clone, Copy)]
/// Similar to `sync_full_tilewise`, but includes additional validation checks.
///
/// This function operates only on the LHS (left-hand side).
///
/// - In the single-row case, behavior is similar to `tilewise` with row-major tiling order.
///   However, it will explicitly fail if any plane does not load its entire row.
/// - In the multi-row case, it too will fail if a plane does not load all its rows.
///   Within each plane, the local tiling order is column-major.
pub struct LoadingStrategy {}

impl LoadingValidation for LoadingStrategy {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let tiling = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);

        let num_planes = config.num_planes();
        let num_tiles = tiling.tile_count();

        if num_tiles % num_planes != 0 {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Number of planes {:?} must divide number of tiles {:?} for ordered loading.",
                    num_planes, num_tiles,
                )
            }));
        }

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile = comptime!(tiling.tile_size() / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_planes = config.num_planes();
        let plane_dim = config.plane_dim();
        let rows_per_plane = tiling.tile_count_row() / num_planes;

        if num_lines_per_plane % plane_dim != 0 {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Plane dimension {:?} must divide number of lines per plane {:?} for ordered loading.",
                    plane_dim, num_lines_per_plane,
                )
            }));
        }

        if num_tiles_per_plane != rows_per_plane * tiling.tile_count_col() {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Number of tiles per plane {:?} must equal rows_per_plane {:?} times cols {:?} for ordered loading.",
                    num_tiles_per_plane,
                    rows_per_plane,
                    tiling.tile_count_col(),
                )
            }));
        }

        if ident != Ident::Lhs {
            return Err(FormattedConfigError::new(move || {
                format!("Ordered loading only available on Lhs")
            }));
        }

        Ok(())
    }
}

#[cube]
impl SyncFullLoadingStrategy for LoadingStrategy {
    type TilingLayout = ContiguousTilingLayout<OrderedTilingOrder>;
    type Job<MP: MatmulPrecision> = Job;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Self::Job<MP> {
        let tiling = config.tiling_dimensions(input_ident);
        let line_size = config.global_line_size(input_ident);
        let num_planes = config.num_planes();
        let num_tiles = tiling.tile_count();
        let plane_dim = config.plane_dim();

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile = comptime!(tiling.tile_size() / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_lines_per_unit = num_lines_per_plane / plane_dim;

        let num_tiles_to_skip = UNIT_POS_Y * num_tiles_per_plane;
        let num_lines_to_skip = num_tiles_to_skip * num_lines_per_tile;

        Job {
            num_tiles_to_skip,
            num_lines_to_skip,
            num_lines_per_tile,
            num_lines_per_unit,
            plane_dim: config.plane_dim(),
            line_size,
            input_ident,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct Job {
    num_tiles_to_skip: u32,
    num_lines_to_skip: u32,

    #[cube(comptime)]
    num_lines_per_tile: u32,
    #[cube(comptime)]
    num_lines_per_unit: u32,
    #[cube(comptime)]
    plane_dim: u32,
    #[cube(comptime)]
    line_size: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
}

#[cube]
impl<MP: MatmulPrecision, TO: TilingOrder> LoadingJob<MP, ContiguousTilingLayout<TO>> for Job {
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut StageMemory<MP::ES, ContiguousTilingLayout<TO>>,
        quantization: &CubeOption<Quantization<MP>>,
        #[comptime] config: G,
    ) {
        let pos_across_tiles = task_id * this.plane_dim + UNIT_POS_X;
        let nth_tile_for_this_plane = pos_across_tiles / this.num_lines_per_tile;
        let line_index_within_tile = pos_across_tiles % this.num_lines_per_tile;

        let nth_tile_global = nth_tile_for_this_plane + this.num_tiles_to_skip;
        let tile = ContiguousTilingLayout::<TO>::to_x_y::<G::SmmConfig>(
            nth_tile_global,
            comptime!(this.input_ident.as_ident()),
            config.to_smm_config(),
        );

        Job::load_and_store_line::<MP, TO, G>(
            this,
            tile,
            line_index_within_tile,
            nth_tile_for_this_plane * this.num_lines_per_tile,
            tensor_reader,
            stage,
            quantization,
            config,
        );
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        comptime!(this.num_lines_per_unit)
    }
}

#[cube]
impl Job {
    #[allow(clippy::too_many_arguments)]
    fn load_and_store_line<MP: MatmulPrecision, TO: TilingOrder, G: GlobalConfig>(
        this: &Self,
        tile: (u32, u32),
        line_index_within_tile: u32,
        num_lines_to_skip_local: u32,
        tensor_reader: &TensorReader<MP::EI>,
        stage: &mut StageMemory<MP::ES, ContiguousTilingLayout<TO>>,
        quantization: &CubeOption<Quantization<MP>>,
        #[comptime] config: G,
    ) {
        let line_read = tensor_reader.load_coalesced_in_tile::<G>(
            tile.0,
            tile.1,
            line_index_within_tile * this.line_size,
            this.input_ident,
            config,
        );

        let offset = this.num_lines_to_skip + line_index_within_tile + num_lines_to_skip_local;

        stage.as_slice_mut(this.line_size)[offset] = match quantization {
            CubeOption::Some(quantization) => quantization.dequantize(line_read, this.input_ident),
            CubeOption::None => Line::cast_from(line_read),
        };
    }
}
