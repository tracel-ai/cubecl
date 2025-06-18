use std::marker::PhantomData;

use crate::components::global::load::SyncBufferLoadingStrategy;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::global::{Quantization, RoleRule};
use crate::components::stage::TilingOrderEnum;
use crate::components::{
    FormattedConfigError, Ident, InputIdent, InvalidConfigError, MatmulPrecision, TilingScheme,
};
use crate::components::{
    global::{GlobalConfig, tensor_view::TensorReader},
    stage::{ContiguousTilingLayout, StageMemory, TilingOrder},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Each tile is guaranteed to be loaded entirely by the same plane.
/// Each plane can load multiple tiles, provided the number of planes evenly divides the number of tiles.
/// In this case, a plane loads contiguous tiles following the `TilingOrder`,
/// until it would otherwise write to the opposite buffer. At that point, it continues on the next
/// row or column of the same buffer, skipping over the memory region of the other buffer.
///
/// Only supports RowMajorTilingOrder for Lhs and ColMajorTilingOrder for Rhs
pub struct LoadingStrategy<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for LoadingStrategy<TO> {
    fn max_round_plane_count(
        tiling_scheme: &TilingScheme,
        ident: InputIdent,
        _line_size: u8,
        _plane_dim: u32,
    ) -> u32 {
        tiling_scheme.tiles_in_stage(ident)
    }
}

impl<T: TilingOrder> LoadingValidation for LoadingStrategy<T> {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let line_size = config.global_line_size(ident);
        let num_planes = config.num_loading_planes(ident);
        let num_tiles = config.tiling_scheme().tiles_in_stage(ident);

        if num_tiles % num_planes != 0 {
            return Err(FormattedConfigError::new(move || {
                "Number of planes {num_planes:?} must divide number of tiles {num_tiles:?} for tilewise loading.".to_string()
            }));
        }

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile =
            comptime!(config.tiling_scheme().elements_in_tile(ident) / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_planes = config.plane_dim();

        if num_lines_per_plane % num_planes != 0 {
            return Err(FormattedConfigError::new(move || {
                "Number of planes {num_planes:?} must divide number of lines per plane {num_lines_per_plane:?} for tilewise loading.".to_string()
            }));
        }

        match ident.as_input_ident() {
            InputIdent::Lhs => {
                if !matches!(T::to_enum(), TilingOrderEnum::RowMajor) {
                    return Err(FormattedConfigError::new(move || {
                        format!(
                            "Sync buffer tilewise on Lhs is only supported with RowMajor tiling order",
                        )
                    }));
                }
            }
            InputIdent::Rhs => {
                if !matches!(T::to_enum(), TilingOrderEnum::ColMajor) {
                    return Err(FormattedConfigError::new(move || {
                        "Sync buffer tilewise on Rhs is only supported with ColMajor tiling order"
                            .to_string()
                    }));
                }
            }
        }

        if config.plane_role_config().has_specialization() {
            return Err(FormattedConfigError::new(move || {
                format!("Sync buffer tilewise not supported with specialization",)
            }));
        }

        Ok(())
    }
}

#[cube]
impl<TO: TilingOrder> SyncBufferLoadingStrategy for LoadingStrategy<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type Job<MP: MatmulPrecision> = Job;

    fn new_job<MP: MatmulPrecision, G: GlobalConfig>(
        #[comptime] buffer_index: u32,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> Job {
        let line_size = config.global_line_size(input_ident);
        let num_planes = config.num_loading_planes(input_ident);
        let num_tiles = config.tiling_scheme().tiles_in_stage(input_ident);
        let plane_dim = config.plane_dim();

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile =
            comptime!(config.tiling_scheme().elements_in_tile(input_ident) / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_lines_per_unit = num_lines_per_plane / plane_dim;

        let num_stages = config.num_stages(input_ident);
        let stage_width = comptime!(match input_ident {
            InputIdent::Lhs => config.tiling_scheme().tiles_in_stage_col(input_ident),
            InputIdent::Rhs => config.tiling_scheme().tiles_in_stage_row(input_ident),
        });
        let row_col_stride = num_stages * stage_width;
        let buffer_offset = stage_width * buffer_index;

        let starting_tile_within_stage = RoleRule::new(config.role_rule_config())
            .load_index(input_ident, config.specialized_loading_sides())
            * num_tiles_per_plane;
        let row_col_index = starting_tile_within_stage / stage_width;
        let inner_offset = starting_tile_within_stage % stage_width;
        let num_tiles_to_skip = row_col_index * row_col_stride + inner_offset + buffer_offset;

        Job {
            num_tiles_to_skip,
            row_col_stride,
            stage_width,
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

    #[cube(comptime)]
    row_col_stride: u32,
    #[cube(comptime)]
    stage_width: u32,
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

        let row_col_index_local = nth_tile_for_this_plane / this.stage_width;
        let inner_offset = nth_tile_for_this_plane % this.stage_width;
        let num_tiles_to_skip_local = row_col_index_local * this.row_col_stride + inner_offset;
        let nth_tile_global = this.num_tiles_to_skip + num_tiles_to_skip_local;

        let (total_tile_count_row, total_tile_count_col) = match comptime!(this.input_ident) {
            InputIdent::Lhs => (
                comptime!(config.tiling_scheme().tiles_in_stage_m()),
                comptime!(
                    config.tiling_scheme().tiles_in_stage_k() * config.num_stages(InputIdent::Lhs)
                ),
            ),
            InputIdent::Rhs => (
                comptime!(
                    config.tiling_scheme().tiles_in_stage_k() * config.num_stages(InputIdent::Rhs)
                ),
                comptime!(config.tiling_scheme().tiles_in_stage_n()),
            ),
        };

        let tile = TO::to_row_col::<G::StageConfig>(
            nth_tile_global,
            total_tile_count_row,
            total_tile_count_col,
            comptime!(this.input_ident.as_ident()),
            config.stage_config(),
        );

        let num_lines_to_skip_global = nth_tile_global * this.num_lines_per_tile;

        Job::load_and_store_line::<MP, TO, G>(
            this,
            tile,
            line_index_within_tile,
            num_lines_to_skip_global,
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
        num_lines_to_skip_global: u32,
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

        let offset = line_index_within_tile + num_lines_to_skip_global;

        stage.as_slice_mut(this.line_size)[offset] = match quantization {
            CubeOption::Some(quantization) => quantization.dequantize(line_read, this.input_ident),
            CubeOption::None => Line::cast_from(line_read),
        };
    }
}
