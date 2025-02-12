use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation};
use crate::matmul::components::stage::{
    ColMajorTiling, RowMajorTiling, TilingOrder, TilingOrderConfig,
};
use crate::matmul::components::{FormattedConfigError, Ident, InvalidConfigError};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use pipeline::Pipeline;

use super::loader::LoadingStrategy;
use super::LoadMode;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using
/// one plane per tile.
pub struct TilewiseLoading {}

impl LoadingValidation for TilewiseLoading {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let tiling = config.stage_tiling(ident);
        let line_size = config.global_line_size(ident);

        let num_planes = config.num_planes();
        let num_tiles = tiling.tile_count();

        if num_planes != num_tiles {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Number of planes {:?} must equal number of tiles {:?} for tilewise loading.",
                    num_planes, num_tiles,
                )
            }));
        }

        if line_size != config.stage_line_size(ident) {
            return Err(Box::new(
                "Global and stage line sizes must match for tilewise loading.",
            ));
        }

        if config.transpose_load(ident) {
            return Err(Box::new(
                "Transpose load not yet supported in tilewise loading setup",
            ));
        }

        if let LoadMode::Window = config.load_mode() {
            return Err(Box::new(
                "Window load not yet supported in tilewise loading setup",
            ));
        }

        Ok(())
    }
}

#[cube]
impl LoadingStrategy for TilewiseLoading {
    fn load_window<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        _read_view: &TensorReader<EG>,
        _slice: &mut SliceMut<Line<ES>>,
        _pipeline: Pipeline<ES>,
        #[comptime] _ident: Ident,
        #[comptime] _config: G,
    ) {
        let _ = unimplemented!();
    }

    fn load_to_slice<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        read_view: &TensorReader<EG>,
        slice: &mut SliceMut<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let tiling = config.stage_tiling(ident);
        let line_size = config.global_line_size(ident);

        let num_lines_per_tile = comptime!(tiling.tile_size() / line_size);

        let nth_tile = UNIT_POS_Y;
        let offset_base = num_lines_per_tile * nth_tile;

        let num_loads_per_unit = num_lines_per_tile / config.plane_dim();

        let (tile_x, tile_y) = match config.tiling_order(ident) {
            TilingOrderConfig::RowMajor => {
                RowMajorTiling::to_x_y(nth_tile, tiling.tile_count_row(), tiling.tile_count_col())
            }
            TilingOrderConfig::ColMajor => {
                ColMajorTiling::to_x_y(nth_tile, tiling.tile_count_row(), tiling.tile_count_col())
            }
        };

        for i in 0..num_loads_per_unit {
            let pos_within_tile = i * config.plane_dim() + UNIT_POS_X;

            let line_read = read_view.load_coalesced::<G>(
                tile_x,
                tile_y,
                pos_within_tile * line_size,
                ident,
                config,
            );

            let offset = offset_base + pos_within_tile;
            slice[offset] = Line::cast_from(line_read);
        }
    }
}
