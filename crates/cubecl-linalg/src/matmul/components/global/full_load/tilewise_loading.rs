use crate::matmul::components::global::args::GmmArgs;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::Config;
use crate::matmul::components::stage::{
    ColMajorTiling, RowMajorTiling, TilingOrder, TilingOrderConfig,
};
use crate::matmul::components::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::loader::LoadingStrategy;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using
/// one plane per tile.
pub struct TilewiseLoading {}

#[cube]
impl LoadingStrategy for TilewiseLoading {
    fn load_to_slice<GA: GmmArgs<EG>, EG: Numeric, ES: Numeric, G: Config>(
        read_view: &TensorReader<GA, EG>,
        slice: &mut SliceMut<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let stage_dim = config.stage_dim(ident);
        let line_size = config.global_line_size(ident);

        #[allow(clippy::all)]
        let _ = comptime! {
            check_num_planes(config.num_planes(), stage_dim.num_tiles());
            check_line_sizes(line_size, config.stage_line_size(ident))
        };

        let num_lines_per_tile = comptime!(stage_dim.tile_num_elements() / line_size);

        let nth_tile = UNIT_POS_Y;
        let offset_base = num_lines_per_tile * nth_tile;

        let num_loads_per_unit = num_lines_per_tile / config.plane_dim();

        let (tile_x, tile_y) = match config.tiling_order(ident) {
            TilingOrderConfig::RowMajor => RowMajorTiling::to_x_y(
                nth_tile,
                stage_dim.num_tiles_x_dim(),
                stage_dim.num_tiles_y_dim(),
            ),
            TilingOrderConfig::ColMajor => ColMajorTiling::to_x_y(
                nth_tile,
                stage_dim.num_tiles_x_dim(),
                stage_dim.num_tiles_y_dim(),
            ),
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

            match config.transpose_load(ident) {
                false => slice[offset] = Line::cast_from(line_read),
                true => {
                    #[allow(clippy::all)]
                    let _ = comptime!(unsupported_transpose_load());
                }
            }
        }
    }
}

fn check_num_planes(num_planes: u32, num_tiles: u32) {
    assert!(
        num_planes == num_tiles,
        "Number of planes {:?} must equal number of tiles {:?} for tilewise loading.",
        num_planes,
        num_tiles
    );
}

fn check_line_sizes(global_line_size: u32, stage_line_size: u32) {
    assert!(
        global_line_size == stage_line_size,
        "Global and stage line sizes must match for tilewise loading."
    );
}

fn unsupported_transpose_load() {
    panic!("Transpose load not yet supported in tilewise loading setup")
}
