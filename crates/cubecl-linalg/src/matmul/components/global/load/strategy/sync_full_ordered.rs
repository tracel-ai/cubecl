use crate::matmul::components::global::load::SyncFullLoadingStrategy;
use crate::matmul::components::stage::OrderedTilingOrder;
use crate::matmul::components::{
    FormattedConfigError, Ident, InputIdent, InvalidConfigError, MatmulPrecision,
};
use crate::matmul::components::{global::GlobalConfig, stage::ContiguousTilingLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{LoadingValidation, sync_full_tilewise};

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
        if ident != Ident::Lhs {
            return Err(FormattedConfigError::new(move || {
                "Ordered loading only available on Lhs".to_string()
            }));
        }

        let tiling = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);

        let num_planes = config.num_planes();
        let num_tiles = tiling.tile_count();

        if num_tiles % num_planes != 0 {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Number of planes {num_planes:?} must divide number of tiles {num_tiles:?} for ordered loading.",
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
                    "Plane dimension {plane_dim:?} must divide number of lines per plane {num_lines_per_plane:?} for ordered loading.",
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

        Ok(())
    }
}

#[cube]
impl SyncFullLoadingStrategy for LoadingStrategy {
    type TilingLayout = ContiguousTilingLayout<OrderedTilingOrder>;
    type Job<MP: MatmulPrecision> = sync_full_tilewise::Job;

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

        // Ordered is just a tilewise loader using the ordered tiling order
        sync_full_tilewise::Job {
            num_tiles_to_skip,
            num_lines_to_skip,
            num_lines_per_tile,
            num_lines_per_unit,
            plane_dim,
            line_size,
            input_ident,
        }
    }
}
