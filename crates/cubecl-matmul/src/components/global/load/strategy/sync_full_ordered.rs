use crate::components::global::RoleRule;
use crate::components::global::load::SyncFullLoadingStrategy;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::stage::OrderedTilingOrder;
use crate::components::{
    FormattedConfigError, InputPrecision, InvalidConfigError, MatmulIdent, TilingScheme,
};
use crate::components::{global::GlobalConfig, stage::ContiguousTilingLayout};
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
pub struct SyncFullOrderedLoading {}

impl LoadingValidation for SyncFullOrderedLoading {
    fn check<C: GlobalConfig>(config: &C, ident: MatmulIdent) -> Result<(), InvalidConfigError> {
        if ident != MatmulIdent::Lhs {
            return Err(FormattedConfigError::new(move || {
                "Ordered loading only available on Lhs".to_string()
            }));
        }

        let line_size = config.global_line_size(ident);
        let num_planes = config.num_loading_planes(ident);
        let num_tiles = config.tiling_scheme().tiles_in_stage(ident);

        if !num_tiles.is_multiple_of(num_planes) {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Number of planes {num_planes:?} must divide number of tiles {num_tiles:?} for ordered loading.",
                )
            }));
        }

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile =
            comptime!(config.tiling_scheme().elements_in_tile(ident) / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_planes = config.num_loading_planes(ident);
        let plane_dim = config.plane_dim();
        let rows_per_plane = config.tiling_scheme().tiles_in_stage_row(ident) / num_planes;

        if num_lines_per_plane % plane_dim != 0 {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Plane dimension {plane_dim:?} must divide number of lines per plane {num_lines_per_plane:?} for ordered loading.",
                )
            }));
        }

        let tile_count_col = config.tiling_scheme().tiles_in_stage_col(ident);
        if num_tiles_per_plane != rows_per_plane * tile_count_col {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Number of tiles per plane {num_tiles_per_plane:?} must equal rows_per_plane {rows_per_plane:?} times cols {tile_count_col:?} for ordered loading.",
                )
            }));
        }

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for SyncFullOrderedLoading {
    fn max_round_plane_count(
        tiling_scheme: &TilingScheme,
        ident: MatmulIdent,
        _line_size: u8,
        _plane_dim: u32,
    ) -> u32 {
        tiling_scheme.tiles_in_stage(ident)
    }
}

#[cube]
impl SyncFullLoadingStrategy for SyncFullOrderedLoading {
    type TilingLayout = ContiguousTilingLayout<OrderedTilingOrder>;
    type Job<IP: InputPrecision> = sync_full_tilewise::SyncFullTilewiseJob;

    fn new_job<IP: InputPrecision, G: GlobalConfig>(
        #[comptime] ident: MatmulIdent,
        #[comptime] config: G,
    ) -> Self::Job<IP> {
        let line_size = config.global_line_size(ident);
        let num_planes = config.num_loading_planes(ident);
        let num_tiles = config.tiling_scheme().tiles_in_stage(ident);
        let plane_dim = config.plane_dim();

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile =
            comptime!(config.tiling_scheme().elements_in_tile(ident) / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_lines_per_unit = num_lines_per_plane / plane_dim;

        let num_tiles_to_skip = RoleRule::new(config.role_rule_config())
            .load_index(ident, config.specialized_loading_sides())
            * num_tiles_per_plane;
        let num_lines_to_skip = num_tiles_to_skip * num_lines_per_tile;

        // Ordered is just a tilewise loader using the ordered tiling order
        sync_full_tilewise::SyncFullTilewiseJob {
            num_tiles_to_skip,
            num_lines_to_skip,
            num_lines_per_tile,
            num_lines_per_unit,
            plane_dim,
            line_size,
            ident,
        }
    }
}
