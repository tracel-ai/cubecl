use crate::components::global::GlobalReaderConfig;
use crate::components::global::read::FullLoadingStrategy;
use crate::components::global::read::validate_swizzle_atom_size;
use crate::components::global::{multi_stage::LoadMaxRoundPlaneCount, read::sync::Synchronous};
use crate::components::stage::ContiguousTilingLayout;
use crate::components::stage::OrderedTilingOrder;
use crate::components::{FormattedConfigError, InvalidConfigError, StageIdent};
use crate::components::{MatmulElems, MatmulProblem};
use crate::components::{global::RoleRule, stage::TilingValidation};
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
    fn check<R: Runtime>(
        _client: &ComputeClient<R>,
        _problem: &MatmulProblem,
        config: &GlobalReaderConfig,
        dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        if config.stage_ident != StageIdent::Lhs {
            return Err(FormattedConfigError::new(move || {
                "Ordered loading only available on Lhs".to_string()
            }));
        }

        let line_size = config.gmem_config.line_size;
        let num_planes = config.loading_planes_count();
        let num_tiles = config.smem_config.tiles_per_stage();

        if !num_tiles.is_multiple_of(num_planes) {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Number of planes {num_planes:?} must divide number of tiles {num_tiles:?} for ordered loading.",
                )
            }));
        }

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile = comptime!(config.smem_config.elements_per_tile() / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_planes = config.loading_planes_count();
        let plane_dim = config.plane_dim;
        let rows_per_plane = config.smem_config.tiles_per_stage_along_row() / num_planes;

        if num_lines_per_plane % plane_dim != 0 {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Plane dimension {plane_dim:?} must divide number of lines per plane {num_lines_per_plane:?} for ordered loading.",
                )
            }));
        }

        let tile_count_col = config.smem_config.tiles_per_stage_along_col();
        if num_tiles_per_plane != rows_per_plane * tile_count_col {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Number of tiles per plane {num_tiles_per_plane:?} must equal rows_per_plane {rows_per_plane:?} times cols {tile_count_col:?} for ordered loading.",
                )
            }));
        }

        validate_swizzle_atom_size(config.smem_config, config.stage_ident, dtypes)?;
        ContiguousTilingLayout::<OrderedTilingOrder>::check(config.smem_config)?;

        Ok(())
    }
}

impl LoadMaxRoundPlaneCount for SyncFullOrderedLoading {
    fn max_round_plane_count(
        _elements_per_tile: u32,
        tiles_per_stage: u32,
        _line_size: u8,
        _plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        tiles_per_stage
    }
}

#[cube]
impl FullLoadingStrategy for SyncFullOrderedLoading {
    type TilingLayout = ContiguousTilingLayout<OrderedTilingOrder>;
    type SyncStrategy = Synchronous;
    type Job<EG: Numeric, ES: Numeric> = sync_full_tilewise::SyncFullTilewiseJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES> {
        let num_planes = config.loading_planes_count();
        let num_tiles = config.smem_config.tiles_per_stage();
        let plane_dim = config.plane_dim;

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile = comptime!(config.smem_config.elements_per_tile() / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_lines_per_unit = num_lines_per_plane / plane_dim;

        let num_tiles_to_skip = RoleRule::new(config.plane_role_config.rule)
            .load_index(config.specialization_tensor_config)
            * num_tiles_per_plane;
        let num_lines_to_skip = num_tiles_to_skip * num_lines_per_tile;

        // Ordered is just a tilewise reader using the ordered tiling order
        sync_full_tilewise::SyncFullTilewiseJob {
            num_tiles_to_skip,
            num_lines_to_skip,
            num_lines_per_tile,
            num_lines_per_unit,
            plane_dim,
            line_size,
        }
    }
}
