use std::marker::PhantomData;

use crate::components::global::RoleRule;
use crate::components::global::load::SyncFullLoadingStrategy;
use crate::components::global::multi_stage::LoadMaxRoundPlaneCount;
use crate::components::{
    FormattedConfigError, InputPrecision, InvalidConfigError, MatmulIdent, TilingScheme,
};
use crate::components::{
    global::{GlobalConfig, memory::TensorReader},
    stage::{ContiguousTilingLayout, StageMemory, TilingOrder},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Each tile is guaranteed to be loaded entirely by the same plane.
/// Each plane can load multiple tiles, provided the number of planes evenly divides the number of tiles.
/// In this case, a plane loads contiguous tiles following the TilingOrder.
///
/// If number of planes = number of rows of Lhs and TilingOrder is RowMajor,
/// each plane loads its own row and a sync can be saved.
/// In multi-row, number of planes must divide number of rows,
/// and each plane loads a contiguous chunk of rows (e.g. plane 0 loads rows 0–1, plane 1 loads 2–3, etc.).
pub struct SyncFullTilewiseLoading<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for SyncFullTilewiseLoading<TO> {
    fn max_round_plane_count(
        tiling_scheme: &TilingScheme,
        ident: MatmulIdent,
        _line_size: u8,
        _plane_dim: u32,
    ) -> u32 {
        tiling_scheme.tiles_in_stage(ident)
    }
}

impl<T: TilingOrder> LoadingValidation for SyncFullTilewiseLoading<T> {
    fn check<C: GlobalConfig>(config: &C, ident: MatmulIdent) -> Result<(), InvalidConfigError> {
        let line_size = config.global_line_size(ident);
        let num_planes = config.num_loading_planes(ident);
        let num_tiles = config.tiling_scheme().tiles_in_stage(ident);

        if num_tiles % num_planes != 0 {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Number of planes {num_planes:?} must divide number of tiles {num_tiles:?} for tilewise loading.",
                )
            }));
        }

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile =
            comptime!(config.tiling_scheme().elements_in_tile(ident) / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let plane_dim = config.plane_dim();

        if num_lines_per_plane % plane_dim != 0 {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Plane dimension {plane_dim:?} must divide number of lines per plane {num_lines_per_plane:?} for tilewise loading.",
                )
            }));
        }

        Ok(())
    }
}

#[cube]
impl<TO: TilingOrder> SyncFullLoadingStrategy for SyncFullTilewiseLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type Job<IP: InputPrecision> = SyncFullTilewiseJob;

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

        SyncFullTilewiseJob {
            num_tiles_to_skip,
            num_lines_to_skip,
            num_lines_per_tile,
            num_lines_per_unit,
            plane_dim: config.plane_dim(),
            line_size,
            ident,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct SyncFullTilewiseJob {
    pub num_tiles_to_skip: u32,
    pub num_lines_to_skip: u32,

    #[cube(comptime)]
    pub num_lines_per_tile: u32,
    #[cube(comptime)]
    pub num_lines_per_unit: u32,
    #[cube(comptime)]
    pub plane_dim: u32,
    #[cube(comptime)]
    pub line_size: u32,
    #[cube(comptime)]
    pub ident: MatmulIdent,
}

#[cube]
impl<IP: InputPrecision, TO: TilingOrder> LoadingJob<IP, ContiguousTilingLayout<TO>>
    for SyncFullTilewiseJob
{
    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        #[comptime] task_id: u32,
        tensor_reader: &TensorReader<IP::Global>,
        stage: &mut StageMemory<IP::Stage, ContiguousTilingLayout<TO>>,
        #[comptime] config: G,
    ) {
        let pos_across_tiles = task_id * this.plane_dim + UNIT_POS_X;
        let nth_tile_for_this_plane = pos_across_tiles / this.num_lines_per_tile;
        let line_index_within_tile = pos_across_tiles % this.num_lines_per_tile;

        let nth_tile_global = nth_tile_for_this_plane + this.num_tiles_to_skip;
        let tile = ContiguousTilingLayout::<TO>::to_x_y::<G::StageMemoryConfig>(
            nth_tile_global,
            comptime!(this.ident.into_stage()),
            config.stage_memory_config(),
        );

        SyncFullTilewiseJob::load_and_store_line::<IP, TO, G>(
            this,
            tile,
            line_index_within_tile,
            nth_tile_for_this_plane * this.num_lines_per_tile,
            tensor_reader,
            stage,
            config,
        );
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        comptime!(this.num_lines_per_unit)
    }
}

#[cube]
impl SyncFullTilewiseJob {
    #[allow(clippy::too_many_arguments)]
    fn load_and_store_line<IP: InputPrecision, TO: TilingOrder, G: GlobalConfig>(
        this: &Self,
        tile: (u32, u32),
        line_index_within_tile: u32,
        num_lines_to_skip_local: u32,
        tensor_reader: &TensorReader<IP::Global>,
        stage: &mut StageMemory<IP::Stage, ContiguousTilingLayout<TO>>,
        #[comptime] config: G,
    ) {
        let line_read = tensor_reader.load_coalesced_in_tile(
            tile.0,
            tile.1,
            line_index_within_tile * this.line_size,
            comptime!(config.global_memory_config(this.ident)),
        );

        let offset = this.num_lines_to_skip + line_index_within_tile + num_lines_to_skip_local;

        stage.as_slice_mut(this.line_size)[offset] = Line::cast_from(line_read);
    }
}
