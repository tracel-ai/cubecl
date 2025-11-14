use std::marker::PhantomData;

use crate::components::global::GlobalReaderConfig;
use crate::components::global::read::{PartialLoadingStrategy, sync::Synchronous};
use crate::components::global::{RoleRule, read::tiled::TiledLayout};
use crate::components::stage::TilingOrderEnum;
use crate::components::{FormattedConfigError, InvalidConfigError, MatmulIdent, TilingScheme};
use crate::components::{
    global::memory::GlobalIterator,
    stage::{ContiguousTilingLayout, StridedStage, TilingOrder},
};
use crate::components::{global::multi_stage::LoadMaxRoundPlaneCount, stage::TilingValidation};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::layout::Coords2d;

use super::{LoadingJob, LoadingValidation};

#[derive(CubeType, Clone, Copy)]
/// Each tile is guaranteed to be loaded entirely by the same plane.
/// Each plane can load multiple tiles, provided the number of planes evenly divides the number of tiles.
/// In this case, a plane loads contiguous tiles following the `TilingOrder`,
/// until it would otherwise write to the opposite stage. At that point, it continues on the next
/// row or column of the same stage, skipping over the memory region of the other stage.
///
/// Only supports RowMajorTilingOrder for Lhs and ColMajorTilingOrder for Rhs
pub struct SyncPartialTilewiseLoading<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<TO: TilingOrder> LoadMaxRoundPlaneCount for SyncPartialTilewiseLoading<TO> {
    fn max_round_plane_count(
        tiling_scheme: &TilingScheme,
        ident: MatmulIdent,
        _line_size: u8,
        _plane_dim: u32,
    ) -> u32 {
        tiling_scheme.tiles_in_stage(ident)
    }
}

impl<T: TilingOrder> LoadingValidation for SyncPartialTilewiseLoading<T> {
    fn check<R: Runtime>(
        _client: &ComputeClient<R::Server>,
        config: &GlobalReaderConfig,
        ident: MatmulIdent,
    ) -> Result<(), InvalidConfigError> {
        let line_size = config.global_line_size(ident);
        let num_planes = config.num_loading_planes(ident);
        let num_tiles = config.tiling_scheme().tiles_in_stage(ident);

        if !num_tiles.is_multiple_of(num_planes) {
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

        match ident {
            MatmulIdent::Lhs => {
                if !matches!(T::to_enum(), TilingOrderEnum::RowMajor) {
                    return Err(FormattedConfigError::new(move || {
                        "Sync partial tilewise on Lhs is only supported with RowMajor tiling order"
                            .to_string()
                    }));
                }
            }
            MatmulIdent::Rhs => {
                if !matches!(T::to_enum(), TilingOrderEnum::ColMajor) {
                    return Err(FormattedConfigError::new(move || {
                        "Sync partial tilewise on Rhs is only supported with ColMajor tiling order"
                            .to_string()
                    }));
                }
            }
            MatmulIdent::Out => unreachable!(),
        }

        ContiguousTilingLayout::<T>::check(config.global_memory_config(ident))?;

        Ok(())
    }
}

#[cube]
impl<TO: TilingOrder> PartialLoadingStrategy for SyncPartialTilewiseLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = Synchronous;
    type Job<EG: Numeric, ES: Numeric> = SyncPartialTilewiseJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] stage_index: u32,
        #[comptime] ident: MatmulIdent,
        #[comptime] line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> SyncPartialTilewiseJob {
        let num_planes = config.num_loading_planes(ident);
        let num_tiles = config.tiling_scheme().tiles_in_stage(ident);
        let plane_dim = config.plane_dim();

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile =
            comptime!(config.tiling_scheme().elements_in_tile(ident) / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_lines_per_unit = num_lines_per_plane / plane_dim;

        let stage_width = comptime!(match ident {
            MatmulIdent::Lhs => config.tiling_scheme().tiles_in_stage_col(ident),
            MatmulIdent::Rhs => config.tiling_scheme().tiles_in_stage_row(ident),
            MatmulIdent::Out => unreachable!(),
        });

        let num_tiles_to_skip = RoleRule::new(config.role_rule_config())
            .load_index(ident, config.specialized_loading_sides())
            * num_tiles_per_plane;

        SyncPartialTilewiseJob {
            stage_index,
            num_tiles_to_skip,
            stage_width,
            num_lines_per_tile,
            num_lines_per_unit,
            plane_dim: config.plane_dim(),
            line_size,
            ident,
        }
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct SyncPartialTilewiseJob {
    num_tiles_to_skip: u32,
    stage_index: u32,

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
    ident: MatmulIdent,
}

#[cube]
impl<EG: Numeric, ES: Numeric, TO: TilingOrder>
    LoadingJob<EG, ES, ContiguousTilingLayout<TO>, Synchronous> for SyncPartialTilewiseJob
{
    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStage<ES, ContiguousTilingLayout<TO>>,
        _barrier: &mut (),
        #[comptime] config: GlobalReaderConfig,
    ) {
        let mut stage = stage.with_buffer_index(this.stage_index);
        let pos_across_tiles = task_id * this.plane_dim + UNIT_POS_X;
        let nth_tile_for_this_plane = pos_across_tiles / this.num_lines_per_tile;
        let line_index_within_tile = pos_across_tiles % this.num_lines_per_tile;

        let nth_tile_global = this.num_tiles_to_skip + nth_tile_for_this_plane;

        let stage_config = comptime![config.stage_memory_config(this.ident)];

        let tile = TO::to_row_col(
            nth_tile_global,
            stage_config.tiles_in_stage_row,
            stage_config.tiles_in_stage_col,
            stage_config,
        );

        let tile = match comptime![this.ident] {
            MatmulIdent::Lhs => (tile.0, tile.1 + this.stage_index * this.stage_width),
            MatmulIdent::Rhs => (tile.0 + this.stage_index * this.stage_width, tile.1),
            MatmulIdent::Out => tile,
        };

        let num_lines_to_skip_global = nth_tile_global * this.num_lines_per_tile;

        SyncPartialTilewiseJob::load_and_store_line::<EG, ES, TO>(
            this,
            tile,
            line_index_within_tile,
            num_lines_to_skip_global,
            global_iter,
            &mut stage,
            config,
        );
    }

    fn task_count(this: &Self) -> comptime_type!(u32) {
        comptime!(this.num_lines_per_unit)
    }
}

#[cube]
impl SyncPartialTilewiseJob {
    #[allow(clippy::too_many_arguments)]
    fn load_and_store_line<EG: Numeric, ES: Numeric, TO: TilingOrder>(
        this: &Self,
        tile: Coords2d,
        line_index_within_tile: u32,
        num_lines_to_skip_global: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStage<ES, ContiguousTilingLayout<TO>>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let layout = TiledLayout::new(comptime!(config.global_memory_config(this.ident)));
        let view = global_iter.view().view(layout);

        let line_read = view.read_checked((tile, line_index_within_tile * this.line_size));

        let offset = line_index_within_tile + num_lines_to_skip_global;

        stage.as_slice_mut(this.line_size)[offset] = Line::cast_from(line_read);
    }
}
