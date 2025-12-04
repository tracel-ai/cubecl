use std::marker::PhantomData;

use crate::components::global::GlobalReaderConfig;
use crate::components::global::read::validate_swizzle_atom_size;
use crate::components::global::read::{PartialLoadingStrategy, sync::Synchronous};
use crate::components::global::{RoleRule, read::tiled::TiledLayout};
use crate::components::stage::StridedStageFamily;
use crate::components::stage::StridedStageMemory;
use crate::components::stage::TilingOrderEnum;
use crate::components::{FormattedConfigError, InvalidConfigError, StageIdent};
use crate::components::{MatmulElems, MatmulProblem};
use crate::components::{
    global::memory::GlobalIterator,
    stage::{ContiguousTilingLayout, TilingOrder},
};
use crate::components::{global::multi_stage::LoadMaxRoundPlaneCount, stage::TilingValidation};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{tensor::layout::Coords2d, type_size};

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
        _elements_per_tile: u32,
        tiles_per_stage: u32,
        _line_size: u8,
        _plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        tiles_per_stage
    }
}

impl<T: TilingOrder> LoadingValidation for SyncPartialTilewiseLoading<T> {
    fn check<R: Runtime>(
        _client: &ComputeClient<R>,
        _problem: &MatmulProblem,
        config: &GlobalReaderConfig,
        dtypes: &MatmulElems,
    ) -> Result<(), InvalidConfigError> {
        let line_size = config.gmem_config.line_size;
        let num_planes = config.loading_planes_count();
        let num_tiles = config.smem_config.tiles_per_stage();

        if !num_tiles.is_multiple_of(num_planes) {
            return Err(FormattedConfigError::new(move || {
                "Number of planes {num_planes:?} must divide number of tiles {num_tiles:?} for tilewise loading.".to_string()
            }));
        }

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile = comptime!(config.smem_config.elements_per_tile() / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_planes = config.plane_dim;

        if num_lines_per_plane % num_planes != 0 {
            return Err(FormattedConfigError::new(move || {
                "Number of planes {num_planes:?} must divide number of lines per plane {num_lines_per_plane:?} for tilewise loading.".to_string()
            }));
        }

        match config.stage_ident {
            StageIdent::Lhs => {
                if !matches!(T::to_enum(), TilingOrderEnum::RowMajor) {
                    return Err(FormattedConfigError::new(move || {
                        "Sync partial tilewise on Lhs is only supported with RowMajor tiling order"
                            .to_string()
                    }));
                }
            }
            StageIdent::Rhs => {
                if !matches!(T::to_enum(), TilingOrderEnum::ColMajor) {
                    return Err(FormattedConfigError::new(move || {
                        "Sync partial tilewise on Rhs is only supported with ColMajor tiling order"
                            .to_string()
                    }));
                }
            }
            _ => unreachable!(),
        }

        validate_swizzle_atom_size(config.smem_config, config.stage_ident, dtypes)?;
        ContiguousTilingLayout::<T>::check(config.smem_config)?;

        Ok(())
    }
}

#[cube]
impl<TO: TilingOrder> PartialLoadingStrategy for SyncPartialTilewiseLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = Synchronous;
    type Stage = StridedStageFamily;

    type Job<EG: Numeric, ES: Numeric> = SyncPartialTilewiseJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] stage_index: u32,
        #[comptime] line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> SyncPartialTilewiseJob {
        let num_planes = config.loading_planes_count();
        let num_tiles = config.smem_config.tiles_per_stage();
        let plane_dim = config.plane_dim;

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile = comptime!(config.smem_config.elements_per_tile() / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_lines_per_unit = num_lines_per_plane / plane_dim;

        let stage_width = comptime!(match config.stage_ident {
            StageIdent::Lhs => config.smem_config.tiles_per_stage_along_col(),
            StageIdent::Rhs => config.smem_config.tiles_per_stage_along_row(),
            _ => unreachable!(),
        });

        let num_tiles_to_skip = RoleRule::new(config.plane_role_config.rule)
            .load_index(config.specialization_tensor_config)
            * num_tiles_per_plane;

        SyncPartialTilewiseJob {
            stage_index,
            num_tiles_to_skip,
            stage_width,
            num_lines_per_tile,
            num_lines_per_unit,
            plane_dim,
            line_size,
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
}

#[cube]
impl<EG: Numeric, ES: Numeric, TO: TilingOrder>
    LoadingJob<EG, ES, ContiguousTilingLayout<TO>, Synchronous> for SyncPartialTilewiseJob
{
    type Stage = StridedStageFamily;

    fn execute_task(
        this: &mut Self,
        #[comptime] task_id: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, ContiguousTilingLayout<TO>>,
        _barrier: &mut (),
        #[comptime] config: GlobalReaderConfig,
    ) {
        let mut stage = stage.with_buffer_index(this.stage_index);
        let pos_across_tiles = task_id * this.plane_dim + UNIT_POS_X;
        let nth_tile_for_this_plane = pos_across_tiles / this.num_lines_per_tile;
        let line_index_within_tile = pos_across_tiles % this.num_lines_per_tile;

        let nth_tile_global = this.num_tiles_to_skip + nth_tile_for_this_plane;

        let tile = TO::to_row_col(
            nth_tile_global,
            config.smem_config.tiles_per_stage_along_row(),
            config.smem_config.tiles_per_stage_along_col(),
            config.smem_config,
        );

        let tile = match comptime![config.stage_ident] {
            StageIdent::Lhs => (tile.0, tile.1 + this.stage_index * this.stage_width),
            StageIdent::Rhs => (tile.0 + this.stage_index * this.stage_width, tile.1),
            _ => tile,
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
        stage: &mut StridedStageMemory<ES, ContiguousTilingLayout<TO>>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let layout = TiledLayout::new(config.stage_ident, config.smem_config);
        let view = global_iter.view().view(layout);

        let line_read = view.read_checked((tile, line_index_within_tile * this.line_size));

        let offset = line_index_within_tile + num_lines_to_skip_global;
        let type_size = type_size::<ES>(this.line_size);
        let offset = stage.swizzle.apply(offset, type_size);

        stage.as_slice_mut(this.line_size)[offset] = Line::cast_from(line_read);
    }
}
