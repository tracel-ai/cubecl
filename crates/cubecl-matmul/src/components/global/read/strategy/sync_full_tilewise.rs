use std::marker::PhantomData;

use crate::components::global::GlobalReaderConfig;
use crate::components::global::read::validate_swizzle_atom_size;
use crate::components::global::read::{FullLoadingStrategy, sync::Synchronous};
use crate::components::global::{RoleRule, read::tiled::TiledLayout};
use crate::components::stage::StridedStageFamily;
use crate::components::stage::{StridedStageMemory, TilingOrder};
use crate::components::{FormattedConfigError, InvalidConfigError};
use crate::components::{MatmulElems, MatmulProblem};
use crate::components::{global::memory::GlobalIterator, stage::ContiguousTilingLayout};
use crate::components::{global::multi_stage::LoadMaxRoundPlaneCount, stage::TilingValidation};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{tensor::layout::Coords2d, type_size};

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
        _elements_per_tile: u32,
        tiles_per_stage: u32,
        _line_size: u8,
        _plane_dim: u32,
        _dtype: StorageType,
    ) -> u32 {
        tiles_per_stage
    }
}

impl<T: TilingOrder> LoadingValidation for SyncFullTilewiseLoading<T> {
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
                format!(
                    "Number of planes {num_planes:?} must divide number of tiles {num_tiles:?} for tilewise loading.",
                )
            }));
        }

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile = comptime!(config.smem_config.elements_per_tile() / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let plane_dim = config.plane_dim;

        if num_lines_per_plane % plane_dim != 0 {
            return Err(FormattedConfigError::new(move || {
                format!(
                    "Plane dimension {plane_dim:?} must divide number of lines per plane {num_lines_per_plane:?} for tilewise loading.",
                )
            }));
        }

        validate_swizzle_atom_size(config.smem_config, config.stage_ident, dtypes)?;
        ContiguousTilingLayout::<T>::check(config.smem_config)?;

        Ok(())
    }
}

#[cube]
impl<TO: TilingOrder> FullLoadingStrategy for SyncFullTilewiseLoading<TO> {
    type TilingLayout = ContiguousTilingLayout<TO>;
    type SyncStrategy = Synchronous;
    type Job<EG: Numeric, ES: Numeric> = SyncFullTilewiseJob;

    fn new_job<EG: Numeric, ES: Numeric>(
        #[comptime] line_size: u32,
        #[comptime] config: GlobalReaderConfig,
    ) -> Self::Job<EG, ES> {
        let num_planes = config.loading_planes_count();
        let num_tiles = config.smem_config.tiles_per_stage();

        let num_tiles_per_plane = comptime!(num_tiles / num_planes);
        let num_lines_per_tile = comptime!(config.smem_config.elements_per_tile() / line_size);
        let num_lines_per_plane = num_lines_per_tile * num_tiles_per_plane;
        let num_lines_per_unit = num_lines_per_plane / config.plane_dim;

        let num_tiles_to_skip = RoleRule::new(config.plane_role_config.rule)
            .load_index(config.specialization_tensor_config)
            * num_tiles_per_plane;
        let num_lines_to_skip = num_tiles_to_skip * num_lines_per_tile;

        SyncFullTilewiseJob {
            num_tiles_to_skip,
            num_lines_to_skip,
            num_lines_per_tile,
            num_lines_per_unit,
            plane_dim: config.plane_dim,
            line_size,
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
}

#[cube]
impl<EG: Numeric, ES: Numeric, TO: TilingOrder>
    LoadingJob<EG, ES, ContiguousTilingLayout<TO>, Synchronous> for SyncFullTilewiseJob
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
        let pos_across_tiles = task_id * this.plane_dim + UNIT_POS_X;
        let nth_tile_for_this_plane = pos_across_tiles / this.num_lines_per_tile;
        let line_index_within_tile = pos_across_tiles % this.num_lines_per_tile;

        let nth_tile_global = nth_tile_for_this_plane + this.num_tiles_to_skip;
        let tile =
            ContiguousTilingLayout::<TO>::to_x_y(nth_tile_global, comptime!(config.smem_config));

        SyncFullTilewiseJob::load_and_store_line::<EG, ES, TO>(
            this,
            tile,
            line_index_within_tile,
            nth_tile_for_this_plane * this.num_lines_per_tile,
            global_iter,
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
    fn load_and_store_line<EG: Numeric, ES: Numeric, TO: TilingOrder>(
        this: &Self,
        tile: Coords2d,
        line_index_within_tile: u32,
        num_lines_to_skip_local: u32,
        global_iter: &GlobalIterator<Line<EG>>,
        stage: &mut StridedStageMemory<ES, ContiguousTilingLayout<TO>>,
        #[comptime] config: GlobalReaderConfig,
    ) {
        let layout = TiledLayout::new(config.stage_ident, config.smem_config);
        let view = global_iter.view().view(layout);

        let line_read = view.read_checked((tile, line_index_within_tile * this.line_size));

        let offset = this.num_lines_to_skip + line_index_within_tile + num_lines_to_skip_local;
        let type_size = type_size::<ES>(this.line_size);
        let offset = stage.swizzle.apply(offset, type_size);

        stage.as_slice_mut(this.line_size)[offset] = Line::cast_from(line_read);
    }
}
