use std::marker::PhantomData;

use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation};
use crate::matmul::components::stage::{ContiguousTilingLayout, TilingOrderTrait};
use crate::matmul::components::{FormattedConfigError, Ident, InvalidConfigError};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::loader::SyncLoadingStrategy;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using
/// one plane per tile.
pub struct TilewiseCoalescedLoading<T: TilingOrderTrait> {
    tiling_order: PhantomData<T>,
}

impl<T: TilingOrderTrait> LoadingValidation for TilewiseCoalescedLoading<T> {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let tiling = config.tiling_dimensions(ident);
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

        Ok(())
    }
}

#[cube]
impl<T: TilingOrderTrait> SyncLoadingStrategy for TilewiseCoalescedLoading<T> {
    type TilingLayout = ContiguousTilingLayout<T>;

    fn load<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        read_view: &TensorReader<EG>,
        slice: &mut SliceMut<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let tiling = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);

        let num_lines_per_tile = comptime!(tiling.tile_size() / line_size);

        let nth_tile = UNIT_POS_Y;
        let offset_base = num_lines_per_tile * nth_tile;

        let num_loads_per_unit = num_lines_per_tile / config.plane_dim();

        let (tile_x, tile_y) = ContiguousTilingLayout::<T>::to_x_y::<G::SmmConfig>(
            nth_tile,
            ident,
            config.to_smm_config(),
        );

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
