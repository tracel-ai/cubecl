use std::marker::PhantomData;

use crate::matmul::components::{
    global::{tensor_view::TensorReader, GlobalConfig, LoadingValidation},
    stage::{ContiguousTilingLayout, Stage, TilingOrder},
    FormattedConfigError, Ident, InvalidConfigError,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::SyncLoadingStrategy;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using
/// one plane per tile.
pub struct TilewiseCoalescedLoading<T: TilingOrder> {
    tiling_order: PhantomData<T>,
}

impl<T: TilingOrder> LoadingValidation for TilewiseCoalescedLoading<T> {
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

        Ok(())
    }
}

#[cube]
impl<T: TilingOrder> SyncLoadingStrategy for TilewiseCoalescedLoading<T> {
    type TilingLayout = ContiguousTilingLayout<T>;

    fn load_full<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
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

            let line_read = read_view.load_coalesced_in_tile::<G>(
                tile_x,
                tile_y,
                pos_within_tile * line_size,
                ident,
                config,
            );

            let offset = offset_base + pos_within_tile;
            stage.as_slice_mut()[offset] = Line::cast_from(line_read);
        }
    }

    fn load_buffer<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        _read_view: &TensorReader<EG>,
        _stage: &mut Stage<ES, Self::TilingLayout>,
        _buffer_index: u32,
        #[comptime] _ident: Ident,
        #[comptime] _config: G,
    ) {
        // TODO
    }
}
