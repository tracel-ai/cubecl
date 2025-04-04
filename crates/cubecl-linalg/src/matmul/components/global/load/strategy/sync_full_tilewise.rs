use std::marker::PhantomData;

use crate::matmul::components::global::Quantization;
use crate::matmul::components::{
    FormattedConfigError, Ident, InputIdent, InvalidConfigError, MatmulPrecision,
};
use crate::matmul::components::{
    global::{GlobalConfig, LoadingValidation, tensor_view::TensorReader},
    stage::{ContiguousTilingLayout, Stage, TilingOrder},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand};

use super::SyncFullLoadingStrategy;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using
/// one plane per tile.
pub struct SyncFullTilewiseLoading<T: TilingOrder> {
    #[cube(comptime)]
    tiling_order: PhantomData<T>,
}

impl<T: TilingOrder> LoadingValidation for SyncFullTilewiseLoading<T> {
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
impl<T: TilingOrder> SyncFullLoadingStrategy for SyncFullTilewiseLoading<T> {
    type TilingLayout = ContiguousTilingLayout<T>;

    fn load_full<MP: MatmulPrecision, G: GlobalConfig>(
        read_view: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, Self::TilingLayout>,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) {
        let tiling = config.tiling_dimensions(input_ident);
        let line_size = config.global_line_size(input_ident);

        let num_lines_per_tile = comptime!(tiling.tile_size() / line_size);

        let nth_tile = UNIT_POS_Y;
        let offset_base = num_lines_per_tile * nth_tile;

        let num_loads_per_unit = num_lines_per_tile / config.plane_dim();

        let (tile_x, tile_y) = ContiguousTilingLayout::<T>::to_x_y::<G::SmmConfig>(
            nth_tile,
            input_ident.as_ident(),
            config.to_smm_config(),
        );

        for i in 0..num_loads_per_unit {
            let pos_within_tile = i * config.plane_dim() + UNIT_POS_X;

            let line_read = read_view.load_coalesced_in_tile::<G>(
                tile_x,
                tile_y,
                pos_within_tile * line_size,
                input_ident,
                config,
            );

            let offset = offset_base + pos_within_tile;

            stage.as_slice_mut()[offset] = match quantization {
                CubeOption::Some(quantization) => quantization.dequantize(line_read),
                CubeOption::None => Line::cast_from(line_read),
            }
        }
    }
}
