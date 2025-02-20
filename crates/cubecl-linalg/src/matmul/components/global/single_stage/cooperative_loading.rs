use crate::matmul::components::global::tensor_view::{TensorReader, Window};
use crate::matmul::components::global::{GlobalConfig, LoadingValidation};
use crate::matmul::components::stage::TilingLayout;
use crate::matmul::components::{Ident, InvalidConfigError, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use pipeline::Pipeline;

use super::loader::LoadingStrategy;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all tiles in the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct CooperativeLoading {}

impl LoadingValidation for CooperativeLoading {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let tiling_layout = config.tiling_layout(ident);

        // match config.load_mode(ident) {
        //     LoadMode::Coalesced => {
        //         return Err(Box::new(
        //             "Coalesced mode not supported in cooperative loading",
        //         ));
        //     }

        //     LoadMode::Window => {
        //         if let TilingLayout::Contiguous(_) = tiling_layout {
        //             return Err(Box::new(
        //                 "Contiguous tiling layout not supported in cooperative loading",
        //             ));
        //         }
        //     }
        // };

        Ok(())
    }
}

#[cube]
impl LoadingStrategy for CooperativeLoading {
    fn load_window<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        _read_view: &TensorReader<EG>,
        _stage_slice: &mut SliceMut<Line<ES>>,
        _pipeline: Pipeline<ES>,
        #[comptime] _ident: Ident,
        #[comptime] _config: G,
    ) {
    }

    fn load_to_slice<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        read_view: &TensorReader<EG>,
        stage_slice: &mut SliceMut<Line<ES>>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        // This is for testing when memcpy_async is not available.
        // It is extremely inefficient as all units perform all the load

        let matrix_layout = config.matrix_layout(ident);
        let tiling_dimensions = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);
        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => tiling_dimensions.total_row(),
            MatrixLayout::ColMajor => tiling_dimensions.total_col(),
        };
        let expected_window_size = match matrix_layout {
            MatrixLayout::RowMajor => tiling_dimensions.total_col(),
            MatrixLayout::ColMajor => tiling_dimensions.total_row(),
        } / line_size;

        for nth_slice in 0..num_slices {
            let window: Window<EG> = read_view.load_window_no_tile::<G>(nth_slice, ident, config);
            let mut destination: SliceMut<Line<ES>> = TilingLayout::nth_slice::<ES, G::SmmConfig>(
                stage_slice,
                nth_slice,
                ident,
                config.to_smm_config(),
            );
            memcpy_slow(window.slice.try_cast_unchecked(), &mut destination);

            // If padding needed: TODO comptime conditional
            for i in window.size..expected_window_size {
                destination[i] = Line::cast_from(0);
            }
        }
    }
}

#[cube]
fn memcpy_slow<ES: Numeric>(source: Slice<Line<ES>>, destination: &mut SliceMut<Line<ES>>) {
    for i in 0..source.len() {
        destination[i] = source[i];
    }
}
