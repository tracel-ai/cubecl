use crate::matmul::components::global::tensor_view::{TensorReader, Window};
use crate::matmul::components::global::{GlobalConfig, LoadingValidation};
use crate::matmul::components::stage::StridedTilingLayout;
use crate::matmul::components::{Ident, InvalidConfigError, MatrixLayout};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::loader::{AsyncLoadingStrategy, CopyMechanism};

#[derive(CubeType, Clone, Copy)]
/// Loads global memory into the stage without modification,  
/// dividing the stage into the smallest possible contiguous slices.  
///
/// Each `memcpy_async` is called with the same arguments for cooperative behaviour
pub struct WindowCooperativeLoading {}

impl LoadingValidation for WindowCooperativeLoading {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        if config.check_row_bounds(ident) || config.check_col_bounds(ident) {
            return Err(Box::new(
                "Check bounds are not yet supported on window loading.",
            ));
        }

        if config.transpose_load(ident) {
            return Err(Box::new(
                "Transpose load is not supported with window loading.",
            ));
        }

        Ok(())
    }
}

#[cube]
impl AsyncLoadingStrategy for WindowCooperativeLoading {
    type TilingLayout = StridedTilingLayout;

    fn load<EG: Numeric, ES: Numeric, G: GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EG>,
        stage_slice: &mut SliceMut<Line<ES>>,
        mechanism: CM,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let matrix_layout = config.matrix_layout(ident);
        let tiling_dimensions = config.tiling_dimensions(ident);

        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => tiling_dimensions.total_row(),
            MatrixLayout::ColMajor => tiling_dimensions.total_col(),
        };

        for nth_slice in 0..num_slices {
            let window: Window<EG> = read_view.load_window_no_tile::<G>(nth_slice, ident, config);
            let mut destination: SliceMut<Line<ES>> =
                StridedTilingLayout::nth_slice::<ES, G::SmmConfig>(
                    stage_slice,
                    nth_slice,
                    ident,
                    config.to_smm_config(),
                );

            CM::memcpy_async(
                &mechanism,
                &window.slice.try_cast_unchecked(),
                &mut destination,
            );
        }
    }
}
