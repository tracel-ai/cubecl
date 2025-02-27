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
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
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
        let line_size = config.global_line_size(ident);

        let (num_slices, expected_window_size) = match matrix_layout {
            MatrixLayout::RowMajor => (
                tiling_dimensions.total_row(),
                tiling_dimensions.total_col() / line_size,
            ),
            MatrixLayout::ColMajor => (
                tiling_dimensions.total_col(),
                tiling_dimensions.total_row() / line_size,
            ),
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

            // If padding needed: TODO comptime conditional
            for i in window.size..expected_window_size {
                destination[i] = Line::cast_from(0);
            }
        }
    }
}
