use crate::matmul::components::{
    global::{
        tensor_view::{TensorReader, Window},
        GlobalConfig, LoadingValidation,
    },
    stage::StridedTilingLayout,
    Ident, InvalidConfigError, MatrixLayout,
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use super::{AsyncLoadingStrategy, CopyMechanism};

#[derive(CubeType, Clone, Copy)]
/// Loads global memory into the stage without modification,  
/// dividing the stage into the smallest possible contiguous slices.  
///
/// Each `memcpy_async` is called with the same arguments for cooperative behaviour
pub struct WindowCooperativeLoading {}

impl LoadingValidation for WindowCooperativeLoading {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
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

    fn load_full<EG: Numeric, ES: Numeric, G: GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EG>,
        stage_slice: &mut SliceMut<Line<ES>>,
        mechanism: &CM,
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
            let window: Window<EG> = read_view.load_window_in_stage::<G>(nth_slice, ident, config);
            let mut destination: SliceMut<Line<ES>> =
                StridedTilingLayout::nth_slice::<ES, G::SmmConfig>(
                    stage_slice,
                    nth_slice,
                    ident,
                    config.to_smm_config(),
                );

            CM::memcpy_async(
                mechanism,
                &window.slice.try_cast_unchecked(),
                &mut destination,
            );
        }
    }

    fn load_buffer<EG: Numeric, ES: Numeric, G: GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EG>,
        stage_slice: &mut SliceMut<Line<ES>>,
        buffer_index: u32,
        mechanism: &CM,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        // TODO
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_coop(0u32)
    }
}
