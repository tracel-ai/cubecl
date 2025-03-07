use crate::matmul::components::{
    global::{
        tensor_view::{TensorReader, Window},
        GlobalConfig, LoadingValidation,
    },
    stage::{Stage, StridedTilingLayout},
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
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[cube]
impl AsyncLoadingStrategy for WindowCooperativeLoading {
    type TilingLayout = StridedTilingLayout;

    fn load_full<EG: Numeric, ES: Numeric, G: GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
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
                    stage,
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
        _read_view: &TensorReader<EG>,
        _stage: &mut Stage<ES, Self::TilingLayout>,
        _buffer_index: u32,
        _mechanism: &CM,
        #[comptime] _ident: Ident,
        #[comptime] _config: G,
    ) {
        // TODO
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_coop(0u32)
    }
}
