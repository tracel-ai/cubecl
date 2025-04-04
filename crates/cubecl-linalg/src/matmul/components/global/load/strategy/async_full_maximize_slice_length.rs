use crate::matmul::components::{
    Ident, InputIdent, InvalidConfigError, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig, LoadingValidation,
        tensor_view::{TensorReader, Window},
    },
    stage::{Stage, StridedTilingLayout},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

use super::AsyncFullLoadingStrategy;

#[derive(CubeType, Clone, Copy)]
/// Executes one memcpy_async call per contiguous slice.
/// The goal is to reduce the total number of memcpy_async calls, though it may result in idle threads.
pub struct AsyncFullMaximizeSliceLengthLoading {}

impl LoadingValidation for AsyncFullMaximizeSliceLengthLoading {
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[cube]
impl AsyncFullLoadingStrategy for AsyncFullMaximizeSliceLengthLoading {
    type TilingLayout = StridedTilingLayout;

    fn load_full<EG: Numeric, ES: Numeric, G: GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        mechanism: &CM,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) {
        let matrix_layout = config.matrix_layout(input_ident);
        let tiling_dimensions = config.tiling_dimensions(input_ident);

        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => tiling_dimensions.total_row(),
            MatrixLayout::ColMajor => tiling_dimensions.total_col(),
        };
        let unit_count = config.plane_dim() * config.num_planes();

        let slices_per_unit = (num_slices + unit_count - 1) / unit_count;

        // Typically there will be only 1 slice per unit
        #[unroll(slices_per_unit==1)]
        for nth_slice_local in 0..slices_per_unit {
            let nth_slice = unit_count * nth_slice_local + UNIT_POS;

            #[allow(clippy::collapsible_else_if)]
            if comptime!(num_slices % unit_count == 0) {
                load_nth_slice::<EG, ES, CM, G>(
                    nth_slice,
                    read_view,
                    stage,
                    mechanism,
                    input_ident,
                    config,
                );
            } else {
                if nth_slice < num_slices {
                    load_nth_slice::<EG, ES, CM, G>(
                        nth_slice,
                        read_view,
                        stage,
                        mechanism,
                        input_ident,
                        config,
                    );
                }
            };
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[cube]
fn load_nth_slice<EG: Numeric, ES: Numeric, CM: CopyMechanism<ES>, G: GlobalConfig>(
    nth_slice: u32,
    read_view: &TensorReader<EG>,
    stage: &mut Stage<ES, StridedTilingLayout>,
    mechanism: &CM,
    #[comptime] input_ident: InputIdent,
    #[comptime] config: G,
) {
    let window: Window<EG> = read_view.load_window_in_stage::<G>(nth_slice, input_ident, config);
    let mut destination: SliceMut<Line<ES>> = StridedTilingLayout::nth_slice::<ES, G::SmmConfig>(
        stage,
        nth_slice,
        comptime!(input_ident.as_ident()),
        config.to_smm_config(),
    );

    CM::memcpy_async(
        mechanism,
        &window.slice.try_cast_unchecked(),
        &mut destination,
    );
}
