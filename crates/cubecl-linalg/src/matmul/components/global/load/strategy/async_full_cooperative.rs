use crate::matmul::components::{
    global::{
        load::AsyncFullLoadingStrategy, tensor_view::{TensorReader, Window}, CopyMechanism, GlobalConfig, LoadingValidation, Quantization
    }, stage::{Stage, StridedTilingLayout}, Ident, InputIdent, InvalidConfigError, MatmulPrecision, MatrixLayout
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};
use cubecl_std::CubeOption;


#[derive(CubeType, Clone, Copy)]
/// Loads global memory into the stage without modification,  
/// dividing the stage into the smallest possible contiguous slices.  
///
/// Each `memcpy_async` is called with the same arguments for cooperative behaviour
pub struct AsyncFullCooperativeLoading {}

impl LoadingValidation for AsyncFullCooperativeLoading {
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[cube]
impl AsyncFullLoadingStrategy for AsyncFullCooperativeLoading {
    type TilingLayout = StridedTilingLayout;

    fn load_full<MP: MatmulPrecision, G: GlobalConfig, CM: CopyMechanism<MP::ES>>(
        read_view: &TensorReader<MP::EI>,
        stage: &mut Stage<MP::ES, Self::TilingLayout>,
        mechanism: &CM,
        _quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) {
        let matrix_layout = config.matrix_layout(input_ident);
        let tiling_dimensions = config.tiling_dimensions(input_ident);

        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => tiling_dimensions.total_row(),
            MatrixLayout::ColMajor => tiling_dimensions.total_col(),
        };

        for nth_slice in 0..num_slices {
            let window: Window<MP::EI> =
                read_view.load_window_in_stage::<G>(nth_slice, input_ident, config);
            let mut destination: SliceMut<Line<MP::ES>> =
                StridedTilingLayout::nth_slice::<MP::ES, G::SmmConfig>(
                    stage,
                    nth_slice,
                    input_ident.as_ident(),
                    config.to_smm_config(),
                );

            CM::memcpy_async(
                mechanism,
                &window.slice.try_cast_unchecked(),
                &mut destination,
            );
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_coop(0u32)
    }
}
