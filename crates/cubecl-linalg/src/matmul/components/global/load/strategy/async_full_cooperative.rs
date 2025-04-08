use crate::matmul::components::{
    Ident, InputIdent, InvalidConfigError, MatmulPrecision, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig, LoadingValidation, Quantization,
        load::AsyncFullLoadingStrategy,
        tensor_view::{TensorReader, Window},
    },
    stage::{Stage, StridedTilingLayout},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};
use cubecl_std::CubeOption;

use super::LoadingJob;

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
    type Job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>> = AsyncFullCooperativeJob<MP, CM>;

    fn load_full<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        read_view: TensorReader<MP::EI>,
        mut stage: Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
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
                    &mut stage,
                    nth_slice,
                    input_ident.as_ident(),
                    config.to_smm_config(),
                );

            CM::memcpy_async(
                &mechanism,
                &window.slice.try_cast_unchecked(),
                &mut destination,
            );
        }
    }

    fn job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        read_view: TensorReader<MP::EI>,
        stage: Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
        _quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> AsyncFullCooperativeJob<MP, CM> {
        todo!()
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_coop(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullCooperativeJob<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>> {
    read_view: TensorReader<MP::EI>,
    stage: Stage<MP::ES, StridedTilingLayout>,
    mechanism: CM,
    _quantization: CubeOption<Quantization<MP>>,
}

#[cube]
impl<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>> LoadingJob<MP>
    for AsyncFullCooperativeJob<MP, CM>
{
    fn len(_this: &Self) -> u32 {
        todo!()
    }

    fn execute_task<G: GlobalConfig>(this: &mut Self, task_id: u32, #[comptime] config: G) {
        // TODO
    }
}
