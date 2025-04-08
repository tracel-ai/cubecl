use crate::matmul::components::{
    Ident, InputIdent, InvalidConfigError, MatmulPrecision, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig, LoadingValidation, Quantization,
        load::{AsyncFullLoadingStrategy, default_async_full_load},
        tensor_view::{TensorReader, Window},
    },
    stage::{Stage, StridedTilingLayout},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};
use cubecl_std::CubeOption;

use super::LoadingJob;

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
    type Job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>> =
        AsyncFullMaximizeSliceLengthJob<MP, CM>;

    fn load_full<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        read_view: &TensorReader<MP::EI>,
        stage: Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
        quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) {
        default_async_full_load::<Self, MP, G, CM>(
            read_view,
            stage,
            mechanism,
            quantization,
            input_ident,
            config,
        )
    }

    fn job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>, G: GlobalConfig>(
        stage: Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
        _quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> AsyncFullMaximizeSliceLengthJob<MP, CM> {
        let matrix_layout = config.matrix_layout(input_ident);
        let tiling_dimensions = config.tiling_dimensions(input_ident);

        let num_slices = match matrix_layout {
            MatrixLayout::RowMajor => tiling_dimensions.total_row(),
            MatrixLayout::ColMajor => tiling_dimensions.total_col(),
        };
        let unit_count = config.plane_dim() * config.num_planes();

        let num_tasks = comptime!((num_slices + unit_count - 1) / unit_count);

        AsyncFullMaximizeSliceLengthJob::<MP, CM> {
            stage,
            mechanism,
            num_tasks,
            unit_count,
            num_slices,
            input_ident,
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullMaximizeSliceLengthJob<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>> {
    stage: Stage<MP::ES, StridedTilingLayout>,
    mechanism: CM,

    #[cube(comptime)]
    num_tasks: u32,
    #[cube(comptime)]
    unit_count: u32,
    #[cube(comptime)]
    num_slices: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
}

#[cube]
impl<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>> LoadingJob<MP>
    for AsyncFullMaximizeSliceLengthJob<MP, CM>
{
    fn len(this: &Self) -> u32 {
        this.num_tasks.runtime()
    }

    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        task_id: u32,
        read_view: &TensorReader<MP::EI>,
        #[comptime] config: G,
    ) {
        let nth_slice = this.unit_count * task_id + UNIT_POS;

        #[allow(clippy::collapsible_else_if)]
        if comptime!(this.num_slices % this.unit_count == 0) {
            load_nth_slice::<MP::EI, MP::ES, CM, G>(
                nth_slice,
                read_view,
                &mut this.stage,
                &this.mechanism,
                this.input_ident,
                config,
            );
        } else {
            if nth_slice < this.num_slices {
                load_nth_slice::<MP::EI, MP::ES, CM, G>(
                    nth_slice,
                    read_view,
                    &mut this.stage,
                    &this.mechanism,
                    this.input_ident,
                    config,
                );
            }
        };
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
