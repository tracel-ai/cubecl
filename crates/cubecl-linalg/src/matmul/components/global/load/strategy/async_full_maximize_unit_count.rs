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
/// Executes one memcpy_async call per unit.
/// The objective is to reduce branching, prioritizing this over maximizing memory slice length.
pub struct AsyncFullMaximizeUnitCountLoading {}

impl LoadingValidation for AsyncFullMaximizeUnitCountLoading {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let matrix_layout = config.matrix_layout(ident);
        let tiling_dimensions = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);

        let (num_slices, slice_length) = match matrix_layout {
            MatrixLayout::RowMajor => (
                tiling_dimensions.total_row(),
                tiling_dimensions.total_col() / line_size,
            ),
            MatrixLayout::ColMajor => (
                tiling_dimensions.total_col(),
                tiling_dimensions.total_row() / line_size,
            ),
        };
        let unit_count = config.plane_dim() * config.num_planes();

        if unit_count % num_slices != 0 {
            return Err(Box::new(
                "Number of slices must divide number of units evenly",
            ));
        }
        if slice_length % (unit_count / num_slices) != 0 {
            return Err(Box::new(
                "Number of units per slice must divide slice length evenly",
            ));
        }

        Ok(())
    }
}

#[cube]
impl AsyncFullLoadingStrategy for AsyncFullMaximizeUnitCountLoading {
    type TilingLayout = StridedTilingLayout;
    type Job<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>> =
        AsyncFullMaximizeUnitCountJob<MP, CM>;

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
        mut stage: Stage<MP::ES, Self::TilingLayout>,
        mechanism: CM,
        _quantization: CubeOption<Quantization<MP>>,
        #[comptime] input_ident: InputIdent,
        #[comptime] config: G,
    ) -> AsyncFullMaximizeUnitCountJob<MP, CM> {
        let matrix_layout = config.matrix_layout(input_ident);
        let tiling_dimensions = config.tiling_dimensions(input_ident);
        let line_size = config.global_line_size(input_ident);

        let (num_slices, slice_length) = match matrix_layout {
            MatrixLayout::RowMajor => (
                tiling_dimensions.total_row(),
                tiling_dimensions.total_col() / line_size,
            ),
            MatrixLayout::ColMajor => (
                tiling_dimensions.total_col(),
                tiling_dimensions.total_row() / line_size,
            ),
        };

        let unit_count = config.plane_dim() * config.num_planes();

        let units_per_slice = comptime!(unit_count / num_slices);
        let nth_slice = UNIT_POS / units_per_slice;

        let destination: SliceMut<Line<MP::ES>> =
            StridedTilingLayout::nth_slice::<MP::ES, G::SmmConfig>(
                &mut stage,
                nth_slice,
                input_ident.as_ident(),
                config.to_smm_config(),
            );

        let segment_length = comptime!(slice_length / units_per_slice);
        let nth_segment = UNIT_POS % units_per_slice;

        AsyncFullMaximizeUnitCountJob::<MP, CM> {
            destination,
            mechanism,
            nth_slice,
            nth_segment,
            segment_length,
            input_ident,
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}

#[derive(CubeType, Clone, Copy)]
pub struct AsyncFullMaximizeUnitCountJob<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>> {
    destination: SliceMut<Line<MP::ES>>,
    mechanism: CM,
    nth_slice: u32,
    nth_segment: u32,
    #[cube(comptime)]
    segment_length: u32,
    #[cube(comptime)]
    input_ident: InputIdent,
}

#[cube]
impl<MP: MatmulPrecision, CM: CopyMechanism<MP::ES>> LoadingJob<MP>
    for AsyncFullMaximizeUnitCountJob<MP, CM>
{
    fn len(_this: &Self) -> u32 {
        1u32
    }

    fn execute_task<G: GlobalConfig>(
        this: &mut Self,
        _task_id: u32,
        read_view: &TensorReader<MP::EI>,
        #[comptime] config: G,
    ) {
        let window: Window<MP::EI> =
            read_view.load_window_in_stage::<G>(this.nth_slice, this.input_ident, config);
        let seg_start = Min::min(this.nth_segment * this.segment_length, window.size);
        let seg_end = Min::min((this.nth_segment + 1) * this.segment_length, window.size);

        let src_segment = window.slice.slice(seg_start, seg_end);
        let mut dest_segment = this.destination.slice_mut(seg_start, seg_end);

        CM::memcpy_async(
            &this.mechanism,
            &src_segment.try_cast_unchecked(),
            &mut dest_segment,
        );
    }
}
