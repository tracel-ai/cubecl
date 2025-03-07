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
/// Executes one memcpy_async call per unit.
/// The objective is to reduce branching, prioritizing this over maximizing memory slice length.
pub struct MaximizeUnitCountLoading {}

impl LoadingValidation for MaximizeUnitCountLoading {
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
impl AsyncLoadingStrategy for MaximizeUnitCountLoading {
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

        let units_per_slice = unit_count / num_slices;
        let nth_slice = UNIT_POS / units_per_slice;

        let window: Window<EG> = read_view.load_window_in_stage::<G>(nth_slice, ident, config);
        let mut destination: SliceMut<Line<ES>> = StridedTilingLayout::nth_slice::<ES, G::SmmConfig>(
            stage,
            nth_slice,
            ident,
            config.to_smm_config(),
        );

        let segment_length = slice_length / units_per_slice;
        let nth_segment = UNIT_POS % units_per_slice;

        let seg_start = Min::min(nth_segment * segment_length, window.size);
        let seg_end = Min::min((nth_segment + 1) * segment_length, window.size);

        let src_segment = window.slice.slice(seg_start, seg_end);
        let mut dest_segment = destination.slice_mut(seg_start, seg_end);

        CM::memcpy_async(
            mechanism,
            &src_segment.try_cast_unchecked(),
            &mut dest_segment,
        );
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
        BarrierLevel::cube_manual(0u32)
    }
}
