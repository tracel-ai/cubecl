use crate::matmul::components::{
    Ident, InputIdent, InvalidConfigError, MatrixLayout,
    global::{
        CopyMechanism, GlobalConfig, LoadingValidation,
        single_stage::AsyncBufferLoadingStrategy,
        tensor_view::{TensorReader, Window},
    },
    stage::{Stage, StridedTilingLayout},
};
use cubecl_core::prelude::*;
use cubecl_core::{self as cubecl, prelude::barrier::BarrierLevel};

#[derive(CubeType, Clone, Copy)]
/// Executes one memcpy_async call per contiguous slice.
/// The goal is to reduce the total number of memcpy_async calls, though it may result in idle threads.
pub struct MaximizeSliceLengthBufferLoading {}

impl LoadingValidation for MaximizeSliceLengthBufferLoading {
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[cube]
impl AsyncBufferLoadingStrategy for MaximizeSliceLengthBufferLoading {
    type TilingLayout = StridedTilingLayout;

    fn load_buffer<EG: Numeric, ES: Numeric, G: GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        mechanism: &CM,
        #[comptime] buffer_index: u32,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let matrix_layout = config.matrix_layout(ident);
        let tiling_dimensions = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);
        let num_buffers = 2;

        // If buffer is parallel to slices, slices are as long as in full stage, but there are less.
        // Otherwise, slices are shorter but there are as many as in full stage
        let (num_slices, num_slices_buffer_offset, slice_length, slice_buffer_offset) = comptime! {
            match (ident.as_input(), matrix_layout) {
                (InputIdent::Lhs, MatrixLayout::RowMajor) => {
                    let num_slices = tiling_dimensions.total_row();
                    let num_slices_buffer_offset = 0;
                    let slice_length = tiling_dimensions.total_col() / (num_buffers * line_size);
                    let slice_buffer_offset = buffer_index * slice_length;

                    (num_slices, num_slices_buffer_offset, slice_length, slice_buffer_offset)
                },
                (InputIdent::Lhs, MatrixLayout::ColMajor) => {
                    let num_slices = tiling_dimensions.total_col() / num_buffers;
                    let num_slices_buffer_offset = buffer_index * num_slices;
                    let slice_length = tiling_dimensions.total_row() / line_size;
                    let slice_buffer_offset = 0;

                    (num_slices, num_slices_buffer_offset, slice_length, slice_buffer_offset)
                },
                (InputIdent::Rhs, MatrixLayout::RowMajor) => {
                    let num_slices = tiling_dimensions.total_row() / num_buffers;
                    let num_slices_buffer_offset = buffer_index * num_slices;
                    let slice_length = tiling_dimensions.total_col() / line_size;
                    let slice_buffer_offset = 0;

                    (num_slices, num_slices_buffer_offset, slice_length, slice_buffer_offset)
                },
                (InputIdent::Rhs, MatrixLayout::ColMajor) => {
                    let num_slices = tiling_dimensions.total_col();
                    let num_slices_buffer_offset = 0;
                    let slice_length = tiling_dimensions.total_row() / (num_buffers * line_size);
                    let slice_buffer_offset = buffer_index * slice_length;

                    (num_slices, num_slices_buffer_offset, slice_length, slice_buffer_offset)
                },
            }
        };

        let unit_count = config.plane_dim() * config.num_planes();
        let slices_per_unit = (num_slices + unit_count - 1) / unit_count;

        // Typically there will be only 1 slice per unit
        #[unroll(slices_per_unit==1)]
        for nth_slice_local in 0..slices_per_unit {
            let nth_slice_in_buffer = unit_count * nth_slice_local + UNIT_POS;

            let nth_slice = nth_slice_in_buffer + num_slices_buffer_offset;

            let window: Window<EG> = read_view.load_window_in_stage::<G>(nth_slice, ident, config);
            let mut destination: SliceMut<Line<ES>> =
                StridedTilingLayout::nth_slice::<ES, G::SmmConfig>(
                    stage,
                    nth_slice,
                    ident,
                    config.to_smm_config(),
                );

            let start = slice_buffer_offset;
            let limit = select(
                slice_buffer_offset < window.size,
                slice_buffer_offset,
                window.size,
            );
            let end = start + Min::min(window.size - limit, slice_length);

            let src = window.slice.slice(start, end);
            let mut dest = destination.slice_mut(start, end);

            #[allow(clippy::collapsible_else_if)]
            if comptime!(num_slices % unit_count == 0) {
                CM::memcpy_async(mechanism, &src.try_cast_unchecked(), &mut dest);
            } else {
                if nth_slice_in_buffer < num_slices {
                    CM::memcpy_async(mechanism, &src.try_cast_unchecked(), &mut dest);
                }
            };
        }
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}
