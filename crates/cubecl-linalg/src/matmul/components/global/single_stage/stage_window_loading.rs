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
/// **Duplicated loading:**  
/// Each unit independently calls `memcpy_async` with the same arguments.
pub struct WindowDuplicatedLoading {}

#[derive(CubeType, Clone, Copy)]
/// Loads global memory into the stage without modification,  
/// dividing the stage into the smallest possible contiguous slices.  
///
/// **Elected loading:**  
/// A single designated unit performs all `memcpy_async` calls,  
/// while the others remain idle.
pub struct WindowElectedLoading {}

#[derive(CubeType, Clone, Copy)]
/// Loads global memory into the stage without modification,  
/// dividing the stage into the smallest possible contiguous slices.  
///
/// **Elected-only loading:**  
/// A single designated unit performs all `memcpy_async` calls,  
/// while the others remain idle. This will also do all the overhead code
/// with the elected unit only
pub struct WindowElectedOnlyLoading {}

#[derive(CubeType, Clone, Copy)]
/// Loads global memory into the stage without modification,  
/// dividing the stage into the smallest possible contiguous slices.  
///
/// **Split unit loading:**  
/// `memcpy_async` calls are distributed across multiple units,  
/// with each unit handling one call at a time until all calls are completed.  
/// Remaining units stay idle once no more work is left.
pub struct WindowSplitUnitLoading {}

#[derive(CubeType, Clone, Copy)]
/// Loads global memory into the stage without modification,  
/// dividing the stage into the smallest possible contiguous slices.  
///
/// **Split plane loading:**  
/// `memcpy_async` calls are distributed across all warps.  
/// Each plane assigns one call per unit until all calls are completed.  
/// Remaining units stay idle once no more work is left.
pub struct WindowSplitPlaneLoading {}

impl LoadingValidation for WindowDuplicatedLoading {
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

impl LoadingValidation for WindowElectedLoading {
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

impl LoadingValidation for WindowElectedOnlyLoading {
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

impl LoadingValidation for WindowSplitUnitLoading {
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

impl LoadingValidation for WindowSplitPlaneLoading {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let unit_count = config.num_planes() * config.plane_dim();
        let slice_count = match config.matrix_layout(ident) {
            MatrixLayout::RowMajor => config.tiling_dimensions(ident).total_row(),
            MatrixLayout::ColMajor => config.tiling_dimensions(ident).total_col(),
        };

        if unit_count < slice_count {
            return Err(Box::new(
                "WindowSplitPlaneLoading does not support multiple slices per unit for now.",
            ));
        }

        if slice_count < config.num_planes() {
            return Err(Box::new(
                "WindowSplitPlaneLoading needs every plane to perform at least once for now.",
            ));
        }

        Ok(())
    }
}

#[cube]
impl AsyncLoadingStrategy for WindowDuplicatedLoading {
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

#[cube]
impl AsyncLoadingStrategy for WindowElectedLoading {
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

            if UNIT_POS == 0 {
                CM::memcpy_async(
                    &mechanism,
                    &window.slice.try_cast_unchecked(),
                    &mut destination,
                );
            }

            // If padding needed: TODO comptime conditional
            for i in window.size..expected_window_size {
                destination[i] = Line::cast_from(0);
            }
        }
    }
}

#[cube]
impl AsyncLoadingStrategy for WindowElectedOnlyLoading {
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

        if UNIT_POS == 0 {
            for nth_slice in 0..num_slices {
                let window: Window<EG> =
                    read_view.load_window_no_tile::<G>(nth_slice, ident, config);
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
}

#[cube]
impl AsyncLoadingStrategy for WindowSplitUnitLoading {
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

        let plane_dim = config.plane_dim();
        let plane_count = config.num_planes();
        let unit_count = plane_dim * plane_count;
        let unit_index = UNIT_POS;

        let slices_per_unit = (num_slices + unit_count - 1) / unit_count;

        #[unroll(slices_per_unit==1)]
        for nth_slice_local in 0..slices_per_unit {
            let nth_slice = unit_count * nth_slice_local + unit_index;

            // TODO no if when always fits
            if nth_slice < num_slices {
                let window: Window<EG> =
                    read_view.load_window_no_tile::<G>(nth_slice, ident, config);
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
}

#[cube]
impl AsyncLoadingStrategy for WindowSplitPlaneLoading {
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

        let plane_count = config.num_planes();
        let plane_index = UNIT_POS_Y;

        let slices_per_plane = (num_slices + plane_count - 1) / plane_count;
        // Equals number of used units

        // If we're one of the units that need to perform
        if UNIT_POS_X < slices_per_plane {
            // Assuming there never is more than 1 slice to do per unit
            let nth_slice = plane_index * slices_per_plane + UNIT_POS_X;

            // TODO no if when always fits
            if nth_slice < num_slices {
                let window: Window<EG> =
                    read_view.load_window_no_tile::<G>(nth_slice, ident, config);
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
}
