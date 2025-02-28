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
/// Executes one memcpy_async call per unit.
/// The objective is to reduce branching, prioritizing this over maximizing memory slice length.
pub struct MaximizeUnitCountLoading {}

impl LoadingValidation for MaximizeUnitCountLoading {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        if config.check_row_bounds(ident) || config.check_col_bounds(ident) {
            return Err(Box::new(
                "Check bounds are not yet supported on window loading.",
            ));
        }

        if config.transpose_load(ident) {
            return Err(Box::new(
                "Transpose load is not supported with window loading.",
            ));
        }

        Ok(())
    }
}

#[cube]
impl AsyncLoadingStrategy for MaximizeUnitCountLoading {
    type TilingLayout = StridedTilingLayout;

    fn load<EG: Numeric, ES: Numeric, G: GlobalConfig, CM: CopyMechanism<ES>>(
        read_view: &TensorReader<EG>,
        stage_slice: &mut SliceMut<Line<ES>>,
        mechanism: &CM,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        // TODO
        //
        // Find num_slices
        // Divide unit_count per num_slices
        // -> if num_slices > unit_count, abort
        // -> if not exactly divisible, abort
        // Gives how many units_per_slice
        // Divide slice_length per units_per_slice
        // -> if not exactly divisible, abort
        // No for loop.
        // Will need a new load_window and layout::slice because it's not exactly nth_slice
        // Then perform memcpy
    }

    fn barrier_level() -> BarrierLevel {
        BarrierLevel::cube_manual(0u32)
    }
}
