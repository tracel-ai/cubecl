use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation};
use crate::matmul::components::stage::{Stage, StridedTilingLayout};
use crate::matmul::components::{Ident, InvalidConfigError};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::SyncLoadingStrategy;

#[derive(CubeType, Clone, Copy)]
/// Loads the content of all the tensor view using all planes,
/// iterating with steps determined by the plane's dimension.
pub struct StridedCoalescedLoading {}

impl LoadingValidation for StridedCoalescedLoading {
    fn check<C: GlobalConfig>(config: &C, ident: Ident) -> Result<(), InvalidConfigError> {
        let tiling = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);

        let num_stage_lines = tiling.total_size() / line_size;
        let total_units = config.num_planes() * config.plane_dim();

        if num_stage_lines % total_units != 0 {
            return Err(Box::new(
                "Too many data will be loaded, resulting in out of bounds.
        Try setting line size and number of planes so that total unit count {:?} divides number of lines in stage.",
            ));
        }

        Ok(())
    }
}

#[cube]
impl SyncLoadingStrategy for StridedCoalescedLoading {
    type TilingLayout = StridedTilingLayout;

    fn load_full<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        read_view: &TensorReader<EG>,
        stage: &mut Stage<ES, Self::TilingLayout>,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        let tiling = config.tiling_dimensions(ident);
        let line_size = config.global_line_size(ident);
        let num_stage_lines = tiling.total_size() / line_size;
        let unit_count = config.num_planes() * config.plane_dim();
        let num_loads_per_unit = comptime!(num_stage_lines / unit_count);

        let unit_base_position = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;

        for i in 0..num_loads_per_unit {
            let unit_position = unit_base_position + i * unit_count;

            let line_read =
                read_view.load_coalesced_in_stage::<G>(unit_position * line_size, ident, config);

            stage.as_slice_mut()[unit_position] = Line::cast_from(line_read);
        }
    }

    fn load_buffer<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        _read_view: &TensorReader<EG>,
        _stage: &mut Stage<ES, Self::TilingLayout>,
        _buffer_index: u32,
        #[comptime] _ident: Ident,
        #[comptime] _config: G,
    ) {
        // TODO
    }
}
