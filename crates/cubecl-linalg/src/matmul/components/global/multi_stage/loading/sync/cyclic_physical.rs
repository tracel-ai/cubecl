use crate::matmul::components::global::multi_stage::double_buffering::BufferId;
use crate::matmul::components::global::tensor_view::TensorReader;
use crate::matmul::components::global::{GlobalConfig, LoadingValidation};
use crate::matmul::components::stage::{
    ContiguousTilingLayout, DualStage, DualStageExpand, DualStageFormat, RowMajorTilingOrder,
};
use crate::matmul::components::{Ident, InputIdent, InvalidConfigError};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::SyncBufferLoadingStrategy;

#[derive(CubeType, Clone, Copy)]
pub struct CyclicCoalescedPhysicalBufferLoading {}

impl LoadingValidation for CyclicCoalescedPhysicalBufferLoading {
    fn check<C: GlobalConfig>(_config: &C, _ident: Ident) -> Result<(), InvalidConfigError> {
        Ok(())
    }
}

#[cube]
impl SyncBufferLoadingStrategy for CyclicCoalescedPhysicalBufferLoading {
    // RowMajorTilingOrder hardcoded as it has no impact, because buffers are 1D
    type TilingLayout = ContiguousTilingLayout<RowMajorTilingOrder>;

    fn load_buffer<EG: Numeric, ES: Numeric, G: GlobalConfig>(
        read_view: &TensorReader<EG>,
        stage: &mut DualStage<ES, Self::TilingLayout>,
        #[comptime] buffer_id: BufferId,
        #[comptime] ident: Ident,
        #[comptime] config: G,
    ) {
        match stage {
            DualStage::Virtual(_) => {
                comptime!(panic!("This loader is for physical dual stage only"));
            }
            DualStage::Physical(stage) => {
                let tiling = config.tiling_dimensions(ident);
                let line_size = config.global_line_size(ident);
                let num_buffer_elements = match ident.as_input() {
                    InputIdent::Lhs => tiling.tile_count_row() * tiling.tile_size(),
                    InputIdent::Rhs => tiling.tile_count_col() * tiling.tile_size(),
                };
                let jump_length = comptime!(config.num_planes() * config.plane_dim() * line_size);
                let num_loads_per_unit = comptime!(num_buffer_elements / jump_length);

                let unit_id = UNIT_POS_Y * config.plane_dim() + UNIT_POS_X;
                let unit_position_base = unit_id * line_size;

                for i in 0..num_loads_per_unit {
                    let unit_position = unit_position_base + i * jump_length;

                    let tile_num_elements = tiling.tile_size();
                    let nth_tile = unit_position / tile_num_elements;
                    let pos_within_tile = unit_position % tile_num_elements;

                    let (tile_x, tile_y) = match ident.as_input() {
                        InputIdent::Lhs => (nth_tile, buffer_id.to_u32().runtime()),
                        InputIdent::Rhs => (buffer_id.to_u32().runtime(), nth_tile),
                    };

                    let line_read = read_view.load_coalesced_in_tile::<G>(
                        tile_x,
                        tile_y,
                        pos_within_tile,
                        ident,
                        config,
                    );

                    stage.as_slice_mut(buffer_id)[unit_position / line_size] =
                        Line::cast_from(line_read);
                }
            }
        };
    }

    fn dual_stage_format() -> DualStageFormat {
        DualStageFormat::new_Physical()
    }
}
