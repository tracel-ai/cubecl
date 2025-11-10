use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::{
    MatrixLayout,
    global::{
        PartitionedStage, WriteEvent, WriteEventExpand, WriteEventListener,
        memory::GlobalMemoryConfig,
        read::tiled::{TiledCoords, TiledLayout},
        unit_write,
    },
    stage::StageMemoryConfig,
};
use cubecl_std::tensor::{View, layout::Coords2d};

use crate::components::{
    global::simple::{AttentionWriter, AttentionWriterExpand},
    stage::{AttentionPartitioner, StageAttentionConfig, unit::UnitPartitioner},
};

#[derive(CubeType)]
pub struct UnitAttentionWriter<ES: Numeric, EG: Numeric> {
    global: View<Line<EG>, TiledCoords, ReadWrite>,
    stage: PartitionedStage<ES>,

    #[cube(comptime)]
    config: GlobalMemoryConfig,
}

#[cube]
impl<ES: Numeric, EG: Numeric> WriteEventListener for UnitAttentionWriter<ES, EG> {
    fn on_event(this: &mut Self, event: WriteEvent) {
        #[allow(clippy::single_match)]
        match event {
            WriteEvent::TileStored { tile } => unit_write::<ES, EG>(
                &mut this.global,
                &this.stage.unit_tile,
                tile,
                comptime![this.config],
            ),
            _ => {}
        }
    }
}

#[cube]
impl<ES: Numeric, EG: Numeric> AttentionWriter<ES, EG> for UnitAttentionWriter<ES, EG> {
    fn new<S: StageAttentionConfig>(
        global: View<Line<EG>, Coords2d, ReadWrite>,
        #[comptime] global_config: GlobalMemoryConfig,
        #[comptime] stage_config: S,
    ) -> Self {
        let stage_mem_config = comptime! {
            let elements_in_tile_row = stage_config.tiling_scheme().elements_in_partition_seq_q();
            let elements_in_tile_col = stage_config.tiling_scheme().elements_in_partition_val_dim();
            let planes = stage_config.num_planes();

            StageMemoryConfig {
                num_main_flow_planes: planes,
                elements_in_tile_row,
                elements_in_tile_col,
                // Each unit has its slot in row direction
                tiles_in_stage_row: planes,
                // Each unit needs only one slot
                tiles_in_stage_col: 1,
                stage_line_size: 1,
                matrix_layout: MatrixLayout::RowMajor,
                num_stages: 1,
            }
        };

        let stage = PartitionedStage::new((UnitPartitioner::seq_q_index(), 0u32), stage_mem_config);

        UnitAttentionWriter::<ES, EG> {
            global: global.view_mut(TiledLayout::new(global_config)),
            stage,
            config: global_config,
        }
    }

    fn stage(&mut self) -> PartitionedStage<ES> {
        self.stage
    }
}
