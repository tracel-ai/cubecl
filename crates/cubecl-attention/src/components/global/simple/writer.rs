use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::{
    MatrixLayout, MatrixPrecision,
    global::{
        PartitionedStage, WriteEvent, WriteEventExpand, WriteEventListener,
        memory::GlobalMemoryConfig,
        plane_write,
        read::tiled::{TiledCoords, TiledLayout},
    },
    stage::StageMemoryConfig,
};
use cubecl_std::tensor::{View, layout::Coords2d};

use crate::components::stage::StageAttentionConfig;

#[derive(CubeType)]
pub struct AttentionWriter<IP: MatrixPrecision> {
    global: View<Line<IP::Global>, TiledCoords, ReadWrite>,
    stage: PartitionedStage<IP::Stage>,
    #[cube(comptime)]
    plane_dim: u32,
    #[cube(comptime)]
    config: GlobalMemoryConfig,
}

#[cube]
impl<IP: MatrixPrecision> AttentionWriter<IP> {
    pub fn new<S: StageAttentionConfig>(
        global: View<Line<IP::Global>, Coords2d, ReadWrite>,
        #[comptime] global_config: GlobalMemoryConfig,
        #[comptime] stage_config: S,
    ) -> Self {
        let stage_mem_config = comptime! {
            let tile_rows = stage_config.tiling_scheme().elements_in_partition_val_dim();
            let tile_cols = stage_config.tiling_scheme().elements_in_partition_seq_q();
            let planes = stage_config.num_planes();

            StageMemoryConfig {
                num_main_flow_planes: planes,
                elements_in_tile_row: tile_rows,
                elements_in_tile_col: tile_cols,
                tiles_in_stage_row: 1,
                tiles_in_stage_col: planes,
                stage_line_size: 1,
                matrix_layout: MatrixLayout::RowMajor,
                num_stages: 1,
            }
        };

        let stage = PartitionedStage::new((0u32, UNIT_POS_Y), stage_mem_config);

        AttentionWriter::<IP> {
            global: global.view_mut(TiledLayout::new(global_config)),
            stage,
            plane_dim: stage_config.plane_dim(),
            config: global_config,
        }
    }

    fn write(&mut self, tile_pos: Coords2d) {
        plane_write::<IP::Stage, IP::Global>(
            &mut self.global,
            &self.stage.unit_tile,
            tile_pos,
            comptime![self.plane_dim],
            comptime![self.config],
        )
    }

    pub fn stage(&mut self) -> PartitionedStage<IP::Stage> {
        self.stage
    }
}

#[cube]
impl<IP: MatrixPrecision> WriteEventListener for AttentionWriter<IP> {
    fn on_event(this: &mut Self, event: WriteEvent) {
        #[allow(clippy::single_match)]
        match event {
            WriteEvent::TileStored { tile } => {
                this.write(tile);
            }
            _ => {}
        }
    }
}
