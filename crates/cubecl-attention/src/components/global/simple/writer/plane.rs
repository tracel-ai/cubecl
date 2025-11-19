use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::global::{
    GlobalWriterConfig, PartitionedStage, WriteEvent, WriteEventExpand, WriteEventListener,
    plane_write,
    read::tiled::{TiledCoords, TiledLayout},
};
use cubecl_std::tensor::{View, layout::Coords2d};

use crate::components::{
    global::simple::{AttentionWriter, AttentionWriterExpand},
    stage::{AttentionPartitioner, StageAttentionConfig, plane::PlanePartitioner},
};

#[derive(CubeType)]
pub struct PlaneAttentionWriter<ES: Numeric, EO: Numeric> {
    global: View<Line<EO>, TiledCoords, ReadWrite>,
    stage: PartitionedStage<ES>,

    #[cube(comptime)]
    config: GlobalWriterConfig,
}

#[cube]
impl<ES: Numeric, EG: Numeric> PlaneAttentionWriter<ES, EG> {}

#[cube]
impl<ES: Numeric, EG: Numeric> WriteEventListener for PlaneAttentionWriter<ES, EG> {
    fn on_event(this: &mut Self, event: WriteEvent) {
        #[allow(clippy::single_match)]
        match event {
            WriteEvent::TileStored { tile } => plane_write::<ES, EG>(
                &mut this.global,
                &this.stage.unit_tile,
                tile,
                this.config.plane_dim,
                comptime!(this.config.smem_config.elements_in_tile()),
            ),
            _ => {}
        }
    }
}

#[cube]
impl<ES: Numeric, EG: Numeric> AttentionWriter<ES, EG> for PlaneAttentionWriter<ES, EG> {
    fn init<S: StageAttentionConfig>(
        global: View<Line<EG>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        // let stage_mem_config = comptime! {
        //     let elements_in_tile_row = stage_config.elements_in_partition_seq_q();
        //     let elements_in_tile_col= stage_config.elements_in_partition_val_dim();
        //     let planes = stage_config.num_planes();

        //     StageMemoryConfig {
        //         num_reading_planes: planes,
        //         elements_in_tile_row,
        //         elements_in_tile_col,
        //         // Each plane has its slot in row direction
        //         tiles_in_stage_row: planes,
        //         // Each plane needs only one slot
        //         tiles_in_stage_col: 1,
        //         line_size: 1,
        //         matrix_layout: MatrixLayout::RowMajor,
        //         swizzle: SwizzleMode::None,
        //         num_stages: 1,
        //     }
        // };
        comment!("plou");

        let stage =
            PartitionedStage::new((PlanePartitioner::seq_q_index(), 0u32), config.smem_config);

        comment!("ploucxc");
        PlaneAttentionWriter::<ES, EG> {
            global: global.view_mut(TiledLayout::new(config.smem_config)),
            stage,
            config,
        }
    }

    fn stage(&mut self) -> PartitionedStage<ES> {
        self.stage
    }
}
