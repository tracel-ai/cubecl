use cubecl::prelude::*;
use cubecl_core::{self as cubecl};
use cubecl_matmul::components::{
    StageIdent,
    global::{
        GlobalWriterConfig, PartitionedStage, WriteEvent, WriteEventExpand, WriteEventListener,
        read::tiled::{TiledCoords, TiledLayout},
        unit_write,
    },
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
    config: GlobalWriterConfig,
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
                comptime!(this.config.smem_config.elements_per_tile()),
            ),
            _ => {}
        }
    }
}

#[cube]
impl<ES: Numeric, EG: Numeric> AttentionWriter<ES, EG> for UnitAttentionWriter<ES, EG> {
    fn init<S: StageAttentionConfig>(
        global: View<Line<EG>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        let stage =
            PartitionedStage::new((UnitPartitioner::seq_q_index(), 0u32), config.smem_config);

        UnitAttentionWriter::<ES, EG> {
            global: global.view_mut(TiledLayout::new(StageIdent::Out, config.smem_config)),
            stage,
            config,
        }
    }

    fn stage(&mut self) -> PartitionedStage<ES> {
        self.stage
    }
}
