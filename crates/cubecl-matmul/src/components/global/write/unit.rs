use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::{View, layout::Coords2d};

use crate::components::{
    MatrixPrecision, StageIdent,
    global::{
        GlobalWriter, GlobalWriterConfig, GlobalWriterFamily, PartitionedStage,
        PartitionedStageFamily, WriteEvent, WriteEventExpand, WriteEventListener,
        read::tiled::{TiledCoords, TiledLayout},
    },
    stage::{StageMemoryConfig, StagePartitioner, UnitPartitioner},
    tile::StridedTile,
};

#[derive(CubeType)]
/// Writes tiles from out shared memory to output global memory
/// using a unit for each tile
pub struct UnitWriter<IP: MatrixPrecision> {
    global: View<Line<IP::Global>, TiledCoords, ReadWrite>,
    stage: PartitionedStage<IP::Stage>,

    #[cube(comptime)]
    smem_config: StageMemoryConfig,
}

#[cube]
impl<IP: MatrixPrecision> UnitWriter<IP> {
    pub fn new(
        global: View<Line<IP::Global>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        let smem_config = config.smem_config;
        let stage = PartitionedStage::new(
            UnitPartitioner::coordinates(
                config.role_rule_config,
                config.plane_dim,
                smem_config.partitions_per_stage_along_col,
            ),
            smem_config,
        );

        UnitWriter::<IP> {
            global: global.view_mut(TiledLayout::new(StageIdent::Out, smem_config)),
            stage,
            smem_config,
        }
    }

    fn write(&mut self, tile: Coords2d) {
        unit_write(
            &mut self.global,
            &self.stage.unit_tile,
            tile,
            comptime!(self.smem_config.elements_per_tile()),
        )
    }
}

#[cube]
pub fn unit_write<ES: Numeric, EG: Numeric>(
    global: &mut View<Line<EG>, TiledCoords, ReadWrite>,
    smem_tile: &StridedTile<ES, ReadWrite>,
    tile_pos: Coords2d,
    #[comptime] elements_in_tile: u32,
) {
    let output_line_size = global.line_size();
    let out_smem_stage = smem_tile.stage.with_line_size(output_line_size);

    let num_lines = elements_in_tile / output_line_size;

    for i in 0..num_lines {
        let value = out_smem_stage[smem_tile.stage_offset(i)];
        global.write_checked((tile_pos, i * output_line_size), Line::cast_from(value));
    }
}

#[cube]
impl<IP: MatrixPrecision> WriteEventListener for UnitWriter<IP> {
    fn on_event(this: &mut Self, event: super::WriteEvent) {
        #[allow(clippy::single_match)]
        match event {
            WriteEvent::TileStored { tile } => this.write(tile),
            _ => {}
        }
    }
}

#[cube]
impl<IP: MatrixPrecision> GlobalWriter<IP> for UnitWriter<IP> {
    type Stage = PartitionedStage<IP::Stage>;

    fn init(
        tensor: View<Line<IP::Global>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        Self::new(tensor, config)
    }

    fn stage(this: &Self) -> Self::Stage {
        this.stage
    }
}

pub struct UnitWriterFamily;

impl GlobalWriterFamily for UnitWriterFamily {
    type Stage = PartitionedStageFamily;
    type Writer<IP: MatrixPrecision> = UnitWriter<IP>;
}
