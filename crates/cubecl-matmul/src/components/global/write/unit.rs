use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::{View, layout::Coords2d};

use crate::components::{
    MatrixPrecision, StageIdent,
    global::{
        GlobalWriter, GlobalWriterFamily, PartitionedStage, PartitionedStageFamily, WriteEvent,
        WriteEventExpand, WriteEventListener,
        memory::GlobalMemoryConfig,
        read::tiled::{TiledCoords, TiledLayout},
    },
    stage::{StageConfig, StageMemoryConfig, StagePartitioner, UnitPartitioner},
    tile::StridedTile,
};

#[derive(CubeType)]
/// Writes tiles from out shared memory to output global memory
/// using a unit for each tile
pub struct UnitWriter<IP: MatrixPrecision> {
    global: View<Line<IP::Global>, TiledCoords, ReadWrite>,
    stage: PartitionedStage<IP::Stage>,

    #[cube(comptime)]
    config: GlobalMemoryConfig,
}

#[cube]
impl<IP: MatrixPrecision> UnitWriter<IP> {
    pub fn new<S: StageConfig>(
        global: View<Line<IP::Global>, Coords2d, ReadWrite>,
        #[comptime] global_config: GlobalMemoryConfig,
        #[comptime] stage_config: S,
    ) -> Self {
        let stage_mem_config = comptime![stage_memory_config(stage_config)];
        let stage = PartitionedStage::new(tile_pos::<S>(stage_config), stage_mem_config);

        UnitWriter::<IP> {
            global: global.view_mut(TiledLayout::new(global_config)),
            stage,
            config: global_config,
        }
    }

    fn write(&mut self, tile: Coords2d) {
        unit_write(&mut self.global, &self.stage.unit_tile, tile, self.config)
    }
}

#[cube]
pub fn unit_write<ES: Numeric, EG: Numeric>(
    global: &mut View<Line<EG>, TiledCoords, ReadWrite>,
    smem_tile: &StridedTile<ES, ReadWrite>,
    tile_pos: Coords2d,
    #[comptime] config: GlobalMemoryConfig,
) {
    let tile_size = config.elements_in_tile();
    let output_line_size = global.line_size();
    let out_smem_slice = smem_tile.slice.with_line_size(output_line_size);

    let num_lines = tile_size / output_line_size;

    for i in 0..num_lines {
        let value = out_smem_slice[i];
        global.write_checked((tile_pos, i * output_line_size), Line::cast_from(value));
    }
}

#[cube]
fn tile_pos<S: StageConfig>(#[comptime] config: S) -> (u32, u32) {
    UnitPartitioner::coordinates::<S>(config)
}

fn stage_memory_config<S: StageConfig>(config: S) -> StageMemoryConfig {
    let units = config.num_main_flow_planes() * config.plane_dim();
    let size_n = config.tiling_scheme().stage_partitions_in_stage_n();
    let base = config.stage_memory_config(StageIdent::Acc);
    StageMemoryConfig {
        tiles_in_stage_row: units / size_n,
        tiles_in_stage_col: size_n,
        ..base
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

    fn init<S: StageConfig>(
        tensor: View<Line<IP::Global>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalMemoryConfig,
        #[comptime] stage_config: S,
    ) -> Self {
        Self::new::<S>(tensor, config, stage_config)
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
