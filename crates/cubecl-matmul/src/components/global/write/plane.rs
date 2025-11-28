use crate::components::{
    MatrixPrecision, StageIdent,
    global::{
        GlobalWriter, GlobalWriterConfig, GlobalWriterFamily, PartitionedStage,
        PartitionedStageFamily, WriteEvent, WriteEventExpand, WriteEventListener,
        read::tiled::{TiledCoords, TiledLayout},
    },
    stage::{PlanePartitioner, StageMemoryConfig, StagePartitioner},
    tile::StridedTile,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::View;
use cubecl_std::tensor::layout::Coords2d;

#[derive(CubeType)]
/// Writes tiles from out shared memory to output global memory
/// using a plane for each tile
pub struct PlaneWriter<IP: MatrixPrecision> {
    global: View<Line<IP::Global>, TiledCoords, ReadWrite>,
    stage: PartitionedStage<IP::Stage>,

    #[cube(comptime)]
    plane_dim: u32,
    #[cube(comptime)]
    smem_config: StageMemoryConfig,
}

#[cube]
impl<IP: MatrixPrecision> PlaneWriter<IP> {
    pub fn new(
        global: View<Line<IP::Global>, Coords2d, ReadWrite>,
        #[comptime] config: GlobalWriterConfig,
    ) -> Self {
        let stage = PartitionedStage::new(
            PlanePartitioner::coordinates(
                config.role_rule_config,
                config.plane_dim,
                config.smem_config.partitions_per_stage_along_col,
            ),
            config.smem_config,
        );

        PlaneWriter::<IP> {
            global: global.view_mut(TiledLayout::new(StageIdent::Out, config.smem_config)),
            stage,
            plane_dim: config.plane_dim,
            smem_config: config.smem_config,
        }
    }

    fn write(&mut self, tile_pos: Coords2d) {
        plane_write::<IP::Stage, IP::Global>(
            &mut self.global,
            &self.stage.unit_tile,
            tile_pos,
            comptime!(self.plane_dim),
            comptime!(self.smem_config.elements_per_tile()),
        )
    }
}

#[cube]
impl<IP: MatrixPrecision> WriteEventListener for PlaneWriter<IP> {
    fn on_event(this: &mut Self, event: super::WriteEvent) {
        #[allow(clippy::single_match)]
        match event {
            WriteEvent::TileStored { tile } => {
                this.write(tile);
            }
            _ => {}
        }
    }
}

#[cube]
impl<IP: MatrixPrecision> GlobalWriter<IP> for PlaneWriter<IP> {
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

#[cube]
pub fn plane_write<ES: Numeric, EG: Numeric>(
    global: &mut View<Line<EG>, TiledCoords, ReadWrite>,
    smem_tile: &StridedTile<ES, ReadWrite>,
    tile_pos: Coords2d,
    #[comptime] plane_dim: u32,
    #[comptime] elements_in_tile: u32,
) {
    let output_line_size = global.line_size();

    let unit_step = comptime![plane_dim * output_line_size];
    let num_unit_writes = comptime!(elements_in_tile.div_ceil(unit_step));
    let balanced_workload = comptime!(elements_in_tile.is_multiple_of(unit_step));

    #[unroll(num_unit_writes == 1)]
    for i in 0..num_unit_writes {
        let unit_write = UNIT_POS_X * output_line_size + i * unit_step;

        #[allow(clippy::collapsible_else_if)]
        if comptime!(balanced_workload) {
            write_line(global, smem_tile, unit_write, tile_pos);
        } else {
            if unit_write < elements_in_tile {
                write_line(global, smem_tile, unit_write, tile_pos);
            }
        }
    }
}

#[cube]
fn write_line<ES: Numeric, EG: Numeric>(
    view: &mut View<Line<EG>, TiledCoords, ReadWrite>,
    out_smem_tile: &StridedTile<ES, ReadWrite>,
    unit_write: u32,
    tile: Coords2d,
) {
    let output_line_size = view.line_size();
    let out_smem_line_size = out_smem_tile.stage.line_size();

    let value = if comptime!(output_line_size == out_smem_line_size) {
        out_smem_tile.stage[out_smem_tile.stage_offset(unit_write / output_line_size)]
    } else if comptime!(
        out_smem_line_size < output_line_size
            && output_line_size.is_multiple_of(out_smem_line_size)
    ) {
        let mut value = Line::empty(output_line_size);
        #[unroll]
        for i in 0..comptime!(output_line_size / out_smem_line_size) {
            let offs = out_smem_tile.stage_offset(unit_write + i);
            #[unroll]
            for j in 0..out_smem_line_size {
                value[i * out_smem_line_size + j] = out_smem_tile.stage[offs][j];
            }
        }
        value
    } else {
        unimplemented!()
    };

    view.write_checked((tile, unit_write), Line::cast_from(value));
}

pub struct PlaneWriterFamily;

impl GlobalWriterFamily for PlaneWriterFamily {
    type Stage = PartitionedStageFamily;
    type Writer<IP: MatrixPrecision> = PlaneWriter<IP>;
}
