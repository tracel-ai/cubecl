use crate::components::{
    MatrixPrecision, StageIdent,
    global::{
        GlobalWriter, GlobalWriterFamily, PartitionedStage, PartitionedStageFamily, WriteEvent,
        WriteEventExpand, WriteEventListener,
        memory::GlobalMemoryConfig,
        read::tiled::{TiledCoords, TiledLayout},
    },
    stage::{PlanePartitioner, StageConfig, StageMemoryConfig, StagePartitioner},
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
    config: GlobalMemoryConfig,
}

#[cube]
impl<IP: MatrixPrecision> PlaneWriter<IP> {
    pub fn new<S: StageConfig>(
        global: View<Line<IP::Global>, Coords2d, ReadWrite>,
        #[comptime] global_config: GlobalMemoryConfig,
        #[comptime] stage_config: S,
    ) -> Self {
        let stage_mem_config = comptime![stage_memory_config(stage_config)];
        let stage = PartitionedStage::new(tile_pos::<S>(stage_config), stage_mem_config);

        PlaneWriter::<IP> {
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
}

#[cube]
fn tile_pos<S: StageConfig>(#[comptime] config: S) -> (u32, u32) {
    PlanePartitioner::coordinates::<S>(config)
}

fn stage_memory_config<S: StageConfig>(config: S) -> StageMemoryConfig {
    let planes = config.num_main_flow_planes();
    let size_n = config.tiling_scheme().stage_partitions_in_stage_n();
    let base = config.stage_memory_config(StageIdent::Acc);
    StageMemoryConfig {
        tiles_in_stage_row: planes / size_n,
        tiles_in_stage_col: size_n,
        ..base
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

#[cube]
pub fn plane_write<ES: Numeric, EG: Numeric>(
    global: &mut View<Line<EG>, TiledCoords, ReadWrite>,
    smem_tile: &StridedTile<ES, ReadWrite>,
    tile_pos: Coords2d,
    #[comptime] plane_dim: u32,
    #[comptime] config: GlobalMemoryConfig,
) {
    let tile_size = config.elements_in_tile();
    let output_line_size = global.line_size();

    let unit_step = comptime![plane_dim * output_line_size];
    let num_unit_writes = comptime!(tile_size.div_ceil(unit_step));
    let balanced_workload = comptime!(tile_size.is_multiple_of(unit_step));

    #[unroll(num_unit_writes == 1)]
    for i in 0..num_unit_writes {
        let unit_write = UNIT_POS_X * output_line_size + i * unit_step;

        #[allow(clippy::collapsible_else_if)]
        if comptime!(balanced_workload) {
            write_line(global, &smem_tile.slice, unit_write, tile_pos);
        } else {
            if unit_write < tile_size {
                write_line(global, &smem_tile.slice, unit_write, tile_pos);
            }
        }
    }
}

#[cube]
fn write_line<ES: Numeric, EG: Numeric>(
    view: &mut View<Line<EG>, TiledCoords, ReadWrite>,
    out_smem_slice: &Slice<Line<ES>, ReadWrite>,
    unit_write: u32,
    tile: Coords2d,
) {
    let output_line_size = view.line_size();
    let out_smem_line_size = out_smem_slice.line_size();

    let value = if comptime!(output_line_size == out_smem_line_size) {
        out_smem_slice[unit_write / output_line_size]
    } else if comptime!(
        out_smem_line_size < output_line_size
            && output_line_size.is_multiple_of(out_smem_line_size)
    ) {
        let mut value = Line::empty(output_line_size);
        #[unroll]
        for i in 0..comptime!(output_line_size / out_smem_line_size) {
            #[unroll]
            for j in 0..out_smem_line_size {
                value[i * out_smem_line_size + j] = out_smem_slice[unit_write + i][j];
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
