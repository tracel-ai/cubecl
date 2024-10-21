use super::TensorView;
use crate::matmul::config::PlaneMapper;
use crate::matmul::matmul_global::GmmConfig;
use crate::matmul::matrix::Ident;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[derive(CubeType)]
pub struct SimpleSmemUnloader {}

#[cube]
impl PlaneMapper for SimpleSmemUnloader {
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }
}

#[cube]
pub(crate) fn unload_from_slice<EG: Numeric, ES: Numeric, G: GmmConfig>(
    view: &mut TensorView<EG>,
    slice: &Slice<'_, Line<ES>>,
    row_tile_begin: u32,
    col_tile_begin: u32,
    #[comptime] config: G
) {
    let stage_dim = config.stage_dim(Ident::Out);
    let slice_line_size = config.out_smem_line_size();
    let out_line_size = config.line_size(Ident::Out);

    let unit_jump = config.plane_dim() * view.tensor.line_size();
    let num_unit_writes = stage_dim.tile_num_elements() / unit_jump;

    let _ = comptime!(check_line_size(out_line_size, slice_line_size));

    for i in 0..num_unit_writes {
        let unit_write = SimpleSmemUnloader::plane_unit() * out_line_size + i * unit_jump;

        let row = row_tile_begin + unit_write / stage_dim.tile_size_y;
        let col = col_tile_begin + unit_write % stage_dim.tile_size_y;

        let value = slice[unit_write / out_line_size];
        write_coalesced::<EG, ES>(view, row, col, value);
    }
}


#[cube]
fn write_coalesced<EG: Numeric, ES: Numeric>(
    view: &mut TensorView<EG>,
    write_x: u32,
    write_y: u32,
    value: Line<ES>,
) {
    let tensor = &mut view.tensor;
    let view_x = write_x + view.x_offset;
    let view_y = write_y + view.y_offset;

    let write_position = (view_x * view.stride_x + view_y * view.stride_y) / tensor.line_size();

    // TODO: will need comptime checkbound condition because we can't use select for not writing
    if write_x < view.shape_x && write_y < view.shape_y {
        tensor[write_position] = Line::cast_from(value);
    }
}


fn check_line_size(out_line_size: u32, slice_line_size: u32) {
    assert_eq!(out_line_size, slice_line_size, 
        "Error: Expected global output and output shared memory to have equal line size, but found out_line_size = {} and slice_line_size = {}.",
        out_line_size, slice_line_size
    );
}
