use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube]
pub trait PlaneMapper {
    fn tile_index(row_offset: u32, col_offset: u32) -> u32;
    // let compute_plane_id = 0; // TMP

    // let num_planes_per_row = (definition.n / 16) / acc.len();
    // let tile_row = compute_plane_id / num_planes_per_row;
    // let tile_col_base = (compute_plane_id % num_planes_per_row) * acc.len();
}

// let tile_index = SL::tile_index(0, buffer_iter);
// let lhs_slice_length = I::M * I::K;
// let start = tile_index * lhs_slice_length;
// let end = start + lhs_slice_length;
// let tile_index = SR::tile_index(buffer_iter, accumulator_iter);
// let rhs_slice_length = I::K * I::N;
// let start = tile_index * rhs_slice_length;
// let end = start + rhs_slice_length;

// let tile_index = SO::tile_index(0, accumulator_iter);
// let out_slice_length = I::M * I::N;
// let start = tile_index * out_slice_length;
// let end = start + out_slice_length;

// fn acc_read(acc: &Self::Accumulator, out: &mut Out, #[comptime] _config: &Self::Config) {
//     let plane_id = UNIT_POS_Y;
//     let num_planes = 8u32;
//     let line_size = 4u32;

//     let tile_size = Lhs::TILE_SIZE_X * Rhs::TILE_SIZE_Y;
//     let mut output_smem = SharedMemory::<Elem>::new_lined(num_planes * tile_size, line_size);

//     let plane_offset = plane_id * tile_size;

//     for accumulator_iter in 0..acc.len() {
//         let accumulator = acc.index(accumulator_iter);
//         let slice = output_smem.slice_mut(plane_offset, plane_offset + tile_size);

//         Instr::read_output(accumulator, slice);

//         Out::write(out, slice.as_slice(), 0u32, accumulator_iter);
//     }
// }
