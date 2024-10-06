use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfo;
use crate::matmul::id_map::PlaneMapper;

// #[cube]
// fn write_to_output<F: Float>(
//     out: &mut Tensor<F>,
//     accumulators: Sequence<cmma::Matrix<F>>,
// ) {
//     let num_accumulators = comptime_info.num_accumulators;
//     let num_compute_planes = comptime_info.num_compute_planes;
//     let plane_id = runtime_info.compute_ids.plane;

//     let smem_stride = comptime_info.tile_size_m * comptime_info.tile_size_n;
//     let smem_size = num_compute_planes * smem_stride;

//     let acc_sm = SharedMemory::<F>::new(smem_size);

//     let slice_offset = plane_id * smem_stride;
//     let slice = acc_sm.slice_mut_unsafe(slice_offset, slice_offset + smem_stride);

//     #[unroll]
//     for n in 0..num_accumulators {
//         cmma::store(
//             slice,
//             accumulators.index(n),
//             comptime_info.tile_size_n,
//             cmma::MatrixLayout::RowMajor,
//         );

//         shared_memory_to_output(out, plane_id, acc_sm, n, runtime_info, comptime_info);
//     }
// }

// #[cube]
// pub(crate) fn shared_memory_to_output<F: Float>(
//     out: &mut Tensor<F>,
//     smem_position: u32,
//     accumulator_sm: SharedMemory<F>,
//     n_iter: u32,
//     runtime_info: RuntimeCmmaInfo,
//     #[comptime] comptime_info: ComptimeCmmaInfo,
// ) {
//     let check_m_bounds = comptime_info.check_m_bounds;
//     let check_n_bounds = comptime_info.check_n_bounds;

//     if check_m_bounds {
//         if check_n_bounds {
//             write_tile::<F, WholeCheckBlockIO>(
//                 out,
//                 smem_position,
//                 accumulator_sm,
//                 n_iter,
//                 runtime_info,
//                 comptime_info,
//             );
//         } else {
//             write_tile::<F, VerticalCheckBlockIO>(
//                 out,
//                 smem_position,
//                 accumulator_sm,
//                 n_iter,
//                 runtime_info,
//                 comptime_info,
//             );
//         }
//     } else if check_n_bounds {
//         write_tile::<F, HorizontalCheckBlockIO>(
//             out,
//             smem_position,
//             accumulator_sm,
//             n_iter,
//             runtime_info,
//             comptime_info,
//         );
//     } else {
//         write_tile::<F, UncheckedBlockIO>(
//             out,
//             smem_position,
//             accumulator_sm,
//             n_iter,
//             runtime_info,
//             comptime_info,
//         );
//     }
// }

#[cube]
pub trait Smem2Tensor {
    fn smem_to_tensor<E: Numeric, C: CubePrimitive>(
        out: &mut Tensor<Line<E>>,
        smem_slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        cube_offsets: (u32, u32),
        #[comptime] block_info: BlockInfo,
    );
}

#[derive(CubeType)]
pub struct Smem2TensorSimple {}

#[cube]
impl PlaneMapper for Smem2TensorSimple {
    fn plane_id() -> u32 {
        UNIT_POS_Y
    }

    fn plane_unit() -> u32 {
        UNIT_POS_X
    }

    fn num_planes() -> u32 {
        CUBE_DIM_Y
    }

    fn plane_dim() -> u32 {
        CUBE_DIM_X
    }
}

#[cube]
impl Smem2Tensor for Smem2TensorSimple {
    fn smem_to_tensor<E: Numeric, C: CubePrimitive>(
        gmem: &mut Tensor<Line<E>>,
        smem_slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
        cube_offsets: (u32, u32),
        #[comptime] block_info: BlockInfo,
    ) {
        let row_tile_begin = (cube_offsets.0 + compute_plane_offset) * block_info.tile_size_x;
        let col_tile_begin = (cube_offsets.1 + accumulator_offset) * block_info.tile_size_y;

        let unit_jump = Self::plane_dim() * gmem.line_size();
        let num_unit_writes = block_info.tile_size_x * block_info.tile_size_y / unit_jump;

        for i in 0..num_unit_writes {
            let unit_write = Self::plane_unit() * gmem.line_size() + i * unit_jump;

            let row = row_tile_begin + unit_write / block_info.tile_size_y;
            let col = col_tile_begin + unit_write % block_info.tile_size_y;

            write_single(gmem, smem_slice, unit_write, row, col);
        }
    }
}

#[cube]
/// Assumes (write_row, write_col) is within bounds
/// Does not account for batch offset
fn write_single<E: Numeric, C: CubePrimitive>(
    gmem: &mut Tensor<Line<E>>,
    smem_slice: &Slice<'_, C>,
    read_position: u32,
    write_row: u32,
    write_col: u32,
) {
    let write_position = (write_row * gmem.stride(gmem.rank() - 2) + write_col) / gmem.line_size();
    gmem[write_position] = Line::cast_from(smem_slice[read_position]);
}
