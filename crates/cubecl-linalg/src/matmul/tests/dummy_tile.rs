use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfo;
use crate::matmul::tile_io::TileReader;
use crate::matmul::tile_io::TileWriter;

use super::test_utils::assert_equals_approx;

#[derive(CubeType)]
pub struct DummyLhsReader<E: Numeric> {
    pub memory: SharedMemory<Line<E>>,
    pub block_info: BlockInfo,
}

#[derive(CubeType)]
pub struct DummyRhsReader<E: Numeric> {
    pub memory: SharedMemory<Line<E>>,
    pub block_info: BlockInfo,
}

#[derive(CubeType)]
pub struct DummyWriter<E: Numeric> {
    pub memory: SharedMemory<Line<E>>,
    pub block_info: BlockInfo,
}

#[cube]
impl<E: Numeric> TileReader<Line<E>> for DummyLhsReader<E> {
    fn read(
        reader: &Self,
        compute_plane_offset: u32,
        buffer_offset: u32,
        _accumulator_offset: u32,
    ) -> &Slice<'_, Line<E>> {
        let num_tile_elements = reader.block_info.tile_size_x * reader.block_info.tile_size_y;
        let num_tile_offset = compute_plane_offset * reader.block_info.num_tiles_y + buffer_offset;

        let start = num_tile_offset * num_tile_elements;
        reader.memory.slice(start, start + num_tile_elements)
    }
}

#[cube]
impl<E: Numeric> TileReader<Line<E>> for DummyRhsReader<E> {
    fn read(
        reader: &Self,
        _compute_plane_offset: u32,
        buffer_offset: u32,
        accumulator_offset: u32,
    ) -> &Slice<'_, Line<E>> {
        let num_tile_elements = reader.block_info.tile_size_x * reader.block_info.tile_size_y;
        let num_tile_offset = buffer_offset * reader.block_info.num_tiles_y + accumulator_offset;

        let start = num_tile_offset * num_tile_elements;
        reader.memory.slice(start, start + num_tile_elements)
    }
}

#[cube]
impl<E: Numeric> TileWriter<Line<E>> for DummyWriter<E> {
    fn write_with_cast<C: Numeric>(
        writer: &mut Self,
        slice: &Slice<'_, C>,
        compute_plane_offset: u32,
        accumulator_offset: u32,
    ) {
        let num_tile_elements = writer.block_info.tile_size_x * writer.block_info.tile_size_y;
        let num_tile_offset =
            compute_plane_offset * writer.block_info.num_tiles_y + accumulator_offset;

        let write_offset = num_tile_offset * num_tile_elements;
        for i in 0..num_tile_elements {
            writer.memory[i + write_offset] = Line::new(E::cast_from(slice[i]));
        }
    }
}

#[cube]
pub(crate) fn array_into_row_major_block_layout<E: Numeric>(
    original_array: &Slice<'_, Line<E>>,
    array_out: &mut SliceMut<'_, Line<E>>,
    block_info: BlockInfo,
    #[comptime] revert: bool,
) {
    let stride_x = block_info.num_tiles_y * block_info.tile_size_y;

    for tile_x in 0..block_info.num_tiles_x {
        let tiled_offset_tile_x =
            tile_x * block_info.num_tiles_y * block_info.tile_size_x * block_info.tile_size_y;
        let continuous_offset_tile_x = tile_x * block_info.tile_size_x * stride_x;

        for tile_y in 0..block_info.num_tiles_y {
            let tiled_offset_tile_y = tile_y * block_info.tile_size_x * block_info.tile_size_y;
            let continuous_offset_tile_y = tile_y * block_info.tile_size_y;

            for elem_x in 0..block_info.tile_size_x {
                let tiled_offset_elem_x = elem_x * block_info.tile_size_y;
                let continuous_offset_elem_x = elem_x * stride_x;

                for elem_y in 0..block_info.tile_size_y {
                    let tiled_offset_elem_y = elem_y;
                    let continuous_offset_elem_y = elem_y;

                    let tiled_offset = tiled_offset_tile_x
                        + tiled_offset_tile_y
                        + tiled_offset_elem_x
                        + tiled_offset_elem_y;
                    let continuous_offset = continuous_offset_tile_x
                        + continuous_offset_tile_y
                        + continuous_offset_elem_x
                        + continuous_offset_elem_y;

                    if !revert {
                        array_out[tiled_offset] = original_array[continuous_offset];
                    } else {
                        array_out[continuous_offset] = original_array[tiled_offset];
                    }
                }
            }
        }
    }
}

#[cube(launch_unchecked)]
pub fn array_into_row_major_block_layout_launch<E: Numeric>(
    array_in: Array<Line<E>>,
    mut array_out: Array<Line<E>>,
    #[comptime] revert: bool,
) {
    array_into_row_major_block_layout(
        array_in.as_slice(),
        array_out.as_slice_mut(),
        BlockInfo {
            num_tiles_x: 2,
            num_tiles_y: 2,
            tile_size_x: 2,
            tile_size_y: 2,
        },
        revert,
    );
}

pub fn array_into_row_major_block_layout_test<R: Runtime>(revert: bool, device: &R::Device) {
    let client = R::client(device);

    let data_no_layout = [
        0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,
    ];
    let data_layout = [
        0., 1., 4., 5., 2., 3., 6., 7., 8., 9., 12., 13., 10., 11., 14., 15.,
    ];

    let array_in = client.create(f32::as_bytes(&f32::from_values(&match revert {
        true => data_layout,
        false => data_no_layout,
    })));
    let array_out = client.empty(16 * f32::as_elem().size());

    unsafe {
        array_into_row_major_block_layout_launch::launch_unchecked::<f32, R>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(32, 1, 1),
            ArrayArg::from_raw_parts(&array_in, 16, 1),
            ArrayArg::from_raw_parts(&array_out, 16, 1),
            revert,
        );
    }

    if let Err(e) = assert_equals_approx::<f32, R>(
        &client,
        array_out,
        &match revert {
            true => data_no_layout,
            false => data_layout,
        },
        10e-1,
    ) {
        panic!("{}", e);
    }
}
