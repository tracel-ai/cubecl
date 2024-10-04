use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfo;

use super::test_utils::assert_equals_approx;
use crate::matmul::tile_io::loading::array_into_row_major_block_layout;

#[cube(launch_unchecked)]
pub fn array_into_row_major_block_layout_launch<E: Numeric>(
    array_in: Array<Line<E>>,
    mut array_out: Array<Line<E>>,
    #[comptime] block_info: BlockInfo,
    #[comptime] revert: bool,
) {
    array_into_row_major_block_layout(
        array_in.as_slice(),
        array_out.as_slice_mut(),
        block_info,
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
            BlockInfo {
                num_tiles_x: 2,
                num_tiles_y: 2,
                tile_size_x: 2,
                tile_size_y: 2,
            },
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
