use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::CubeElement;

use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tile_io::TileReader;
use crate::matmul::tile_io::TileWriter;
use crate::matmul::BlockMatmul;
use crate::matmul::MatmulInstruction;

use super::test_utils::assert_equals_approx;
use super::test_utils::matmul_cpu_reference;

#[derive(CubeType)]
pub struct ArrayTile<E: Numeric> {
    pub array: Array<Line<E>>,
}

#[cube]
impl<E: Numeric> TileReader<Line<E>> for ArrayTile<E> {
    const NUM_TILES_X: u32 = 1;
    const NUM_TILES_Y: u32 = 1;

    const TILE_SIZE_X: u32 = 16;
    const TILE_SIZE_Y: u32 = 16;

    fn read(reader: &Self, _pos_x: u32, _pos_y: u32) -> &Slice<'_, Line<E>> {
        reader.array.as_slice()
    }
}

#[cube]
impl<E: Numeric> TileWriter<Line<E>> for ArrayTile<E> {
    fn from_instruction_to_output<'a, Instr: MatmulInstruction<I, O>, I: Numeric, O: Numeric>(
        writer: &'a mut Self,
        instr_out: &Instr::Out,
        _pos_x: u32,
        _pos_y: u32,
    ) -> &'a mut SliceMut<'a, Line<E>> {
        let slice_mut = writer.array.as_slice_mut();
        Instr::read_output(instr_out, slice_mut);
        slice_mut
    }
}

#[cube(launch_unchecked)]
fn block_matmul_launch<BM: BlockMatmul<E, ArrayTile<E>, ArrayTile<E>, ArrayTile<E>>, E: Numeric>(
    lhs_slice: Array<Line<E>>,
    rhs_slice: Array<Line<E>>,
    out_slice: Array<Line<E>>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
) {
    let lhs = ArrayTile::<E> { array: lhs_slice };
    let rhs = ArrayTile::<E> { array: rhs_slice };
    let mut out = ArrayTile::<E> { array: out_slice };

    let mut acc = BM::acc_init_zeros();
    BM::execute(lhs, rhs, &mut acc, layouts);
    BM::acc_read(&mut acc, &mut out);
}

/// Exported test
pub fn test_block_matmul<BM, E, R>(device: &R::Device)
where
    BM: BlockMatmul<E, ArrayTile<E>, ArrayTile<E>, ArrayTile<E>>,
    E: Numeric + CubeElement,
    R: Runtime,
{
    let client = R::client(device);
    let lhs_size = (BM::M * BM::K) as usize;
    let rhs_size = (BM::K * BM::N) as usize;
    let out_size = (BM::M * BM::N) as usize;

    let lhs_data: Vec<f32> = (0..lhs_size).map(|x| x as f32 / 100.).collect();
    let rhs_data: Vec<f32> = (0..rhs_size).map(|x| x as f32 / 100.).collect();

    let lhs = client.create(E::as_bytes(&E::from_values(&lhs_data)));
    let rhs = client.create(E::as_bytes(&E::from_values(&rhs_data)));
    let out = client.empty(out_size * E::as_elem().size());

    let cube_dim = CubeDim::new(32, 1, 1);
    let cube_count = CubeCount::Static(1, 1, 1);

    unsafe {
        block_matmul_launch::launch_unchecked::<BM, E, R>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(&lhs, lhs_size, 1),
            ArrayArg::from_raw_parts(&rhs, rhs_size, 1),
            ArrayArg::from_raw_parts(&out, out_size, 1),
            (MatrixLayout::Row, MatrixLayout::Row),
        );
    }

    let expected = matmul_cpu_reference(
        &lhs_data,
        &rhs_data,
        BM::M as usize,
        BM::N as usize,
        BM::K as usize,
    );
    if let Err(e) = assert_equals_approx::<E, R>(&client, out, &expected, 10e-1) {
        panic!("{}", e);
    }
}
