use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::server::ComputeServer;

use crate::matmul::cmma_matmul::BlockInfo;
use crate::matmul::launch::cube_matmul_launch;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tensor_io::reader::{LhsTensorReader, RhsTensorReader};
use crate::matmul::tensor_io::writer::OutTensorWriter;
use crate::matmul::tensor_io::{TensorLoader, TensorWriter};
use crate::matmul::tile_io::writer::DummyTensorWriter;
use crate::matmul::{BlockKind, BlockMatmul, CubeMatmul, Matmul, TensorMatmul};

use crate::matmul::tile_io::reader::{SmemLhsReader, SmemRhsReader};

pub struct CmmaCubeMatmul<
    E: Numeric,
    BM: BlockMatmul<E, SmemLhsReader<E>, SmemRhsReader<E>, DummyTensorWriter<E>>,
> {
    _elem: PhantomData<E>,
    _block_matmul: PhantomData<BM>,
}

impl<
        Elem: Numeric,
        BM: BlockMatmul<
            Elem,
            <LhsTensorReader<Elem> as TensorLoader<Elem>>::TileReader,
            <RhsTensorReader<Elem> as TensorLoader<Elem>>::TileReader,
            <OutTensorWriter<Elem> as TensorWriter<Elem>>::TileWriter,
        >,
    > Matmul<Elem, Elem> for CmmaCubeMatmul<Elem, BM>
{
    fn cube_dim_resources() -> CubeDim {
        BM::cube_dim_resources()
    }

    fn cube_count_resources<S: ComputeServer>() -> CubeCount<S> {
        CubeCount::Static(1, 1, 1)
    }
}

impl<
        Elem: Numeric,
        BM: BlockMatmul<
            Elem,
            <LhsTensorReader<Elem> as TensorLoader<Elem>>::TileReader,
            <RhsTensorReader<Elem> as TensorLoader<Elem>>::TileReader,
            <OutTensorWriter<Elem> as TensorWriter<Elem>>::TileWriter,
        >,
    > TensorMatmul<Elem, Elem> for CmmaCubeMatmul<Elem, BM>
{
    unsafe fn launch_unchecked<R: Runtime>(
        client: &ComputeClient<<R as Runtime>::Server, <R as Runtime>::Channel>,
        cube_dim: CubeDim,
        cube_count: CubeCount<<R as Runtime>::Server>,
        lhs: TensorArg<'_, R>,
        rhs: TensorArg<'_, R>,
        out: TensorArg<'_, R>,
        layouts: (MatrixLayout, MatrixLayout),
    ) {
        cube_matmul_launch::launch_unchecked::<Self, Elem, R>(
            &client, cube_count, cube_dim, lhs, rhs, out, layouts,
        );
    }
}

#[cube]
impl<
        Elem: Numeric,
        BM: BlockMatmul<
            Elem,
            <LhsTensorReader<Elem> as TensorLoader<Elem>>::TileReader,
            <RhsTensorReader<Elem> as TensorLoader<Elem>>::TileReader,
            <OutTensorWriter<Elem> as TensorWriter<Elem>>::TileWriter,
        >,
    > CubeMatmul<Elem, LhsTensorReader<Elem>, RhsTensorReader<Elem>, OutTensorWriter<Elem>>
    for CmmaCubeMatmul<Elem, BM>
{
    fn execute(
        mut lhs: LhsTensorReader<Elem>,
        mut rhs: RhsTensorReader<Elem>,
        out: OutTensorWriter<Elem>,
        k_range: (u32, u32),
        layouts: (MatrixLayout, MatrixLayout),
    ) {
        let k_step = BM::K;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let mut acc = BM::acc_init_zeros();

        for block_iter in 0..num_loops {
            let k_offset = block_iter * k_step;

            let lhs_tile_reader = LhsTensorReader::load_tile(&mut lhs, k_offset);
            let rhs_tile_reader = RhsTensorReader::load_tile(&mut rhs, k_offset);

            BM::execute(lhs_tile_reader, rhs_tile_reader, &mut acc, layouts);
        }

        let mut tile_writer = OutTensorWriter::as_tile_writer(out);
        BM::acc_read(&acc, &mut tile_writer);
    }

    fn block_info(#[comptime] block_kind: BlockKind) -> BlockInfo {
        BM::block_info(block_kind)
    }
}
