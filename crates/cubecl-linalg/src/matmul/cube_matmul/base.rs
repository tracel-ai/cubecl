use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_core::server::ComputeServer;

use crate::matmul::cmma_matmul::BlockInfo;
use crate::matmul::launch::cube_matmul_launch;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::tensor_io::{TensorReader, TensorWriter};
use crate::matmul::tests::dummy_tile::{DummyLhsReader, DummyRhsReader, DummyWriter};
use crate::matmul::{BlockMatmul, CubeMatmul, Matmul, TensorMatmul};

struct CmmaCubeMatmul<
    E: Numeric,
    BM: BlockMatmul<E, DummyLhsReader<E>, DummyRhsReader<E>, DummyWriter<E>>,
> {
    _elem: PhantomData<E>,
    _block_matmul: PhantomData<BM>,
}

#[derive(CubeType)]
pub struct LhsTensorReader {}
#[derive(CubeType)]
pub struct RhsTensorReader {}
#[derive(CubeType)]
pub struct OutTensorWriter {}

#[cube]
impl<E: Numeric> TensorReader<E> for LhsTensorReader {
    type TileReader = DummyLhsReader<E>;

    fn read(reader: &Self, k_offset: u32) -> Self::TileReader {
        // TODO, totally TMP
        DummyLhsReader::<E> {
            memory: SharedMemory::new(1),
            block_info: BlockInfo {
                num_tiles_x: 1,
                num_tiles_y: 1,
                tile_size_x: 1,
                tile_size_y: 1,
            },
        }
    }
}

#[cube]
impl<E: Numeric> TensorReader<E> for RhsTensorReader {
    type TileReader = DummyRhsReader<E>;

    fn read(reader: &Self, k_offset: u32) -> Self::TileReader {
        // TODO, totally TMP
        DummyRhsReader::<E> {
            memory: SharedMemory::new(1),
            block_info: BlockInfo {
                num_tiles_x: 1,
                num_tiles_y: 1,
                tile_size_x: 1,
                tile_size_y: 1,
            },
        }
    }
}

#[cube]
impl<E: Numeric> TensorWriter<E> for OutTensorWriter {
    type TileWriter = DummyWriter<E>;

    fn make_tile_writer() -> Self::TileWriter {
        // TODO, totally TMP
        DummyWriter::<E> {
            memory: SharedMemory::new(1),
            block_info: BlockInfo {
                num_tiles_x: 1,
                num_tiles_y: 1,
                tile_size_x: 1,
                tile_size_y: 1,
            },
        }
    }
    fn write(writer: &mut Self, tile_writer: Self::TileWriter) {}
}

impl<
        Elem: Numeric,
        BM: BlockMatmul<
            Elem,
            <LhsTensorReader as TensorReader<Elem>>::TileReader,
            <RhsTensorReader as TensorReader<Elem>>::TileReader,
            <OutTensorWriter as TensorWriter<Elem>>::TileWriter,
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
            <LhsTensorReader as TensorReader<Elem>>::TileReader,
            <RhsTensorReader as TensorReader<Elem>>::TileReader,
            <OutTensorWriter as TensorWriter<Elem>>::TileWriter,
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
        E: Numeric,
        BM: BlockMatmul<
            E,
            <LhsTensorReader as TensorReader<E>>::TileReader,
            <RhsTensorReader as TensorReader<E>>::TileReader,
            <OutTensorWriter as TensorWriter<E>>::TileWriter,
        >,
    > CubeMatmul<E, LhsTensorReader, RhsTensorReader, OutTensorWriter> for CmmaCubeMatmul<E, BM>
{
    fn execute(
        lhs: LhsTensorReader,
        rhs: RhsTensorReader,
        mut out: OutTensorWriter,
        k_range: (u32, u32),
        layouts: (MatrixLayout, MatrixLayout),
    ) {
        let k_step = BM::K;
        let range = k_range.1 - k_range.0;
        let num_loops = (range + k_step - 1) / k_step;

        let mut acc = BM::acc_init_zeros();

        for block_iter in 0..num_loops {
            let k_offset = block_iter * k_step;

            let lhs_tile_reader = LhsTensorReader::read(&lhs, k_offset);
            let rhs_tile_reader = RhsTensorReader::read(&rhs, k_offset);

            BM::execute(lhs_tile_reader, rhs_tile_reader, &mut acc, layouts);
        }

        let mut tile_writer = OutTensorWriter::make_tile_writer();
        BM::acc_read(&acc, &mut tile_writer);
        OutTensorWriter::write(&mut out, tile_writer);
    }
}
