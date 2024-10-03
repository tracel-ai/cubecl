use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::{into_runtime, BlockInfo, BlockInfoR};
use crate::matmul::tensor_io::TensorWriter;
use crate::matmul::tile_io::writer::DummyTensorWriter;

#[derive(CubeType)]
pub struct OutTensorWriter<E: Numeric> {
    out: Tensor<Line<E>>,
    block_info: BlockInfoR,
}

#[cube]
impl<E: Numeric> TensorWriter<E> for OutTensorWriter<E> {
    type TileWriter = DummyTensorWriter<E>;

    fn as_tile_writer(writer: Self) -> Self::TileWriter {
        DummyTensorWriter::<E> {
            memory: writer.out,
            block_info: writer.block_info,
        }
    }
}

#[cube]
pub fn new_out_writer<E: Numeric>(
    gmem: Tensor<Line<E>>,
    #[comptime] block_info: BlockInfo,
) -> OutTensorWriter<E> {
    OutTensorWriter::<E> {
        out: gmem,
        block_info: into_runtime(block_info),
    }
}
