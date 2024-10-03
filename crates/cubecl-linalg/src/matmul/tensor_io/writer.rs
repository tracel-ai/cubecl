use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::BlockInfo;
use crate::matmul::tensor_io::TensorWriter;
use crate::matmul::tile_io::writer::DummyTensorWriter;

#[derive(CubeType)]
pub struct OutTensorWriter<E: Numeric> {
    out: Tensor<Line<E>>,
    block_info: BlockInfo,
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
