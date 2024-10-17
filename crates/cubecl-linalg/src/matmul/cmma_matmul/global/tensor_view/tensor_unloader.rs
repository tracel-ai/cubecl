use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::global::new_tensor_view;
use crate::matmul::cmma_matmul::stage::OutStageWriter;
use crate::matmul::matmul_global::Unloader;

use super::TensorView;

#[derive(CubeType)]
pub struct TensorUnloader<E: Numeric> {
    pub view: TensorView<E>,
}

#[cube]
impl<E: Numeric> Unloader<E> for TensorUnloader<E> {
    type WriteView = TensorView<E>;
    type StageWriter = OutStageWriter<E>;

    fn new(tensor: Tensor<Line<E>>) -> Self {
        let view = new_tensor_view(tensor);
        TensorUnloader::<E> { view }
    }

    fn unload(unloader: Self) -> Self::StageWriter {
        OutStageWriter::<E> {
            tensor_view: unloader.view,
        }
    }
}
