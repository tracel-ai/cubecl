use crate::matmul::matmul_global::new_tensor_view;
use crate::matmul::matmul_global::TensorView;
use crate::matmul::matmul_stage::TensorWriter;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::Unloader;

#[derive(CubeType)]
pub struct TensorUnloader<E: Numeric> {
    pub view: TensorView<E>,
}

#[cube]
impl<E: Numeric> Unloader<E> for TensorUnloader<E> {
    type WriteView = TensorView<E>;
    type StageWriter = TensorWriter<E>;

    fn new(tensor: Tensor<Line<E>>) -> Self {
        let view = new_tensor_view(tensor);
        TensorUnloader::<E> { view }
    }

    fn unload(unloader: Self) -> Self::StageWriter {
        TensorWriter::<E> {
            tensor_view: unloader.view,
        }
    }
}
