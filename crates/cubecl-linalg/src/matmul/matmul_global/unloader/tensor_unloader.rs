use crate::matmul::matmul_global::new_tensor_view;
use crate::matmul::matmul_global::TensorView;
use crate::matmul::matmul_stage::TensorWriter;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfo;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::Unloader;

#[derive(CubeType)]
pub struct TensorUnloader<E: Numeric> {
    pub view: TensorView<E>,
    pub stage_info: StageInfo,
}

#[cube]
impl<E: Numeric> Unloader<E> for TensorUnloader<E> {
    type WriteView = TensorView<E>;
    type StageWriter = TensorWriter<E>;

    fn new(tensor: Tensor<Line<E>>, stage_info: StageInfo) -> Self {
        let view = new_tensor_view(tensor, MatrixLayout::RowMajor);
        TensorUnloader::<E> { view, stage_info }
    }

    fn unload(unloader: Self) -> Self::StageWriter {
        TensorWriter::<E> {
            tensor_view: unloader.view,
            stage_info: unloader.stage_info,
        }
    }
}
