use std::marker::PhantomData;

use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::cmma_matmul::global::new_tensor_view;
use crate::matmul::cmma_matmul::stage::OutStageWriter;
use crate::matmul::matmul_global::{GmmConfig, Unloader};

use super::TensorView;

#[derive(CubeType)]
pub struct TensorUnloader<EG: Numeric, G: GmmConfig> {
    pub view: TensorView<EG>,
    pub _config: PhantomData<G>,
}

#[cube]
impl<EG: Numeric, G: GmmConfig> Unloader<EG, G> for TensorUnloader<EG, G> {
    type StageWriter = OutStageWriter<EG>;

    fn unload(unloader: Self) -> Self::StageWriter {
        OutStageWriter::<EG> {
            tensor_view: unloader.view,
        }
    }
}

#[cube]
pub fn new_tensor_unloader<EG: Numeric, G: GmmConfig>(
    tensor: Tensor<Line<EG>>,
) -> TensorUnloader<EG, G> {
    let view = new_tensor_view(tensor);
    TensorUnloader::<EG, G> {
        view,
        _config: PhantomData::<G>.runtime(),
    }
}
