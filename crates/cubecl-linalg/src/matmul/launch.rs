use crate::matmul::cmma_matmul::global::{
    new_lhs_tensor_loader, new_rhs_tensor_loader, new_tensor_unloader,
};
use crate::matmul::matmul_global::GlobalMatmul;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::cmma_matmul::global::{LhsTensorLoader, RhsTensorLoader, TensorUnloader};
use super::matmul_global::GmmConfig;

#[cube(launch_unchecked)]
pub(crate) fn cube_matmul_launch<
    EG: Numeric,
    ES: Numeric,
    GMM: GlobalMatmul<
        EG,
        ES,
        LhsTensorLoader<EG, ES, G>,
        RhsTensorLoader<EG, ES, G>,
        TensorUnloader<EG, G>,
        G,
    >,
    G: GmmConfig,
>(
    lhs_tensor: Tensor<Line<EG>>,
    rhs_tensor: Tensor<Line<EG>>,
    out_tensor: Tensor<Line<EG>>,
    #[comptime] config: G,
) {
    let k = lhs_tensor.shape(lhs_tensor.rank() - 1);

    let lhs_loader = new_lhs_tensor_loader(lhs_tensor, config);
    let rhs_loader = new_rhs_tensor_loader(rhs_tensor, config);
    let out_unloader = new_tensor_unloader(out_tensor);

    GMM::execute(lhs_loader, rhs_loader, out_unloader, (0, k), config);
}
