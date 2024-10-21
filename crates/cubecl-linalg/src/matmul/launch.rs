use crate::matmul::cmma_matmul::global::{
    new_lhs_tensor_loader, new_rhs_tensor_loader, new_tensor_unloader,
};
use crate::matmul::cmma_matmul::stage::{LhsStageReader, RhsStageReader};
use crate::matmul::matmul_global::GlobalMatmul;
use crate::matmul::matmul_global::Loader;
use crate::matmul::matmul_stage::StageMatmul;
use crate::matmul::matmul_tile::TileMatmul;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::cmma_matmul::global::{LhsTensorLoader, RhsTensorLoader, TensorUnloader};
use super::matmul_global::GmmConfig;
use super::matmul_tile::TmmConfig;

#[cube(launch_unchecked)]
pub(crate) fn matmul_instruction_launch<
    M: TileMatmul<I, O, T>,
    I: Numeric,
    O: Numeric,
    T: TmmConfig,
>(
    lhs_input: Tensor<Line<I>>,
    rhs_input: Tensor<Line<I>>,
    mut output: Tensor<Line<O>>,
    #[comptime] config: T,
) {
    let mut lhs = M::init_lhs(config);
    let mut rhs = M::init_rhs(config);
    let mut out = M::init_output();

    M::fill_lhs(lhs_input.as_slice(), &mut lhs);
    M::fill_rhs(rhs_input.as_slice(), &mut rhs);

    M::execute(&lhs, &rhs, &mut out);
    M::read_output(&out, output.as_slice_mut());
}

#[cube(launch_unchecked)]
pub(crate) fn stage_matmul_launch<
    I: Numeric,
    O: Numeric,
    SMM: StageMatmul<
        I,
        O,
        LhsStageReader<I, G::SmmConfig>,
        RhsStageReader<I, G::SmmConfig>,
        G::SmmConfig,
    >,
    G: GmmConfig,
>(
    lhs_data: Tensor<Line<I>>,
    rhs_data: Tensor<Line<I>>,
    out_result: Tensor<Line<O>>,
    #[comptime] config: G,
) {
    let mut lhs_loader = new_lhs_tensor_loader::<I, I, G>(lhs_data, config);
    let mut rhs_loader = new_rhs_tensor_loader::<I, I, G>(rhs_data, config);
    let mut out_unloader = new_tensor_unloader::<O, G>(out_result);

    let lhs_stage_reader = LhsTensorLoader::fill_stage(&mut lhs_loader, config);
    let rhs_stage_reader = RhsTensorLoader::fill_stage(&mut rhs_loader, config);

    let mut acc = SMM::acc_init_zeros();
    SMM::execute(
        &lhs_stage_reader,
        &rhs_stage_reader,
        &mut acc,
        config.to_smm_config(),
    );
    SMM::acc_read::<TensorUnloader<O, G>, G>(
        &acc,
        &mut out_unloader,
        config.to_smm_config(),
        config,
    );
}

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
    #[comptime] config: GMM::Config,
) {
    let k = lhs_tensor.shape(lhs_tensor.rank() - 1);

    let lhs_loader = new_lhs_tensor_loader(lhs_tensor, config);
    let rhs_loader = new_rhs_tensor_loader(rhs_tensor, config);
    let out_unloader = new_tensor_unloader(out_tensor);

    GMM::execute(lhs_loader, rhs_loader, out_unloader, (0, k), config);
}
