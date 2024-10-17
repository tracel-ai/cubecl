use crate::matmul::cmma_matmul::stage::Stage;
use crate::matmul::cmma_matmul::stage::{LhsStageReader, RhsStageReader};
use crate::matmul::matmul_global::GlobalMatmul;
use crate::matmul::matmul_global::{Loader, Unloader};
use crate::matmul::matmul_stage::StageMatmul;
use crate::matmul::matmul_stage::XMajorTiling;
use crate::matmul::matmul_tile::TileMatmul;
use crate::matmul::matrix::MatrixLayout;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::cmma_matmul::config::CmmaConfig;
use super::cmma_matmul::global::{LhsTensorLoader, RhsTensorLoader, TensorUnloader, TensorView};
use super::cmma_matmul::stage::OutStageWriter;

#[cube(launch_unchecked)]
pub(crate) fn matmul_instruction_launch<M: TileMatmul<I, O>, I: Numeric, O: Numeric>(
    lhs_input: Tensor<Line<I>>,
    rhs_input: Tensor<Line<I>>,
    mut output: Tensor<Line<O>>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
) {
    let mut lhs = M::init_lhs(layouts.0);
    let mut rhs = M::init_rhs(layouts.1);
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
        LhsStageReader<I, Stage<I, XMajorTiling>>,
        RhsStageReader<I, Stage<I, XMajorTiling>>,
        OutStageWriter<O>,
        Config = CmmaConfig,
    >,
>(
    lhs_data: Tensor<Line<I>>,
    rhs_data: Tensor<Line<I>>,
    out_result: Tensor<Line<O>>,
    #[comptime] config: SMM::Config,
) {
    let mut lhs_loader = LhsTensorLoader::new(lhs_data, config);
    let mut rhs_loader = RhsTensorLoader::new(rhs_data, config);
    let out_unloader = TensorUnloader::new(out_result);

    let lhs_stage_reader = LhsTensorLoader::fill_stage(&mut lhs_loader, config);
    let rhs_stage_reader = RhsTensorLoader::fill_stage(&mut rhs_loader, config);
    let mut out_stage_reader = TensorUnloader::unload(out_unloader);

    let mut acc = SMM::acc_init_zeros();
    SMM::execute(&lhs_stage_reader, &rhs_stage_reader, &mut acc, config);
    SMM::acc_read(&acc, &mut out_stage_reader, config);
}

#[cube(launch_unchecked)]
pub(crate) fn cube_matmul_launch<
    EG: Numeric,
    ES: Numeric,
    GMM: GlobalMatmul<EG, ES, Lhs, Rhs, Out>,
    Lhs: Loader<EG, ES, ReadView = TensorView<EG>, Config = GMM::Config>,
    Rhs: Loader<EG, ES, ReadView = TensorView<EG>, Config = GMM::Config>,
    Out: Unloader<EG, WriteView = TensorView<EG>>,
>(
    lhs_tensor: Tensor<Line<EG>>,
    rhs_tensor: Tensor<Line<EG>>,
    out_tensor: Tensor<Line<EG>>,
    #[comptime] config: GMM::Config,
) {
    let k = lhs_tensor.shape(lhs_tensor.rank() - 1);

    let lhs_loader = Lhs::new(lhs_tensor, config);
    let rhs_loader = Rhs::new(rhs_tensor, config);
    let out_unloader = Out::new(out_tensor);

    GMM::execute(lhs_loader, rhs_loader, out_unloader, (0, k), config);
}
