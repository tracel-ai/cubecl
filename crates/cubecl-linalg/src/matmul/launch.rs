use crate::matmul::matmul_global::ArrayUnloader;
use crate::matmul::matmul_global::GlobalMatmul;
use crate::matmul::matmul_global::TensorView;
use crate::matmul::matmul_global::{LhsArrayLoader, RhsArrayLoader};
use crate::matmul::matmul_global::{Loader, Unloader};
use crate::matmul::matmul_stage::ArrayWriter;
use crate::matmul::matmul_stage::LhsStageReader;
use crate::matmul::matmul_stage::RhsStageReader;
use crate::matmul::matmul_stage::SharedMemoryStage;
use crate::matmul::matmul_stage::StageMatmul;
use crate::matmul::matmul_stage::XMajorTiling;
use crate::matmul::matmul_tile::MatmulInstruction;
use crate::matmul::matrix_layout::MatrixLayout;
use crate::matmul::stage_info::StageInfos;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

#[cube(launch_unchecked)]
pub(crate) fn matmul_instruction_launch<M: MatmulInstruction<I, O>, I: Numeric, O: Numeric>(
    lhs_array: Array<I>,
    rhs_array: Array<I>,
    mut out_array: Array<O>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
) {
    let mut lhs = M::init_lhs(layouts.0);
    let mut rhs = M::init_rhs(layouts.1);
    let mut out = M::init_output();

    M::fill_lhs(lhs_array.as_slice(), &mut lhs);
    M::fill_rhs(rhs_array.as_slice(), &mut rhs);

    M::execute(&lhs, &rhs, &mut out);
    M::read_output(&out, out_array.as_slice_mut());
}

#[cube(launch_unchecked)]
pub(crate) fn stage_matmul_launch<
    I: Numeric,
    O: Numeric,
    BM: StageMatmul<
        I,
        O,
        LhsStageReader<I, SharedMemoryStage<I, XMajorTiling>>,
        RhsStageReader<I, SharedMemoryStage<I, XMajorTiling>>,
        ArrayWriter<O>,
    >,
>(
    lhs_data: Array<Line<I>>,
    rhs_data: Array<Line<I>>,
    out_result: Array<Line<O>>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
    #[comptime] stage_infos: StageInfos,
) {
    let mut lhs_loader = LhsArrayLoader::new(lhs_data, layouts.0, stage_infos.lhs);
    let mut rhs_loader = RhsArrayLoader::new(rhs_data, layouts.1, stage_infos.rhs);
    let out_unloader = ArrayUnloader::new(out_result, stage_infos.out);

    let lhs_stage_reader = LhsArrayLoader::fill_block(&mut lhs_loader);
    let rhs_stage_reader = RhsArrayLoader::fill_block(&mut rhs_loader);
    let mut out_stage_reader = ArrayUnloader::unload(out_unloader);

    let mut acc = BM::acc_init_zeros();
    BM::execute(&lhs_stage_reader, &rhs_stage_reader, &mut acc);
    BM::acc_read(&acc, &mut out_stage_reader);
}

#[cube(launch_unchecked)]
pub(crate) fn cube_matmul_launch<
    EG: Numeric,
    ES: Numeric,
    CM: GlobalMatmul<EG, ES, Lhs, Rhs, Out>,
    Lhs: Loader<EG, ES, GlobalView = TensorView<EG>>,
    Rhs: Loader<EG, ES, GlobalView = TensorView<EG>>,
    Out: Unloader<EG, GlobalView = TensorView<EG>>,
>(
    lhs_tensor: Tensor<Line<EG>>,
    rhs_tensor: Tensor<Line<EG>>,
    out_tensor: Tensor<Line<EG>>,
    #[comptime] layouts: (MatrixLayout, MatrixLayout),
    #[comptime] stage_infos: StageInfos,
) {
    let k = lhs_tensor.shape(lhs_tensor.rank() - 1);

    let lhs_loader = Lhs::new(lhs_tensor, layouts.0, stage_infos.lhs);
    let rhs_loader = Rhs::new(rhs_tensor, layouts.1, stage_infos.rhs);
    let out_unloader = Out::new(out_tensor, stage_infos.out);

    CM::execute(lhs_loader, rhs_loader, out_unloader, (0, k));
}
