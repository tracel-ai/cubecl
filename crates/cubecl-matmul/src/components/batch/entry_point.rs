use crate::components::batch::base::BatchMatmul;
use crate::components::global::args::{TensorLhs, TensorRhs};
use crate::components::{batch::CubeCountInput, global::args::TensorAcc};
use crate::components::{
    batch::{BatchConfig, BatchMatmulFamily},
    global::args::{MatmulArgs, TensorOutput},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{CubeOption, CubeOptionExpand, tensor::r#virtual::VirtualTensor};

type Input<Args, Lhs, Rhs, AccG> = <Args as MatmulArgs>::Input<Lhs, Rhs, AccG>;
type Output<Args, AccG> = <Args as MatmulArgs>::Output<AccG>;

#[cube(launch_unchecked)]
/// Launches the matmul kernel
pub(crate) fn matmul<
    Args: MatmulArgs,
    LhsG: Numeric,
    RhsG: Numeric,
    AccG: Numeric,
    LhsS: Numeric,
    RhsS: Numeric,
    AccS: Numeric,
    BMMF: BatchMatmulFamily,
>(
    inputs: &Input<Args, LhsG, RhsG, AccG>,
    output: &mut Output<Args, AccG>,
    cube_count_args: CubeCountInput,
    #[comptime] config: BMMF::Config,
) {
    #[allow(clippy::collapsible_if)]
    if comptime!(config.can_yield_extra_cubes()) {
        if CUBE_POS >= cube_count_args.num_valid_cubes() {
            terminate!()
        }
    }

    let mut state = Args::init_state(inputs, output);

    let lhs = TensorLhs::<LhsG, RhsG, AccG, Args>::new(&state);
    let rhs = TensorRhs::<LhsG, RhsG, AccG, Args>::new(&state);
    let mut out = TensorOutput::<LhsG, RhsG, AccG, Args>::new(&mut state);

    let has_acc = Args::has_acc(&state);
    let acc: CubeOption<VirtualTensor<AccG>> = match has_acc {
        CubeOption::Some(_) => {
            let acc = TensorAcc::<LhsG, RhsG, AccG, Args>::new(&state);
            let acc = VirtualTensor::<AccG>::new::<TensorAcc<LhsG, RhsG, AccG, Args>>(&acc);
            CubeOption::new_Some(acc)
        }
        CubeOption::None => CubeOption::new_None(),
    };

    let lhs = VirtualTensor::<LhsG>::new::<TensorLhs<LhsG, RhsG, AccG, Args>>(&lhs);
    let rhs = VirtualTensor::<RhsG>::new::<TensorRhs<LhsG, RhsG, AccG, Args>>(&rhs);
    let out =
        VirtualTensor::<AccG, ReadWrite>::new::<TensorOutput<LhsG, RhsG, AccG, Args>>(&mut out);

    BMMF::Matmul::<(LhsG, RhsG, AccG, LhsS, RhsS, AccS)>::execute(
        lhs,
        rhs,
        acc,
        out,
        cube_count_args,
        config,
    );
}
