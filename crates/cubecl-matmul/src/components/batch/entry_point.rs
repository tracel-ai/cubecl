use crate::components::batch::CubeCountInput;
use crate::components::batch::base::BatchMatmul;
use crate::components::global::args::{TensorLhs, TensorRhs};
use crate::components::{
    batch::{BatchConfig, BatchMatmulFamily},
    global::args::{self, MatmulArgs, TensorOutput},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

type Input<Args, Lhs, Rhs> = <Args as MatmulArgs>::Input<Lhs, Rhs>;
type Output<Args, EO> = <Args as MatmulArgs>::Output<EO>;

#[cube(launch_unchecked)]
/// Launches the matmul kernel
pub(crate) fn matmul<
    Args: MatmulArgs,
    LhsG: Numeric,
    RhsG: Numeric,
    LhsS: Numeric,
    RhsS: Numeric,
    EA: Numeric,
    EO: Numeric,
    BMMF: BatchMatmulFamily,
>(
    inputs: &Input<Args, LhsG, RhsG>,
    output: &mut Output<Args, EO>,
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

    let lhs = TensorLhs::<LhsG, RhsG, EO, Args>::new(&state);
    let rhs = TensorRhs::<LhsG, RhsG, EO, Args>::new(&state);
    let mut out = TensorOutput::<EI, EO, Args>::new(&mut state);

    let lhs = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&lhs);
    let rhs = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&rhs);
    let out = VirtualTensor::<EO, ReadWrite>::new::<TensorOutput<EI, EO, Args>>(&mut out);

    BMMF::Matmul::<(EI, ES, EA, EO)>::execute(lhs, rhs, out, cube_count_args, config);
}
