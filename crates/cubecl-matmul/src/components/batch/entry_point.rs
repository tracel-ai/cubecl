use crate::components::batch::CubeCountArgs;
use crate::components::batch::base::BatchMatmul;
use crate::components::{
    Quantized,
    batch::{BatchConfig, BatchMatmulFamily},
    global::args::{self, MatmulArgs, TensorInput, TensorOutput},
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::{
    CubeOption,
    tensor::r#virtual::{ReadWrite, VirtualTensor},
};

type Input<Args, EI> = <Args as MatmulArgs>::Input<EI>;
type Output<Args, EO> = <Args as MatmulArgs>::Output<EO>;

#[cube(launch_unchecked)]
pub(crate) fn matmul<
    Args: MatmulArgs,
    EI: Numeric,
    ES: Numeric,
    EA: Numeric,
    EO: Numeric,
    BMMF: BatchMatmulFamily,
>(
    inputs: &Input<Args, EI>,
    output: &mut Output<Args, EO>,
    cube_count_args: CubeCountArgs,
    #[comptime] config: BMMF::Config,
) {
    let mut state = Args::init_state(inputs, output);

    let lhs = TensorInput::<EI, EO, Args>::new(&state, args::TensorInputIdent::Lhs);
    let rhs = TensorInput::<EI, EO, Args>::new(&state, args::TensorInputIdent::Rhs);
    let mut out = TensorOutput::<EI, EO, Args>::new(&mut state);

    let lhs = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&lhs);
    let rhs = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&rhs);
    let out = VirtualTensor::<EO, ReadWrite>::new::<TensorOutput<EI, EO, Args>>(&mut out);

    if config.quantized() {
        let quantization = Args::quantization::<(EI, ES, EA, EO, Quantized)>(&state);
        BMMF::Matmul::<(EI, ES, EA, EO, Quantized)>::execute(
            lhs,
            rhs,
            out,
            CubeOption::new_Some(quantization),
            cube_count_args,
            config,
        );
    } else {
        BMMF::Matmul::<(EI, ES, EA, EO)>::execute(
            lhs,
            rhs,
            out,
            CubeOption::new_None(),
            cube_count_args,
            config,
        );
    };
}
