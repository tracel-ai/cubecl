use crate::components::args;
use crate::components::args::AttentionArgs;
use crate::components::args::TensorInput;
use crate::components::args::TensorOutput;
use crate::components::batch::BatchAttentionFamily;
use crate::components::batch::CubeCountInput;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::{ReadWrite, VirtualTensor};

type Input<Args, EI> = <Args as AttentionArgs>::Input<EI>;
type Output<Args, EO> = <Args as AttentionArgs>::Output<EO>;

#[cube(launch_unchecked)]
/// Launches the attention kernel
pub(crate) fn attention<
    Args: AttentionArgs,
    EI: Numeric,
    ES: Numeric,
    EM: Numeric,
    EA: Numeric,
    EO: Numeric,
    BMMF: BatchAttentionFamily,
>(
    inputs: &Input<Args, EI>,
    output: &mut Output<Args, EO>,
    cube_count_args: CubeCountInput,
    #[comptime] config: BMMF::Config,
) {
    let mut state = Args::init_state(inputs, output);

    let query = TensorInput::<EI, EO, Args>::new(&state, args::TensorInputIdent::Query);
    let key = TensorInput::<EI, EO, Args>::new(&state, args::TensorInputIdent::Key);
    let value = TensorInput::<EI, EO, Args>::new(&state, args::TensorInputIdent::Value);
    let mut out = TensorOutput::<EI, EO, Args>::new(&mut state);

    let query = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&query);
    let key = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&key);
    let value = VirtualTensor::<EI>::new::<TensorInput<EI, EO, Args>>(&value);
    let out = VirtualTensor::<EO, ReadWrite>::new::<TensorOutput<EI, EO, Args>>(&mut out);

    out.write(UNIT_POS, Line::cast_from(999));
}
