use crate::components::args::AttentionArgs;
use crate::components::args::TensorKey;
use crate::components::args::TensorOutput;
use crate::components::args::TensorQuery;
use crate::components::args::TensorValue;
use crate::components::batch::BatchAttentionFamily;
use crate::components::batch::CubeCountInput;
use crate::components::batch::base::BatchAttention;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;

type Input<Args, QG, KG, VG> = <Args as AttentionArgs>::Input<QG, KG, VG>;
type Output<Args, OG> = <Args as AttentionArgs>::Output<OG>;

#[cube(launch_unchecked)]
/// Launches the attention kernel
pub(crate) fn attention<
    Args: AttentionArgs,
    QG: Float,
    QT: Float,
    KG: Float,
    KS: Float,
    VG: Float,
    VS: Float,
    KVT: Float,
    SM: Float,
    ACC: Float,
    MSK: Numeric,
    OG: Float,
    OS: Float,
    BMMF: BatchAttentionFamily,
>(
    inputs: &Input<Args, QG, KG, VG>,
    output: &mut Output<Args, OG>,
    cube_count_args: CubeCountInput,
    #[comptime] config: BMMF::Config,
) {
    let mut state = Args::init_state(inputs, output);

    let query = TensorQuery::<QG, KG, VG, OG, Args>::new(&state);
    let key = TensorKey::<QG, KG, VG, OG, Args>::new(&state);
    let value = TensorValue::<QG, KG, VG, OG, Args>::new(&state);
    let mut out = TensorOutput::<QG, KG, VG, OG, Args>::new(&mut state);

    let query = VirtualTensor::<QG>::new::<TensorQuery<QG, KG, VG, OG, Args>>(&query);
    let key = VirtualTensor::<KG>::new::<TensorKey<QG, KG, VG, OG, Args>>(&key);
    let value = VirtualTensor::<VG>::new::<TensorValue<QG, KG, VG, OG, Args>>(&value);
    let out = VirtualTensor::<OG, ReadWrite>::new::<TensorOutput<QG, KG, VG, OG, Args>>(&mut out);

    BMMF::Attention::<(QG, QT, KG, KS, VG, VS, KVT, SM, ACC, MSK, OG, OS)>::execute(
        query,
        key,
        value,
        out,
        cube_count_args,
        config,
    );
}
