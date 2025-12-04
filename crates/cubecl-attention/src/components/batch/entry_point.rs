use crate::components::AttentionBlueprint;
use crate::components::args::AttentionArgs;
use crate::components::args::TensorKey;
use crate::components::args::TensorMask;
use crate::components::args::TensorOutput;
use crate::components::args::TensorQuery;
use crate::components::args::TensorValue;
use crate::components::batch::BatchAttentionFamily;
use crate::components::batch::CubeCountInput;
use crate::components::batch::base::BatchAttention;
use cubecl_core as cubecl;
use cubecl_core::prelude::*;
use cubecl_std::tensor::r#virtual::VirtualTensor;
use cubecl_std::{CubeOption, CubeOptionExpand};

type Input<Args, QG, KG, VG, MSK> = <Args as AttentionArgs>::Input<QG, KG, VG, MSK>;
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
    inputs: &Input<Args, QG, KG, VG, MSK>,
    output: &mut Output<Args, OG>,
    cube_count_args: CubeCountInput,
    #[comptime] blueprint: AttentionBlueprint,
    #[define(QG, QT, KG, KS, VG, VS, KVT, SM, ACC, MSK, OG, OS)] _elem_types: [StorageType; 12],
) {
    let config = comptime!(BMMF::expand_blueprint(blueprint));
    if comptime!(config.is_err()) {
        push_validation_error(config.err().unwrap().to_string());
        comptime!(return);
    }
    let config = comptime!(config.unwrap());

    let mut state = Args::init_state(inputs, output);

    let query = TensorQuery::<QG, KG, VG, MSK, OG, Args>::new(&state);
    let query = VirtualTensor::<QG>::new::<TensorQuery<QG, KG, VG, MSK, OG, Args>>(&query);

    let key = TensorKey::<QG, KG, VG, MSK, OG, Args>::new(&state);
    let key = VirtualTensor::<KG>::new::<TensorKey<QG, KG, VG, MSK, OG, Args>>(&key);

    let value = TensorValue::<QG, KG, VG, MSK, OG, Args>::new(&state);
    let value = VirtualTensor::<VG>::new::<TensorValue<QG, KG, VG, MSK, OG, Args>>(&value);

    let has_mask = Args::has_mask(&state);
    let mask: CubeOption<VirtualTensor<MSK>> = match has_mask {
        CubeOption::Some(_) => {
            let mask = TensorMask::<QG, KG, VG, MSK, OG, Args>::new(&state);
            let mask = VirtualTensor::<MSK>::new::<TensorMask<QG, KG, VG, MSK, OG, Args>>(&mask);
            CubeOption::new_Some(mask)
        }
        CubeOption::None => CubeOption::new_None(),
    };

    let mut out = TensorOutput::<QG, KG, VG, MSK, OG, Args>::new(&mut state);
    let out =
        VirtualTensor::<OG, ReadWrite>::new::<TensorOutput<QG, KG, VG, MSK, OG, Args>>(&mut out);

    BMMF::Attention::<(QG, QT, KG, KS, VG, VS, KVT, SM, ACC, MSK, OG, OS)>::execute(
        query,
        key,
        value,
        mask,
        out,
        cube_count_args,
        config,
    );
}
