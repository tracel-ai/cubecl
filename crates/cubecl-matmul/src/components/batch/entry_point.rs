use crate::components::batch::CubeCountInput;
use crate::components::batch::base::BatchMatmul;
use crate::components::{
    batch::{BatchConfig, BatchMatmulFamily},
    global::args::MatmulArgs,
};
use cubecl_core as cubecl;
use cubecl_core::prelude::*;

type Input<Args, Lhs, Rhs, AccG> = <Args as MatmulArgs>::Input<Lhs, Rhs, AccG>;
type Output<Args, AccG> = <Args as MatmulArgs>::Output<AccG>;

type GlobalConf<BMMF> = <<BMMF as BatchMatmulFamily>::Config as BatchConfig>::GlobalConfig;

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
    LhsR: Numeric,
    RhsR: Numeric,
    AccR: Numeric,
    BMMF: BatchMatmulFamily,
>(
    inputs: &Input<Args, LhsG, RhsG, AccG>,
    output: &mut Output<Args, AccG>,
    cube_count_args: CubeCountInput,
    #[comptime] config: BMMF::Config,
    #[define(LhsG, RhsG, AccG)] _global: [StorageType; 3],
    #[define(LhsS, RhsS, AccS)] _stage: [StorageType; 3],
    #[define(LhsR, RhsR, AccR)] _register: [StorageType; 3],
) {
    #[allow(clippy::collapsible_if)]
    if comptime!(config.can_yield_extra_cubes()) {
        if CUBE_POS >= cube_count_args.num_valid_cubes() {
            terminate!()
        }
    }

    let mut state = Args::init_state::<LhsG, RhsG, AccG, GlobalConf<BMMF>>(
        inputs,
        output,
        config.global_config(),
    );

    BMMF::Matmul::<((LhsG, LhsS, LhsR), (RhsG, RhsS, RhsR), (AccG, AccS, AccR))>::execute::<Args>(
        &mut state,
        cube_count_args,
        config,
    );
}
