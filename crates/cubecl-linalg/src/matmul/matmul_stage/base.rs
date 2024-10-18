use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::{matmul_global::GmmConfig, Matmul};

use super::{SmmConfig, StageReader, StageWriter};

#[cube]
pub trait StageMatmul<
    I: Numeric,
    O: Numeric,
    Lhs: StageReader<I, S>,
    Rhs: StageReader<I, S>,
    Out: StageWriter<O>,
    S: SmmConfig,
>: 'static + Send + Sync + Matmul<I, O, Config = S>
{
    const M: u32;
    const N: u32;
    const K: u32;

    type Accumulator: CubeType;

    fn execute(lhs: &Lhs, rhs: &Rhs, acc: &mut Self::Accumulator, #[comptime] config: S);

    fn acc_init_zeros() -> Self::Accumulator;
    fn acc_read<G: GmmConfig>(
        acc: &Self::Accumulator,
        out: &mut Out,
        #[comptime] stage_config: S,
        #[comptime] global_config: G,
    );
}
