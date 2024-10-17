use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::Matmul;

use super::{SmmConfig, StageReader, StageWriter};

#[cube]
pub trait StageMatmul<
    I: Numeric,
    O: Numeric,
    Lhs: StageReader<I>,
    Rhs: StageReader<I>,
    Out: StageWriter<O>,
>: 'static + Send + Sync + Matmul<I, O, Config: SmmConfig>
{
    const M: u32;
    const N: u32;
    const K: u32;

    type Accumulator: CubeType;

    fn execute(lhs: &Lhs, rhs: &Rhs, acc: &mut Self::Accumulator, #[comptime] config: Self::Config);

    fn acc_init_zeros() -> Self::Accumulator;
    fn acc_read(acc: &Self::Accumulator, out: &mut Out, #[comptime] config: Self::Config);
}
