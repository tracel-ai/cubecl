use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::FixedShapeMatmul;

use super::{StageReader, StageWriter};

#[cube]
/// Execute a matmul over a fixed-size block, using one Cube.
pub trait StageMatmul<
    I: Numeric,
    O: Numeric,
    Lhs: StageReader<I>,
    Rhs: StageReader<I>,
    Out: StageWriter<O>,
>: 'static + Send + Sync + FixedShapeMatmul<I, O>
{
    type Accumulator: CubeType;

    fn execute(lhs: &Lhs, rhs: &Rhs, acc: &mut Self::Accumulator);

    fn acc_init_zeros() -> Self::Accumulator;
    fn acc_read(acc: &Self::Accumulator, out: &mut Out);
}
