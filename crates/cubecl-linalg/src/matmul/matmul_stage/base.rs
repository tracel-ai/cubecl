use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::matmul::FixedShapeMatmul;

use super::{StageReader, TileWriter};

#[cube]
/// Execute a matmul over a fixed-size block, using one Cube.
pub trait StageMatmul<
    E: Numeric,
    Lhs: StageReader<E>,
    Rhs: StageReader<E>,
    Out: TileWriter<Line<E>>,
>: 'static + Send + Sync + FixedShapeMatmul<E, E>
{
    type Accumulator: CubeType;

    fn execute(lhs: &Lhs, rhs: &Rhs, acc: &mut Self::Accumulator);

    fn acc_init_zeros() -> Self::Accumulator;
    fn acc_read(acc: &Self::Accumulator, out: &mut Out);
}
