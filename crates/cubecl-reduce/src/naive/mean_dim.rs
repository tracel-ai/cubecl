use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use super::base::ReduceDimNaive;
use crate::MeanDim;

#[cube]
impl<EI: Numeric> ReduceDimNaive<EI> for MeanDim {
    type Accumulator = Line<EI>;

    fn initialize_naive(line_size: u32) -> Self::Accumulator {
        Line::empty(line_size).fill(EI::from_int(0))
    }

    fn inner_loop_naive(accumulator: &mut Self::Accumulator, current_value: Line<EI>, _i: u32) {
        *accumulator += current_value;
    }

    fn assign_naive<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: Self::Accumulator,
        shape_reduce_dim: u32,
    ) {
        output[ABSOLUTE_POS] = Line::cast_from(accumulator / Line::empty(output.line_size()).fill(EI::cast_from(shape_reduce_dim)));
    }
}
