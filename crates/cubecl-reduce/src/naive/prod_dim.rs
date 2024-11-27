use cubecl_core as cubecl;
use cubecl_core::prelude::*;

use crate::ProdDim;

use super::base::ReduceDimNaive;

#[cube]
impl<EI: Numeric> ReduceDimNaive<EI> for ProdDim {
    type Accumulator = Line<EI>;

    fn initialize_naive(line_size: u32) -> Line<EI> {
        Line::empty(line_size).fill(EI::from_int(1))
    }

    fn inner_loop_naive(accumulator: &mut Self::Accumulator, current_value: Line<EI>, _i: u32) {
        *accumulator *= current_value;
    }

    fn assign_naive<EO: Numeric>(
        output: &mut Tensor<Line<EO>>,
        accumulator: Self::Accumulator,
        _shape_reduce_dim: u32,
    ) {
        output[ABSOLUTE_POS] = Line::cast_from(accumulator);
    }
}
